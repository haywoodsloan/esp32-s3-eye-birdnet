/**
 * @file ui.c
 * @brief LVGL UI for the ESP32-S3-EYE BirdNET firmware.
 *
 * Minimal two-element layout (240x240):
 *   - scrolling spectrogram canvas (240x200), updated continuously
 *   - color-coded microphone level meter (grey/green/yellow/red by dBFS)
 *
 * Must be called with the LVGL lock held (bsp_display_lock/unlock).
 */
#include "ui.h"

#include <math.h>

#include "birdnet_config.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "lvgl.h"

static const char *TAG = "ui";

#define CANVAS_W BIRDNET_DISP_W   /* 240 */
#define CANVAS_H BIRDNET_DISP_H   /* 200 */

/* Microphone level meter: a segmented PPM-style bargraph (IEC 60268 style).
 * Sized to fill the full screen width and the strip below the spectrogram
 * (240 wide x (240 - canvas height) tall), so it sits flush with no margin. */
#define METER_W          BIRDNET_DISP_W            /* full screen width (240)   */
#define METER_H          (240 - BIRDNET_DISP_H)     /* strip below canvas (40)   */
#define METER_SEGMENTS   28
#define METER_SEG_GAP    1
#define METER_ATTACK     0.60f   /* fast rise toward a higher reading          */
#define METER_RELEASE    0.06f   /* slow fall (PPM-style return, ~25 fps)      */
#define METER_PEAK_HOLD  12      /* frames to hold the peak marker (~0.5 s)    */
#define METER_PEAK_DECAY 1.5f    /* dB/frame the peak falls after the hold     */

static uint16_t *s_canvas_buf = NULL;   /* RGB565, CANVAS_W * CANVAS_H */
static lv_obj_t *s_canvas = NULL;
static uint16_t *s_meter_buf = NULL;    /* RGB565, METER_W * METER_H   */
static lv_obj_t *s_meter = NULL;

static inline uint16_t rgb565(uint8_t r, uint8_t g, uint8_t b)
{
    return (uint16_t)(((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3));
}

/* Google "Turbo" colormap: a vivid blue->cyan->green->yellow->orange->red
 * rainbow with good perceptual range, so faint detail stays visible. */
static uint16_t colormap(float v)
{
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;

    /* 11 anchor colors sampled along the Turbo colormap. */
    static const uint8_t turbo[11][3] = {
        {48, 18, 59}, {62, 73, 184}, {54, 125, 230}, {35, 168, 224},
        {28, 201, 174}, {73, 222, 119}, {140, 228, 76}, {201, 213, 49},
        {244, 178, 45}, {251, 121, 35}, {220, 50, 20},
    };

    float fp = v * 10.0f;          /* 0 .. 10 */
    int i = (int)fp;
    if (i > 9) i = 9;
    float t = fp - (float)i;

    uint8_t r = (uint8_t)(turbo[i][0] + t * (turbo[i + 1][0] - turbo[i][0]));
    uint8_t g = (uint8_t)(turbo[i][1] + t * (turbo[i + 1][1] - turbo[i][1]));
    uint8_t b = (uint8_t)(turbo[i][2] + t * (turbo[i + 1][2] - turbo[i][2]));
    return rgb565(r, g, b);
}

static void draw_spectrogram(const float *display)
{
    if (s_canvas_buf == NULL || display == NULL) {
        return;
    }
    for (int y = 0; y < CANVAS_H; ++y) {
        uint16_t *row = &s_canvas_buf[y * CANVAS_W];
        const float *src = &display[y * BIRDNET_DISP_W];
        for (int x = 0; x < CANVAS_W; ++x) {
            row[x] = colormap(src[x]);
        }
    }
    lv_obj_invalidate(s_canvas);
}

static inline uint16_t hex565(uint32_t rgb)
{
    return rgb565((rgb >> 16) & 0xFF, (rgb >> 8) & 0xFF, rgb & 0xFF);
}

/* Fixed usability colour zones (meanings unchanged): grey = too quiet,
 * green = usable, yellow = loud/hard, red = too loud / clipping. */
static uint32_t meter_zone_color(float db)
{
    if (db < BIRDNET_LEVEL_DB_QUIET)   return 0x707070;  /* grey   */
    if (db < BIRDNET_LEVEL_DB_GOOD_HI) return 0x30C030;  /* green  */
    if (db < BIRDNET_LEVEL_DB_HOT)     return 0xE0C020;  /* yellow */
    return 0xE03030;                                     /* red    */
}

static void meter_fill(int x0, int w, uint16_t c)
{
    for (int y = 0; y < METER_H; ++y) {
        uint16_t *row = &s_meter_buf[y * METER_W];
        for (int x = x0; x < x0 + w; ++x) row[x] = c;
    }
}

/* Segmented PPM-style level meter: logarithmic dBFS scale, fast-attack /
 * slow-release ballistics, and a peak-hold marker, per IEC 60268 practice. */
static void draw_meter(float level)
{
    if (s_meter_buf == NULL) {
        return;
    }

    static float disp_db = BIRDNET_LEVEL_DB_FLOOR;
    static float peak_db = BIRDNET_LEVEL_DB_FLOOR;
    static int   peak_hold = 0;

    float db = 20.0f * log10f(level + 1e-6f);
    if (db < BIRDNET_LEVEL_DB_FLOOR) db = BIRDNET_LEVEL_DB_FLOOR;
    if (db > 0.0f) db = 0.0f;

    /* Fast attack, slow release. */
    disp_db += (db - disp_db) * ((db > disp_db) ? METER_ATTACK : METER_RELEASE);

    /* Peak hold tracks the DISPLAYED bar level (disp_db), not the raw db: the
     * bar only rises by METER_ATTACK per frame, so capturing the instantaneous
     * db would let the marker sit higher than the bar ever drew. Following
     * disp_db keeps the marker at the true high-water mark of the bar. */
    if (disp_db >= peak_db) {
        peak_db = disp_db;
        peak_hold = METER_PEAK_HOLD;
    } else if (peak_hold > 0) {
        peak_hold--;
    } else {
        peak_db -= METER_PEAK_DECAY;
        if (peak_db < disp_db) peak_db = disp_db;
    }

    const float span  = -BIRDNET_LEVEL_DB_FLOOR;

    int n_lit    = (int)(((disp_db - BIRDNET_LEVEL_DB_FLOOR) / span) * METER_SEGMENTS + 0.5f);
    int peak_seg = (int)(((peak_db - BIRDNET_LEVEL_DB_FLOOR) / span) * METER_SEGMENTS);
    int show_peak = (peak_db > BIRDNET_LEVEL_DB_FLOOR + 1.0f);
    if (n_lit < 0) n_lit = 0;
    if (n_lit > METER_SEGMENTS) n_lit = METER_SEGMENTS;
    if (peak_seg < 0) peak_seg = 0;
    if (peak_seg > METER_SEGMENTS - 1) peak_seg = METER_SEGMENTS - 1;

    for (int i = 0; i < METER_W * METER_H; ++i) {
        s_meter_buf[i] = 0x0000;
    }

    for (int s = 0; s < METER_SEGMENTS; ++s) {
        float seg_db = BIRDNET_LEVEL_DB_FLOOR + ((s + 0.5f) / METER_SEGMENTS) * span;
        /* Grey means "too quiet": only show it while the level itself is in the
         * grey zone. Once the level reaches green or above, recolor the lower
         * (would-be grey) segments green so a loud signal shows no grey base. */
        if (seg_db < BIRDNET_LEVEL_DB_QUIET && disp_db >= BIRDNET_LEVEL_DB_QUIET) {
            seg_db = BIRDNET_LEVEL_DB_QUIET;
        }
        uint16_t c;
        if (show_peak && s == peak_seg) {
            c = hex565(0xFFFFFF);                  /* peak-hold marker */
        } else if (s < n_lit) {
            c = hex565(meter_zone_color(seg_db));  /* lit segment      */
        } else {
            c = hex565(0x1E1E1E);                  /* unlit scale      */
        }
        /* Proportional segment span so they exactly fill the full meter width. */
        int x0 = (s * METER_W) / METER_SEGMENTS;
        int x1 = ((s + 1) * METER_W) / METER_SEGMENTS - METER_SEG_GAP;
        meter_fill(x0, x1 - x0, c);
    }

    lv_obj_invalidate(s_meter);
}

int ui_init(void)
{
    lv_obj_t *scr = lv_screen_active();
    lv_obj_set_style_bg_color(scr, lv_color_black(), 0);
    lv_obj_set_style_bg_opa(scr, LV_OPA_COVER, 0);
    lv_obj_set_style_pad_all(scr, 0, 0);

    s_canvas_buf = (uint16_t *)heap_caps_malloc(
        (size_t)CANVAS_W * CANVAS_H * sizeof(uint16_t),
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (s_canvas_buf == NULL) {
        s_canvas_buf = (uint16_t *)heap_caps_malloc(
            (size_t)CANVAS_W * CANVAS_H * sizeof(uint16_t), MALLOC_CAP_8BIT);
    }
    if (s_canvas_buf == NULL) {
        ESP_LOGE(TAG, "failed to allocate canvas buffer");
        return -1;
    }
    for (int i = 0; i < CANVAS_W * CANVAS_H; ++i) {
        s_canvas_buf[i] = 0;
    }

    s_canvas = lv_canvas_create(scr);
    lv_canvas_set_buffer(s_canvas, s_canvas_buf, CANVAS_W, CANVAS_H,
                         LV_COLOR_FORMAT_RGB565);
    lv_obj_align(s_canvas, LV_ALIGN_TOP_MID, 0, 0);

    /* Microphone level meter: a segmented PPM-style bargraph drawn each frame. */
    s_meter_buf = (uint16_t *)heap_caps_malloc(
        (size_t)METER_W * METER_H * sizeof(uint16_t),
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (s_meter_buf == NULL) {
        s_meter_buf = (uint16_t *)heap_caps_malloc(
            (size_t)METER_W * METER_H * sizeof(uint16_t), MALLOC_CAP_8BIT);
    }
    if (s_meter_buf == NULL) {
        ESP_LOGE(TAG, "failed to allocate meter buffer");
        return -1;
    }
    for (int i = 0; i < METER_W * METER_H; ++i) {
        s_meter_buf[i] = 0x0000;
    }

    s_meter = lv_canvas_create(scr);
    lv_canvas_set_buffer(s_meter, s_meter_buf, METER_W, METER_H,
                         LV_COLOR_FORMAT_RGB565);
    lv_obj_align(s_meter, LV_ALIGN_BOTTOM_MID, 0, 0);

    /* Raise LVGL's refresh rate to match the render loop (default is 33 ms /
     * ~30 fps, which would otherwise cap the visible frame rate). */
    lv_timer_set_period(lv_display_get_refr_timer(lv_display_get_default()),
                        BIRDNET_UI_FRAME_MS);

    ESP_LOGI(TAG, "UI initialized (%dx%d canvas, %d-seg meter)",
             CANVAS_W, CANVAS_H, METER_SEGMENTS);
    return 0;
}

void ui_update(const ui_snapshot_t *snap)
{
    if (snap == NULL) {
        return;
    }

    draw_spectrogram(snap->display);
    draw_meter(snap->level);
}
