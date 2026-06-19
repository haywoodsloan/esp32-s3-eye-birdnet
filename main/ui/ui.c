/**
 * @file ui.c
 * @brief LVGL UI for the ESP32-S3-EYE BirdNET firmware.
 *
 * Layout (240x240):
 *   - scrolling spectrogram canvas (240x200), updated continuously
 *   - color-coded microphone level meter (grey/green/yellow/red by dBFS)
 *   - detection overlay: a full-screen card (common name + SD photo +
 *     color-coded confidence bar) shown over the above for ~5 s on a detection
 *
 * Must be called with the LVGL lock held (bsp_display_lock/unlock).
 */
#include "ui.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "birdnet_config.h"
#include "bsp/esp32_s3_eye.h"
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
/* Meter ballistics: the bar uses IEC 61672 "Fast" exponential time weighting
 * (125 ms time constant) for both rise and fall, so it shows the CURRENT sound
 * level responsively. Specified in real time and converted to a per-frame step
 * from BIRDNET_UI_FRAME_MS, so it stays correct at any UI frame rate. */
#define METER_TC_MS           125.0f  /* "Fast" time-weighting time constant     */

static uint16_t *s_canvas_buf = NULL;   /* RGB565, CANVAS_W * CANVAS_H */
static lv_obj_t *s_canvas = NULL;
static uint16_t *s_meter_buf = NULL;    /* RGB565, METER_W * METER_H   */
static lv_obj_t *s_meter = NULL;

/* ---- Detection overlay -------------------------------------------------- *
 * A full-screen card shown over the spectrogram + meter for ~5 s when a bird
 * is detected: common name (title), an SD-loaded photo, and a color-coded
 * confidence bar (red = low, green = high). Built hidden in ui_init() and
 * shown by ui_update() when a new detection arrives. Custom Roboto fonts are
 * compiled in (see ui/assets/). */
LV_FONT_DECLARE(roboto_28);
LV_FONT_DECLARE(roboto_18);
LV_FONT_DECLARE(roboto_30);
LV_FONT_DECLARE(roboto_pct_32);

#define OV_IMG_SZ  232                      /* full-bleed photo size (card - border) */
#define OV_OUTLINE 2                        /* text outline thickness (px)      */
/* Name wrap width: the card's inner content width minus room for the text
 * outline on both sides, so a wrapped line plus its black outline never reaches
 * the coloured border. Derived from geometry (not the font), so changing the
 * title font size does not change the safe wrap width. */
#define OV_TITLE_W (OV_IMG_SZ - 2 * OV_OUTLINE - 4)   /* = 224 */
#define OV_BAR_W   140                      /* confidence bar width (px)        */
#define OV_BAR_H   18                       /* confidence bar thickness (px)    */
#define OV_OPA     ((lv_opa_t)255)          /* overlay text + graphics opacity (100%) */

/* An outlined text element: 8 black copies offset in all 8 directions around a
 * coloured core, giving the core a FULL black outline (legible over any photo,
 * unlike a one-sided drop shadow). The copies are created before the core so
 * they draw behind it. */
typedef struct {
    lv_obj_t *outline[8];
    lv_obj_t *core;
} ov_text_t;

static lv_obj_t      *s_ov         = NULL;  /* card container (hidden default)  */
static lv_obj_t      *s_ov_img     = NULL;  /* species photo (full-bleed, back) */
static ov_text_t      s_ov_title;           /* common name + black outline      */
static ov_text_t      s_ov_pct;             /* "87%" + black outline            */
static lv_obj_t      *s_ov_bar     = NULL;  /* confidence bar                   */
static lv_timer_t    *s_ov_timer   = NULL;  /* auto-hide one-shot               */
static uint32_t       s_ov_seen   = 0;      /* last detection seq shown         */
static uint16_t      *s_bird_buf  = NULL;   /* RGB565 pixels of the current photo */
static lv_image_dsc_t s_bird_dsc;           /* descriptor over s_bird_buf       */

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

/* Precomputed RGB565 colormap, filled once in ui_init(). draw_spectrogram does
 * CANVAS_W*CANVAS_H (~48k) lookups per frame, so a 256-entry table replaces the
 * per-pixel float interpolation above with a single indexed load. */
static uint16_t s_cmap_lut[256];

static void colormap_init(void)
{
    for (int i = 0; i < 256; ++i) {
        s_cmap_lut[i] = colormap((float)i / 255.0f);
    }
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
            int idx = (int)(src[x] * 255.0f + 0.5f);
            if (idx < 0) idx = 0;
            else if (idx > 255) idx = 255;
            row[x] = s_cmap_lut[idx];
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

/* Segmented level meter: logarithmic dBFS scale with IEC 61672 "Fast" time
 * weighting, colored by usability zone (grey/green/yellow/red). */
static void draw_meter(float level)
{
    if (s_meter_buf == NULL) {
        return;
    }

    static float disp_db = BIRDNET_LEVEL_DB_FLOOR;

    float db = 20.0f * log10f(level + 1e-6f);
    if (db < BIRDNET_LEVEL_DB_FLOOR) db = BIRDNET_LEVEL_DB_FLOOR;
    if (db > 0.0f) db = 0.0f;

    /* Per-frame "Fast" (125 ms) weighting coefficient, derived once from the
     * frame period so the response is a true 125 ms at any frame rate. */
    static float tw_coef = -1.0f;
    if (tw_coef < 0.0f) {
        tw_coef = 1.0f - expf(-(float)BIRDNET_UI_FRAME_MS / METER_TC_MS);
    }

    /* Track the current level (same time constant up & down). */
    disp_db += (db - disp_db) * tw_coef;

    const float span  = -BIRDNET_LEVEL_DB_FLOOR;

    int n_lit = (int)(((disp_db - BIRDNET_LEVEL_DB_FLOOR) / span) * METER_SEGMENTS + 0.5f);
    if (n_lit < 0) n_lit = 0;
    if (n_lit > METER_SEGMENTS) n_lit = METER_SEGMENTS;

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
        uint16_t c = (s < n_lit) ? hex565(meter_zone_color(seg_db))  /* lit segment */
                                 : hex565(0x1E1E1E);                 /* unlit scale */
        /* Proportional segment span so they exactly fill the full meter width. */
        int x0 = (s * METER_W) / METER_SEGMENTS;
        int x1 = ((s + 1) * METER_W) / METER_SEGMENTS - METER_SEG_GAP;
        meter_fill(x0, x1 - x0, c);
    }

    lv_obj_invalidate(s_meter);
}

/* ============================ Detection overlay ========================== */

/* Confidence -> color: red (low) through yellow to green (high). Detections
 * only fire in [BIRDNET_DETECT_THRESHOLD, 1.0], so that span is stretched
 * across the full red->green gradient to use its whole range. Returns 0xRRGGBB. */
static uint32_t confidence_color(float score)
{
    float t = (score - BIRDNET_DETECT_THRESHOLD) /
              (1.0f - BIRDNET_DETECT_THRESHOLD);
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;

    uint8_t r, g, b;
    if (t < 0.5f) {                 /* red -> yellow */
        float u = t / 0.5f;
        r = 0xE0;
        g = (uint8_t)(0x30 + u * (0xC8 - 0x30));
        b = 0x28;
    } else {                        /* yellow -> green */
        float u = (t - 0.5f) / 0.5f;
        r = (uint8_t)(0xE0 + u * (0x33 - 0xE0));
        g = 0xC8;
        b = (uint8_t)(0x28 + u * (0x3A - 0x28));
    }
    return ((uint32_t)r << 16) | ((uint32_t)g << 8) | b;
}

/* Build the SD path for a species photo: "<mount>/birds/<sanitized>.bin".
 * Sanitize == lowercase, every non [a-z0-9] byte -> '_' (matches tools/birds). */
static void bird_filename(const char *sci, char *out, size_t out_sz)
{
    int n = snprintf(out, out_sz, "%s%s/", BSP_SD_MOUNT_POINT, BIRDNET_BIRDS_SUBDIR);
    if (n < 0 || (size_t)n >= out_sz) {
        if (out_sz) out[0] = '\0';
        return;
    }
    size_t k = (size_t)n;
    for (const char *p = sci; *p && (k + 5) < out_sz; ++p) {
        char c = *p;
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'))) c = '_';
        out[k++] = c;
    }
    snprintf(out + k, out_sz - k, ".bin");
}

/* Load a species photo from SD into s_bird_buf and point s_bird_dsc at it.
 * Returns true on success; on a missing/malformed file it returns false and
 * leaves the previously shown image untouched. */
static bool bird_image_load(const char *sci)
{
    if (sci == NULL || sci[0] == '\0' || s_bird_buf == NULL) {
        return false;
    }

    char path[96];
    bird_filename(sci, path, sizeof(path));

    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        ESP_LOGI(TAG, "no photo for '%s' (%s)", sci, path);
        return false;
    }

    uint8_t hdr[8];
    bool ok = (fread(hdr, 1, sizeof(hdr), f) == sizeof(hdr)) &&
              hdr[0] == 'B' && hdr[1] == 'N' && hdr[2] == '1' && hdr[3] == '6';
    uint32_t w = ok ? (uint32_t)(hdr[4] | ((uint32_t)hdr[5] << 8)) : 0;
    uint32_t h = ok ? (uint32_t)(hdr[6] | ((uint32_t)hdr[7] << 8)) : 0;
    if (!ok || w == 0 || h == 0 ||
        w > BIRDNET_BIRD_IMG_MAX_PX || h > BIRDNET_BIRD_IMG_MAX_PX) {
        ESP_LOGW(TAG, "bad photo header: %s", path);
        fclose(f);
        return false;
    }

    size_t px  = (size_t)w * h;
    size_t got = fread(s_bird_buf, sizeof(uint16_t), px, f);
    fclose(f);
    if (got != px) {
        ESP_LOGW(TAG, "short photo read: %s (%u/%u px)",
                 path, (unsigned)got, (unsigned)px);
        return false;
    }

    s_bird_dsc.header.magic  = LV_IMAGE_HEADER_MAGIC;
    s_bird_dsc.header.cf     = LV_COLOR_FORMAT_RGB565;
    s_bird_dsc.header.flags  = 0;
    s_bird_dsc.header.w      = w;
    s_bird_dsc.header.h      = h;
    s_bird_dsc.header.stride = (uint16_t)(w * sizeof(uint16_t));
    s_bird_dsc.data          = (const uint8_t *)s_bird_buf;
    s_bird_dsc.data_size     = (uint32_t)(px * sizeof(uint16_t));
    return true;
}

/* Directions for an 8-neighbour text outline (full black halo, not a shadow). */
static const int8_t OV_ODX[8] = { -1,  0,  1, -1,  1, -1,  0,  1 };
static const int8_t OV_ODY[8] = { -1, -1, -1,  0,  0,  1,  1,  1 };

/* Build an outlined-text element on the card: 8 black copies (created first so
 * they sit behind) + a coloured core on top. When wrap_w > 0 every copy gets
 * that width with wrapping + centred text, so a long string breaks onto several
 * lines (identically across all copies) instead of overflowing the screen. */
static void ov_text_create(ov_text_t *t, const lv_font_t *font,
                           lv_color_t core_color, int wrap_w)
{
    for (int i = 0; i < 9; ++i) {
        lv_obj_t *l = lv_label_create(s_ov);
        lv_obj_set_style_text_font(l, font, 0);
        lv_obj_set_style_text_color(l, (i < 8) ? lv_color_black() : core_color, 0);
        lv_obj_set_style_text_opa(l, OV_OPA, 0);   /* render text at ~75% opacity */
        if (wrap_w > 0) {
            lv_obj_set_width(l, wrap_w);
            lv_label_set_long_mode(l, LV_LABEL_LONG_MODE_WRAP);
            lv_obj_set_style_text_align(l, LV_TEXT_ALIGN_CENTER, 0);
        }
        lv_label_set_text(l, "");
        if (i < 8) t->outline[i] = l;
        else       t->core = l;
    }
}

/* Align the core at (x,y) for `align`; each outline copy is offset OV_OUTLINE px
 * in its direction so the core ends up fully outlined. */
static void ov_text_align(ov_text_t *t, lv_align_t align, int x, int y)
{
    for (int i = 0; i < 8; ++i) {
        lv_obj_align(t->outline[i], align,
                     x + OV_ODX[i] * OV_OUTLINE, y + OV_ODY[i] * OV_OUTLINE);
    }
    lv_obj_align(t->core, align, x, y);
}

/* Set the same text on the core and all outline copies. */
static void ov_text_set(ov_text_t *t, const char *s)
{
    for (int i = 0; i < 8; ++i) lv_label_set_text(t->outline[i], s);
    lv_label_set_text(t->core, s);
}

static void overlay_hide_cb(lv_timer_t *t)
{
    if (s_ov) lv_obj_add_flag(s_ov, LV_OBJ_FLAG_HIDDEN);
    lv_timer_pause(t);
}

/* Populate and show the detection card; (re)start the auto-hide timer. */
static void overlay_show(const char *name, const char *sci, float score)
{
    if (s_ov == NULL) {
        return;
    }

    const char *nm = (name && name[0]) ? name : "Bird detected";
    ov_text_set(&s_ov_title, nm);

    int pct = (int)(score * 100.0f + 0.5f);
    if (pct < 0)   pct = 0;
    if (pct > 100) pct = 100;
    lv_color_t col = lv_color_hex(confidence_color(score));

    char pctbuf[8];
    snprintf(pctbuf, sizeof(pctbuf), "%d%%", pct);
    ov_text_set(&s_ov_pct, pctbuf);
    lv_obj_set_style_text_color(s_ov_pct.core, col, 0);
    lv_obj_set_style_border_color(s_ov, col, 0);
    lv_obj_set_style_bg_color(s_ov_bar, col, LV_PART_INDICATOR);
    lv_bar_set_value(s_ov_bar, pct, LV_ANIM_OFF);

    /* Species photo (best effort: hide the image if the SD file is absent). */
    if (bird_image_load(sci)) {
        /* Image cache is disabled (LV_CACHE_DEF_SIZE=0), so the decoder reads
         * s_bird_dsc directly; toggling the src via NULL forces the widget to
         * re-read the (possibly resized) header and redraw. */
        lv_image_set_src(s_ov_img, NULL);
        lv_image_set_src(s_ov_img, &s_bird_dsc);
        /* Re-assert the full-bleed size + COVER scaling: lv_image_set_src can
         * snap the object back to the source's native size, which would shrink
         * the photo. Keep it filling the card. */
        lv_obj_set_size(s_ov_img, OV_IMG_SZ, OV_IMG_SZ);
        lv_image_set_inner_align(s_ov_img, LV_IMAGE_ALIGN_COVER);
        lv_obj_remove_flag(s_ov_img, LV_OBJ_FLAG_HIDDEN);
        /* Push the photo to the BACK so the name + % + bar stay on top of it
         * (set_src can otherwise disturb the draw order). */
        lv_obj_move_background(s_ov_img);
    } else {
        lv_obj_add_flag(s_ov_img, LV_OBJ_FLAG_HIDDEN);
    }

    lv_obj_remove_flag(s_ov, LV_OBJ_FLAG_HIDDEN);
    lv_obj_move_foreground(s_ov);

    lv_timer_set_period(s_ov_timer, BIRDNET_DETECT_OVERLAY_MS);
    lv_timer_reset(s_ov_timer);
    lv_timer_resume(s_ov_timer);
}

esp_err_t ui_init(void)
{
    colormap_init();

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
        return ESP_FAIL;
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
        return ESP_FAIL;
    }
    for (int i = 0; i < METER_W * METER_H; ++i) {
        s_meter_buf[i] = 0x0000;
    }

    s_meter = lv_canvas_create(scr);
    lv_canvas_set_buffer(s_meter, s_meter_buf, METER_W, METER_H,
                         LV_COLOR_FORMAT_RGB565);
    lv_obj_align(s_meter, LV_ALIGN_BOTTOM_MID, 0, 0);

    /* ---- Detection overlay (built hidden; shown on a detection) --------- */
    s_bird_buf = (uint16_t *)heap_caps_malloc(
        (size_t)BIRDNET_BIRD_IMG_MAX_PX * BIRDNET_BIRD_IMG_MAX_PX * sizeof(uint16_t),
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (s_bird_buf == NULL) {
        ESP_LOGW(TAG, "no PSRAM for bird photo buffer; cards show no image");
    }

    s_ov = lv_obj_create(scr);
    lv_obj_remove_style_all(s_ov);
    lv_obj_set_size(s_ov, 240, 240);
    lv_obj_align(s_ov, LV_ALIGN_CENTER, 0, 0);
    lv_obj_set_style_bg_color(s_ov, lv_color_hex(0x0E0E12), 0);
    lv_obj_set_style_bg_opa(s_ov, LV_OPA_COVER, 0);
    lv_obj_set_style_radius(s_ov, 0, 0);            /* square corners          */
    lv_obj_set_style_border_width(s_ov, 4, 0);      /* 1px thicker than before */
    lv_obj_set_style_border_color(s_ov, lv_color_hex(0x30C030), 0);
    lv_obj_set_style_border_opa(s_ov, OV_OPA, 0);   /* border at ~75% opacity   */
    lv_obj_set_style_pad_all(s_ov, 0, 0);
    lv_obj_remove_flag(s_ov, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_add_flag(s_ov, LV_OBJ_FLAG_HIDDEN);

    /* Full-bleed species photo: created FIRST so it sits BEHIND the name and the
     * confidence strip. The image object fills the whole card and COVER-scales
     * the (smaller, ~130 px) source up to fill it while keeping aspect ratio, so
     * the bird is shown as large as the screen allows. */
    s_ov_img = lv_image_create(s_ov);
    lv_obj_set_size(s_ov_img, OV_IMG_SZ, OV_IMG_SZ);
    lv_obj_center(s_ov_img);
    lv_image_set_inner_align(s_ov_img, LV_IMAGE_ALIGN_COVER);
    lv_obj_add_flag(s_ov_img, LV_OBJ_FLAG_HIDDEN);

    /* Common name across the top, directly on the photo (no dark bar), with a
     * FULL black outline so it stays readable over any photo. WRAP long-mode +
     * fixed width means a long name breaks onto multiple lines instead of
     * overflowing the screen. */
    ov_text_create(&s_ov_title, &roboto_30, lv_color_white(), OV_TITLE_W);
    ov_text_align(&s_ov_title, LV_ALIGN_TOP_MID, 0, 6);

    /* Confidence bar pinned bottom-right, directly on the photo. OV_BAR_H tall
     * with fully-rounded (pill) ends. */
    s_ov_bar = lv_bar_create(s_ov);
    lv_obj_set_size(s_ov_bar, OV_BAR_W, OV_BAR_H);
    lv_obj_align(s_ov_bar, LV_ALIGN_BOTTOM_RIGHT, -8, -20);
    lv_bar_set_range(s_ov_bar, 0, 100);
    lv_obj_set_style_radius(s_ov_bar, OV_BAR_H / 2, 0);
    lv_obj_set_style_radius(s_ov_bar, OV_BAR_H / 2, LV_PART_INDICATOR);
    lv_obj_set_style_bg_color(s_ov_bar, lv_color_hex(0x202024), 0);
    lv_obj_set_style_bg_opa(s_ov_bar, LV_OPA_COVER, 0);
    lv_obj_set_style_bg_color(s_ov_bar, lv_color_hex(0x30C030), LV_PART_INDICATOR);
    lv_obj_set_style_opa(s_ov_bar, OV_OPA, 0);   /* whole bar (track+indicator) ~75% */
    lv_bar_set_value(s_ov_bar, 0, LV_ANIM_OFF);

    /* Confidence % in the large digits font (roboto_pct_32) with a full black
     * outline, right-anchored just left of the bar so it hugs it whatever the
     * digit count ("9%" vs "100%"), vertically centred on the bar (bar centre
     * at y = -20 - OV_BAR_H/2; pct line_height 22 so half is 11). Fixed anchors
     * (no align_to), so it never lands off-screen. */
    ov_text_create(&s_ov_pct, &roboto_pct_32, lv_color_white(), 0);
    ov_text_align(&s_ov_pct, LV_ALIGN_BOTTOM_RIGHT, -(8 + OV_BAR_W + 8), -18);

    s_ov_timer = lv_timer_create(overlay_hide_cb, BIRDNET_DETECT_OVERLAY_MS, NULL);
    lv_timer_pause(s_ov_timer);

    /* Raise LVGL's refresh rate to match the render loop (default is 33 ms /
     * ~30 fps, which would otherwise cap the visible frame rate). */
    lv_timer_set_period(lv_display_get_refr_timer(lv_display_get_default()),
                        BIRDNET_UI_FRAME_MS);

    ESP_LOGI(TAG, "UI initialized (%dx%d canvas, %d-seg meter)",
             CANVAS_W, CANVAS_H, METER_SEGMENTS);
    return ESP_OK;
}

void ui_update(const ui_snapshot_t *snap)
{
    if (snap == NULL) {
        return;
    }

    draw_spectrogram(snap->display);
    draw_meter(snap->level);

    /* New detection? Show the species card and (re)start its auto-hide timer.
     * det_seq is published last by the inference task, so name/sci are valid. */
    if (snap->det_seq != 0 && snap->det_seq != s_ov_seen) {
        s_ov_seen = snap->det_seq;
        overlay_show(snap->det_name, snap->det_sci, snap->det_score);
    }
}
