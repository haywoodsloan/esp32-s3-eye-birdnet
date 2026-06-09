/**
 * @file app_main.c
 * @brief ESP32-S3-EYE BirdNET firmware entry point (block pipeline + SD swap).
 *
 * Data flow:
 *
 *   [audio task]  mic --1024 samples--> 3 s ring buffer (+ live RMS level)
 *                        every 1.5 s of new audio --> signal infer
 *
 *   [infer task]  apply any requested model swap (SD -> PSRAM -> interpreter)
 *                 snapshot last 3 s --> STFT magnitude --> classifier
 *                 --> publish resolved species strings + display
 *
 *   [main loop]   snapshot shared state --> LVGL UI (spectrogram, species, model)
 *
 * Model selection (Option B): N expert DS-CNN models live on the microSD card;
 * one is resident in PSRAM at a time. A button press cycles to the next model.
 * If the SD/manifest is unavailable, the firmware falls back to the embedded
 * model (if BIRDNET_HAS_REAL_MODEL) or the mock classifier.
 */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

#include "esp_heap_caps.h"
#include "esp_log.h"

#include "bsp/esp32_s3_eye.h"
#include "iot_button.h"

#include "birdnet_config.h"
#include "audio_capture.h"
#include "stft_frontend.h"
#include "birdnet_model.h"
#include "model_registry.h"
#include "labels.h"
#include "ui.h"

#if BIRDNET_HAS_REAL_MODEL
#include "model_data.h"
#endif

static const char *TAG = "birdnet";

#define DISP_LEN (BIRDNET_DISP_H * BIRDNET_DISP_W)

/* ---- Shared state ------------------------------------------------------- */
static SemaphoreHandle_t s_ring_mutex;    /* guards ring buffer + write idx   */
static SemaphoreHandle_t s_infer_sem;     /* signals the inference task       */

static int16_t *s_ring;                   /* CHUNK_SAMPLES int16 ring buffer  */
static uint32_t s_ring_w = 0;             /* next write index (oldest sample) */
static volatile float s_level = 0.0f;     /* latest mic RMS [0,1]             */

/* ---- Button: cycle to next model --------------------------------------- */
static button_handle_t s_btns[BSP_BUTTON_NUM];

static void on_button(void *arg, void *data)
{
    (void)arg; (void)data;
    model_registry_request_next();
}

static void setup_buttons(void)
{
    int n = 0;
    if (bsp_iot_button_create(s_btns, &n, BSP_BUTTON_NUM) != ESP_OK) {
        ESP_LOGW(TAG, "button init failed; model cycling via button disabled");
        return;
    }
    for (int i = 0; i < n; ++i) {
        iot_button_register_cb(s_btns[i], BUTTON_SINGLE_CLICK, NULL, on_button, NULL);
    }
    ESP_LOGI(TAG, "press any button to cycle models (%d buttons)", n);
}

/* ---- Audio capture task ------------------------------------------------- */
static void audio_task(void *arg)
{
    (void)arg;
    int16_t  read_buf[BIRDNET_READ_SAMPLES];
    uint32_t since_trigger = 0;

    ESP_LOGI(TAG, "audio task started");
    for (;;) {
        if (audio_capture_read(read_buf, BIRDNET_READ_SAMPLES) != ESP_OK) {
            vTaskDelay(pdMS_TO_TICKS(5));
            continue;
        }

        double sum_sq = 0.0;
        for (int i = 0; i < BIRDNET_READ_SAMPLES; ++i) {
            float s = (float)read_buf[i] / 32768.0f;
            sum_sq += (double)s * (double)s;
        }
        s_level = sqrtf((float)(sum_sq / (double)BIRDNET_READ_SAMPLES));

        xSemaphoreTake(s_ring_mutex, portMAX_DELAY);
        for (int i = 0; i < BIRDNET_READ_SAMPLES; ++i) {
            s_ring[s_ring_w] = read_buf[i];
            s_ring_w = (s_ring_w + 1) % BIRDNET_CHUNK_SAMPLES;
        }
        xSemaphoreGive(s_ring_mutex);

        /* Feed the smooth, continuously-scrolling display spectrogram. */
        stft_display_push(read_buf, BIRDNET_READ_SAMPLES);

        since_trigger += BIRDNET_READ_SAMPLES;
        if (since_trigger >= BIRDNET_INFER_INTERVAL_SAMPLES) {
            since_trigger = 0;
            xSemaphoreGive(s_infer_sem);
        }
    }
}

/* ---- Inference task ----------------------------------------------------- */
static void infer_task(void *arg)
{
    (void)arg;
    int16_t *chunk   = (int16_t *)heap_caps_malloc(
        sizeof(int16_t) * BIRDNET_CHUNK_SAMPLES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    float   *spectro = (float *)heap_caps_malloc(
        sizeof(float) * BIRDNET_MODEL_INPUT_LEN, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    static float scores[BIRDNET_NUM_CLASSES_MAX];
    assert(chunk && spectro);

    ESP_LOGI(TAG, "inference task started");
    for (;;) {
        xSemaphoreTake(s_infer_sem, portMAX_DELAY);

        /* Apply a requested model swap (SD read + interpreter rebuild). */
        int pending = model_registry_take_pending();
        if (pending >= 0) {
            model_registry_apply(pending);   /* logs the active model on success */
        }

        /* Snapshot the last 3 s in chronological order (oldest at s_ring_w). */
        xSemaphoreTake(s_ring_mutex, portMAX_DELAY);
        uint32_t w = s_ring_w;
        double sum_sq = 0.0;
        for (uint32_t i = 0; i < BIRDNET_CHUNK_SAMPLES; ++i) {
            int16_t s = s_ring[(w + i) % BIRDNET_CHUNK_SAMPLES];
            chunk[i] = s;
            float f = (float)s / 32768.0f;
            sum_sq += (double)f * (double)f;
        }
        xSemaphoreGive(s_ring_mutex);

        /* Level gate: skip inference when the 3 s chunk is below the meter's
         * "usable" floor (BIRDNET_LEVEL_DB_QUIET, the grey->green boundary).
         * The model self-normalizes, so on near-silence it would otherwise
         * classify room noise and emit confident false positives. Keeping this
         * aligned with the meter means "green = the detector is listening". */
        float chunk_rms = sqrtf((float)(sum_sq / (double)BIRDNET_CHUNK_SAMPLES));
        float chunk_db = 20.0f * log10f(chunk_rms + 1e-6f);
        if (chunk_db < BIRDNET_LEVEL_DB_QUIET) {
            continue;   /* too quiet to analyze */
        }

        /* Model uses the 3 s block; the display has its own live path. */
        stft_frontend_process(chunk, spectro, NULL);

        birdnet_result_t res = { .top_index = -1, .top_score = 0.0f, .is_mock = true };
        if (birdnet_model_run(spectro, scores, &res) != ESP_OK) {
            continue;
        }
        bool detected = (res.top_index >= 0) && (res.top_score >= BIRDNET_DETECT_THRESHOLD);

        /* Detections go to the serial log (the screen shows only the
         * spectrogram + level meter). */
        if (detected) {
            ESP_LOGI(TAG, "%s%s  (%.0f%%)",
                     res.is_mock ? "[mock] " : "", birdnet_label_name(res.top_index),
                     (double)(res.top_score * 100.0f));
        }
    }
}

/* ---- Model selection at boot: SD -> embedded -> mock -------------------- */
static void select_initial_model(void)
{
    if (model_registry_init() == ESP_OK && model_registry_count() > 0) {
        int def = model_registry_default_index();
        if (model_registry_apply(def) == ESP_OK) {
            if (model_registry_count() > 1) {
                setup_buttons();
            }
            ESP_LOGI(TAG, "using SD model '%s' (%d available)",
                     model_registry_active_name(), model_registry_count());
            return;
        }
        ESP_LOGW(TAG, "default SD model failed to load; falling back");
    }

#if BIRDNET_HAS_REAL_MODEL
    if (birdnet_model_set(birdnet_model_tflite, birdnet_model_tflite_len) == ESP_OK) {
        ESP_LOGI(TAG, "using embedded model");
        return;
    }
    ESP_LOGW(TAG, "embedded model failed to load; falling back to mock");
#endif

    ESP_LOGW(TAG, "no model available - MOCK classifier active");
}

void app_main(void)
{
    ESP_LOGI(TAG, "ESP32-S3-EYE BirdNET starting (stm32 DS-CNN, SD multi-model)");

    s_ring_mutex  = xSemaphoreCreateMutex();
    s_infer_sem   = xSemaphoreCreateBinary();
    s_ring = (int16_t *)heap_caps_calloc(BIRDNET_CHUNK_SAMPLES, sizeof(int16_t),
                                         MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    assert(s_ring_mutex && s_infer_sem && s_ring);

    /* Display + UI. Pin the LVGL flush task to core 0 so panel rendering never
     * contends with audio capture + inference (both on core 1). Otherwise the
     * floating LVGL task can land on the compute core and stall mid-frame during
     * a 3 s inference, which shows as periodic stutter. This mirrors
     * bsp_display_start()'s default buffer config, changing only the affinity. */
    bsp_display_cfg_t disp_cfg = {
        .lvgl_port_cfg = ESP_LVGL_PORT_INIT_CONFIG(),
        .buffer_size   = BSP_LCD_H_RES * CONFIG_BSP_LCD_DRAW_BUF_HEIGHT,
        .double_buffer = CONFIG_BSP_LCD_DRAW_BUF_DOUBLE ? 1 : 0,
        .flags = {
            .buff_dma    = true,
            .buff_spiram = false,
            .sw_rotate   = false,
        },
    };
    disp_cfg.lvgl_port_cfg.task_affinity = 0;   /* pin LVGL flush to core 0 */
    bsp_display_start_with_config(&disp_cfg);
    bsp_display_backlight_on();
    bsp_display_lock(0);
    ui_init();
    bsp_display_unlock();

    /* DSP + classifier scaffolding (arena + op resolver; no model yet). */
    ESP_ERROR_CHECK(stft_frontend_init());
    ESP_ERROR_CHECK(stft_display_init());
    ESP_ERROR_CHECK(birdnet_model_init());
    birdnet_labels_reset_default();        /* labels for mock / fallback */

    /* Pick the model: SD manifest -> embedded -> mock. */
    select_initial_model();

    /* Audio last, so the first inference has a model + valid chunk forming. */
    ESP_ERROR_CHECK(audio_capture_init());

    /* Core split: audio capture (prio 6) + inference (prio 5) on core 1; the UI
     * render loop (this task) + LVGL flush run on core 0. Audio outranks infer
     * so it is never delayed, and inference no longer steals time from the
     * display -> smooth scrolling even while a 3 s chunk is being classified. */
    xTaskCreatePinnedToCore(audio_task, "audio", 6144, NULL, 6, NULL, 1);
    xTaskCreatePinnedToCore(infer_task, "infer", 8192, NULL, 5, NULL, 1);

    /* UI snapshot scratch (main-loop owned; safe to pass to ui_update). */
    float *snap_display = (float *)heap_caps_malloc(sizeof(float) * DISP_LEN,
                                                    MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    assert(snap_display);

    for (;;) {
        ui_snapshot_t snap;
        /* Live scrolling spectrogram (its own lock; updated every mic read). */
        stft_display_copy(snap_display);
        snap.display = snap_display;
        snap.level   = s_level;

        bsp_display_lock(0);
        ui_update(&snap);
        bsp_display_unlock();

        vTaskDelay(pdMS_TO_TICKS(BIRDNET_UI_FRAME_MS));
    }
}
