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
#include <sys/stat.h>

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

/* Detection -> UI overlay handoff (inference task -> main render loop). The
 * inference task is also the sole writer of the labels table, so it captures
 * the species names + score here, then bumps s_det_seq LAST as the publish
 * signal. Detections are >= BIRDNET_INFER_INTERVAL_MS (1.5 s) apart, far longer
 * than a 16 ms UI frame, so the buffers are stable by the time the main loop
 * forwards them to the overlay. */
static volatile uint32_t s_det_seq = 0;
static char              s_det_name[64];
static char              s_det_sci[64];
static volatile float    s_det_score = 0.0f;

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

        /* ESP32-S3 has a single-precision FPU only; double math is software-
         * emulated. The samples are normalized to [-1,1], so a float sum is
         * amply precise for level metering and far cheaper. */
        float sum_sq = 0.0f;
        for (int i = 0; i < BIRDNET_READ_SAMPLES; ++i) {
            float s = (float)read_buf[i] / 32768.0f;
            sum_sq += s * s;
        }
        s_level = sqrtf(sum_sq / (float)BIRDNET_READ_SAMPLES);

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

    /* Temporal voting state (see BIRDNET_DETECT_CONSECUTIVE): the candidate
     * species and how many back-to-back inferences it has cleared the score
     * threshold. A detection is only published once the count reaches the
     * required streak, which rejects one-off near-silence spikes. */
    int vote_index = -1;
    int vote_count = 0;

    for (;;) {
        xSemaphoreTake(s_infer_sem, portMAX_DELAY);

        /* Yield the core briefly so the CPU-1 IDLE task can run and feed the
         * task watchdog. A single inference on the full (alpha=1.0) model takes
         * several seconds -- longer than the BIRDNET_INFER_INTERVAL_MS trigger
         * cadence -- so the trigger semaphore is essentially always signalled
         * and this task would otherwise loop back-to-back with no gap, starving
         * IDLE1 and tripping the watchdog. This short sleep costs nothing next
         * to multi-second inference but guarantees the idle task gets a slice. */
        vTaskDelay(pdMS_TO_TICKS(20));

        /* Apply a requested model swap (SD read + interpreter rebuild). */
        int pending = model_registry_take_pending();
        if (pending >= 0) {
            model_registry_apply(pending);   /* logs the active model on success */
        }

        /* Snapshot the last 3 s in chronological order (oldest at s_ring_w).
         * The ring is contiguous in two spans (w..end, then 0..w), so copy each
         * with memcpy instead of a per-sample modulo across 72000 samples, and
         * keep the critical section to just the copy. */
        xSemaphoreTake(s_ring_mutex, portMAX_DELAY);
        uint32_t w = s_ring_w;
        uint32_t first = BIRDNET_CHUNK_SAMPLES - w;
        memcpy(chunk, &s_ring[w], first * sizeof(int16_t));
        if (w) {
            memcpy(chunk + first, s_ring, w * sizeof(int16_t));
        }
        xSemaphoreGive(s_ring_mutex);

        float sum_sq = 0.0f;
        for (uint32_t i = 0; i < BIRDNET_CHUNK_SAMPLES; ++i) {
            float f = (float)chunk[i] / 32768.0f;
            sum_sq += f * f;
        }

        /* Level gate: skip inference when the 3 s chunk is below the meter's
         * "usable" floor (BIRDNET_LEVEL_DB_QUIET, the grey->green boundary).
         * The model self-normalizes, so on near-silence it would otherwise
         * classify room noise and emit confident false positives. Keeping this
         * aligned with the meter means "green = the detector is listening". */
        float chunk_rms = sqrtf(sum_sq / (float)BIRDNET_CHUNK_SAMPLES);
        float chunk_db = 20.0f * log10f(chunk_rms + 1e-6f);
        if (chunk_db < BIRDNET_LEVEL_DB_QUIET) {
            vote_index = -1;   /* silence breaks the detection streak */
            vote_count = 0;
            continue;   /* too quiet to analyze */
        }

        /* Model uses the 3 s block; the display has its own live path. */
        stft_frontend_process(chunk, spectro, NULL);

        birdnet_result_t res = { .top_index = -1, .top_score = 0.0f, .is_mock = true };
        if (birdnet_model_run(spectro, scores, &res) != ESP_OK) {
            continue;
        }

#if BIRDNET_DEBUG_SCORES
        /* Diagnostic: log the input spectrogram range + the top-3 scoring
         * classes every inference. A healthy classifier shows the top class
         * varying with the audio and a clear margin; a stuck top-1 (always the
         * same class, e.g. index 0) with near-equal runners-up points at a
         * preprocessing/contract mismatch rather than the audio. */
        {
            float smin = 1e30f, smax = -1e30f; double smean = 0.0;
            for (int i = 0; i < BIRDNET_MODEL_INPUT_LEN; ++i) {
                float v = spectro[i];
                if (v < smin) smin = v;
                if (v > smax) smax = v;
                smean += v;
            }
            smean /= (double)BIRDNET_MODEL_INPUT_LEN;
            /* Mean of the lowest 9 freq bins (DC..~420 Hz). If this is much
             * larger than the overall mean, low-frequency/DC energy dominates
             * the spectrogram (the "always a low-pitched bird" failure). */
            double lowband = 0.0;
            for (int b = 0; b < 9; ++b) {
                for (int t = 0; t < BIRDNET_SPEC_WIDTH; ++t) {
                    lowband += spectro[b * BIRDNET_SPEC_WIDTH + t];
                }
            }
            lowband /= (double)(9 * BIRDNET_SPEC_WIDTH);
            int b0 = -1, b1 = -1, b2 = -1;
            float v0 = -1e30f, v1 = -1e30f, v2 = -1e30f;
            for (int i = 0; i < birdnet_num_labels; ++i) {
                float v = scores[i];
                if (v > v0)      { b2 = b1; v2 = v1; b1 = b0; v1 = v0; b0 = i; v0 = v; }
                else if (v > v1) { b2 = b1; v2 = v1; b1 = i; v1 = v; }
                else if (v > v2) { b2 = i; v2 = v; }
            }
            ESP_LOGI(TAG, "spectro[mean=%.3f lowband=%.3f] top3: %d:%s=%.2f  %d:%s=%.2f  %d:%s=%.2f",
                     (double)smean, lowband,
                     b0, birdnet_label_name(b0), (double)v0,
                     b1, birdnet_label_name(b1), (double)v1,
                     b2, birdnet_label_name(b2), (double)v2);
        }
#endif

        /* Per-frame candidate: the top class, but only when it clears the score
         * threshold. Random near-silence spikes (the residual class-0 "Cooper's
         * Hawk" attractor) cross the threshold on a single frame and fall back
         * below it on the next, so they never accumulate a streak; a real call
         * that persists across the overlapping 3 s windows does. */
        bool frame_hit = (res.top_index >= 0) &&
                         (res.top_score >= BIRDNET_DETECT_THRESHOLD);
        if (frame_hit && res.top_index == vote_index) {
            vote_count++;
        } else if (frame_hit) {
            vote_index = res.top_index;
            vote_count = 1;
        } else {
            vote_index = -1;
            vote_count = 0;
        }
        bool detected = (vote_count >= BIRDNET_DETECT_CONSECUTIVE);

#if BIRDNET_DEBUG_SCORES
        if (frame_hit && !detected) {
            ESP_LOGI(TAG, "candidate %s (%.0f%%) %d/%d -- waiting for confirmation",
                     birdnet_label_name(res.top_index),
                     (double)(res.top_score * 100.0f),
                     vote_count, BIRDNET_DETECT_CONSECUTIVE);
        }
#endif

        /* Detections go to the serial log AND drive the on-screen overlay. */
        if (detected) {
            ESP_LOGI(TAG, "%s%s  (%.0f%%)",
                     res.is_mock ? "[mock] " : "", birdnet_label_name(res.top_index),
                     (double)(res.top_score * 100.0f));

            /* Publish to the UI overlay. Names are captured here (this task
             * owns the labels table); seq is bumped last so the main loop
             * never reads a half-written name. */
            snprintf(s_det_name, sizeof(s_det_name), "%s",
                     birdnet_label_name(res.top_index));
            snprintf(s_det_sci, sizeof(s_det_sci), "%s",
                     birdnet_label_sci(res.top_index));
            s_det_score = res.top_score;
            s_det_seq++;
        }
    }
}

#if BIRDNET_DEMO_OVERLAY
/* ---- Demo: slideshow the detection overlay through every species -------- *
 * Publishes a synthetic detection for each label in the active table in turn,
 * so the detection card -- title, the SD-loaded species photo, and the
 * color-coded confidence bar -- can be verified end-to-end without the mic or
 * model. Confidence sweeps across the detect..1.0 range so the bar's red->green
 * gradient is exercised too. Drives the exact same publish path as a real
 * detection. Enabled by BIRDNET_DEMO_OVERLAY (see birdnet_config.h). */
static void demo_task(void *arg)
{
    (void)arg;
    ESP_LOGI(TAG, "DEMO overlay: cycling %d species every %d ms (mic + model idle)",
             birdnet_num_labels, BIRDNET_DEMO_OVERLAY_MS);
    int i = 0;
    for (;;) {
        int n = birdnet_num_labels;
        if (n > 0) {
            if (i >= n) i = 0;
            float frac = (n > 1) ? (float)i / (float)(n - 1) : 1.0f;
            s_det_score = BIRDNET_DETECT_THRESHOLD +
                          frac * (1.0f - BIRDNET_DETECT_THRESHOLD);
            snprintf(s_det_name, sizeof(s_det_name), "%s", birdnet_label_name(i));
            snprintf(s_det_sci,  sizeof(s_det_sci),  "%s", birdnet_label_sci(i));
            s_det_seq++;
            ESP_LOGI(TAG, "[demo] %d/%d %s (%s) %.0f%%", i + 1, n,
                     birdnet_label_name(i), birdnet_label_sci(i),
                     (double)(s_det_score * 100.0f));
            ++i;
        }
        vTaskDelay(pdMS_TO_TICKS(BIRDNET_DEMO_OVERLAY_MS));
    }
}
#endif

#if BIRDNET_RECORD_MODE
/* ---- Record mode: capture device audio to SD for training negatives -------
 * Writes BIRDNET_RECORD_CLIPS WAV clips of ~BIRDNET_RECORD_SECS seconds each to
 * <mount>/rec/devNNN.wav at the model's native 24 kHz mono. These device-domain
 * recordings (room tone, speech, HVAC -- captured through THIS microphone) are
 * added to the training "background" class and the model is fine-tuned, so it
 * stops defaulting to a bird on the device's own noise floor. The mic + model
 * are otherwise idle. Enabled by BIRDNET_RECORD_MODE (see birdnet_config.h). */

/* Write a 44-byte canonical PCM WAV header for a known sample count. ESP32-S3 is
 * little-endian and WAV is little-endian, so the integer fields copy directly. */
static void wav_write_header(FILE *f, uint32_t num_samples)
{
    const uint32_t sr         = BIRDNET_SAMPLE_RATE_HZ;
    const uint16_t ch         = 1;
    const uint16_t bits       = 16;
    const uint16_t block_algn = (uint16_t)(ch * (bits / 8));
    const uint32_t byte_rate  = sr * block_algn;
    const uint32_t data_bytes = num_samples * block_algn;
    const uint32_t riff_size  = 36 + data_bytes;
    const uint32_t fmt_size   = 16;
    const uint16_t fmt_pcm    = 1;
    uint8_t h[44];
    memcpy(h + 0,  "RIFF", 4);   memcpy(h + 4,  &riff_size, 4);
    memcpy(h + 8,  "WAVE", 4);
    memcpy(h + 12, "fmt ", 4);   memcpy(h + 16, &fmt_size, 4);
    memcpy(h + 20, &fmt_pcm, 2); memcpy(h + 22, &ch, 2);
    memcpy(h + 24, &sr, 4);      memcpy(h + 28, &byte_rate, 4);
    memcpy(h + 32, &block_algn, 2); memcpy(h + 34, &bits, 2);
    memcpy(h + 36, "data", 4);   memcpy(h + 40, &data_bytes, 4);
    fwrite(h, 1, sizeof(h), f);
}

static void record_task(void *arg)
{
    (void)arg;
    /* Whole number of mic reads per clip, so every read is a full block (no
     * partial last block to special-case for the file or the display feed). */
    const uint32_t blocks =
        ((uint32_t)BIRDNET_RECORD_SECS * BIRDNET_SAMPLE_RATE_HZ +
         BIRDNET_READ_SAMPLES - 1) / BIRDNET_READ_SAMPLES;
    const uint32_t target = blocks * BIRDNET_READ_SAMPLES;
    static int16_t buf[BIRDNET_READ_SAMPLES];

    char dir[64];
    snprintf(dir, sizeof(dir), "%s/rec", BSP_SD_MOUNT_POINT);
    mkdir(dir, 0777);   /* ignore EEXIST: the dir may already exist */

    /* Resume numbering after any existing clips so repeat sessions (different
     * rooms/times across reflashes) ACCUMULATE instead of overwriting. Probe
     * dev000, dev001, ... and start at the first free index. */
    int start = 0;
    for (;;) {
        char probe[128];
        struct stat st;
        snprintf(probe, sizeof(probe), "%s/dev%03d.wav", dir, start);
        if (stat(probe, &st) != 0) {
            break;   /* first missing index -> next free slot */
        }
        ++start;
    }

    ESP_LOGW(TAG, "RECORD MODE: %d clips x ~%d s -> %s/devNNN.wav (24 kHz mono), "
             "continuing from dev%03d (existing clips kept)",
             BIRDNET_RECORD_CLIPS, BIRDNET_RECORD_SECS, dir, start);

    for (int k = 0; k < BIRDNET_RECORD_CLIPS; ++k) {
        const int c = start + k;
        char path[128];
        snprintf(path, sizeof(path), "%s/dev%03d.wav", dir, c);
        FILE *f = fopen(path, "wb");
        if (f == NULL) {
            ESP_LOGE(TAG, "cannot open %s for write (is the SD card present/writable?)", path);
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }
        wav_write_header(f, target);

        for (uint32_t b = 0; b < blocks; ++b) {
            if (audio_capture_read(buf, BIRDNET_READ_SAMPLES) != ESP_OK) {
                vTaskDelay(pdMS_TO_TICKS(5));
                continue;
            }
            fwrite(buf, sizeof(int16_t), BIRDNET_READ_SAMPLES, f);

            /* Drive the meter + live spectrogram so the screen confirms capture
             * (mirrors the normal audio task's display feed). */
            double sum_sq = 0.0;
            for (int i = 0; i < BIRDNET_READ_SAMPLES; ++i) {
                float s = (float)buf[i] / 32768.0f;
                sum_sq += (double)s * (double)s;
            }
            s_level = sqrtf((float)(sum_sq / (double)BIRDNET_READ_SAMPLES));
            stft_display_push(buf, BIRDNET_READ_SAMPLES);
        }
        fclose(f);
        ESP_LOGW(TAG, "recorded %s  (%d/%d this session)", path, k + 1, BIRDNET_RECORD_CLIPS);
    }

    ESP_LOGW(TAG, "RECORD MODE complete: %d new clips in %s (now dev000..dev%03d) -- "
             "copy them off the SD card. Re-flash to record MORE (they accumulate), "
             "or set BIRDNET_RECORD_MODE back to 0 for normal detection.",
             BIRDNET_RECORD_CLIPS, dir, start + BIRDNET_RECORD_CLIPS - 1);
    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(1000));   /* idle until the user reflashes */
    }
}
#endif

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
    ESP_ERROR_CHECK(ui_init());
    bsp_display_unlock();

    /* DSP + classifier scaffolding (arena + op resolver; no model yet). */
    ESP_ERROR_CHECK(stft_frontend_init());
    ESP_ERROR_CHECK(stft_display_init());
    ESP_ERROR_CHECK(birdnet_model_init());
    birdnet_labels_reset_default();        /* labels for mock / fallback */

    /* Pick the model: SD manifest -> embedded -> mock. */
    select_initial_model();

#if BIRDNET_RECORD_MODE
    /* Record mode: capture device audio to the SD card (mounted above by the
     * model registry) as WAV negatives, instead of running detection. audio +
     * infer tasks are intentionally idle. */
    (void)audio_task;
    (void)infer_task;
    ESP_ERROR_CHECK(audio_capture_init());
    xTaskCreatePinnedToCore(record_task, "record", 6144, NULL, 6, NULL, 1);
#elif BIRDNET_DEMO_OVERLAY
    /* Demo mode: cycle the detection overlay through every species (each with
     * its SD photo) instead of running the mic + model, so the photo/overlay
     * path can be verified before testing real detections. audio_task and
     * infer_task are intentionally left idle here. */
    (void)audio_task;   /* defined but unused while demoing */
    (void)infer_task;
    xTaskCreatePinnedToCore(demo_task, "demo", 4096, NULL, 5, NULL, 1);
#else
    /* Audio last, so the first inference has a model + valid chunk forming. */
    ESP_ERROR_CHECK(audio_capture_init());

    /* Core split: audio capture (prio 6) + inference (prio 5) on core 1; the UI
     * render loop (this task) + LVGL flush run on core 0. Audio outranks infer
     * so it is never delayed, and inference no longer steals time from the
     * display -> smooth scrolling even while a 3 s chunk is being classified. */
    xTaskCreatePinnedToCore(audio_task, "audio", 6144, NULL, 6, NULL, 1);
    xTaskCreatePinnedToCore(infer_task, "infer", 8192, NULL, 5, NULL, 1);
#endif

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

        /* Detection overlay event (ui tracks det_seq to fire once per bump). */
        snap.det_seq   = s_det_seq;
        snap.det_name  = s_det_name;
        snap.det_sci   = s_det_sci;
        snap.det_score = s_det_score;

        bsp_display_lock(0);
        ui_update(&snap);
        bsp_display_unlock();

        vTaskDelay(pdMS_TO_TICKS(BIRDNET_UI_FRAME_MS));
    }
}
