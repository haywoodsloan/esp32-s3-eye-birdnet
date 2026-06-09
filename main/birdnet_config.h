/**
 * @file birdnet_config.h
 * @brief Central tunables for the ESP32-S3-EYE BirdNET firmware.
 *
 * =============================== MODEL CONTRACT ===============================
 * This firmware targets the BirdNET-STM32 DS-CNN with the "hybrid" audio
 * frontend (https://github.com/birdnet-team/birdnet-stm32). The model is a
 * self-contained .tflite that takes a LINEAR |STFT| magnitude spectrogram and
 * internally performs: mel projection (1x1 Conv2D) -> ReLU -> per-sample
 * max-normalization -> PWL magnitude scaling -> DS-CNN -> sigmoid head.
 *
 * The firmware therefore only computes the STFT magnitude; everything else is
 * inside the model. These values MUST match the birdnet-stm32 training args
 * (their defaults, shown below) and the emitted *_model_config.json:
 *
 *   --sample_rate 24000   --fft_length 512   --spec_width 256
 *   --num_mels 64 (in-model)  --chunk_duration 3   --audio_frontend hybrid
 *   --mag_scale pwl
 *
 * librosa.stft reference: n_fft=512, win_length=512, window="hann",
 * center=True, hop_length = chunk_samples / spec_width.  Magnitude = |STFT|,
 * then min-max normalized to [0,1] per chunk (the model also self-normalizes,
 * so exact match is not critical).
 * ============================================================================
 */
#pragma once

/* ----------------------------------------------------------------------------
 * Audio capture / chunking
 * ------------------------------------------------------------------------- */
#define BIRDNET_SAMPLE_RATE_HZ      24000   /* mic sample rate (Hz)            */
#define BIRDNET_MIC_GAIN_DB         42.0f   /* digital input gain (BSP value)  */
#define BIRDNET_CHUNK_SECONDS       3       /* analysis window (s)             */
#define BIRDNET_CHUNK_SAMPLES       (BIRDNET_SAMPLE_RATE_HZ * BIRDNET_CHUNK_SECONDS) /* 72000 */
#define BIRDNET_READ_SAMPLES        1024    /* mic read block size (samples)   */

/* ----------------------------------------------------------------------------
 * STFT frontend (must match the trained model's spectrogram)
 * ------------------------------------------------------------------------- */
#define BIRDNET_FFT_SIZE            512     /* librosa n_fft / win_length      */
#define BIRDNET_SPEC_WIDTH         256     /* time frames fed to the model    */
#define BIRDNET_FFT_BINS           ((BIRDNET_FFT_SIZE / 2) + 1)   /* 257       */
#define BIRDNET_STFT_HOP           (BIRDNET_CHUNK_SAMPLES / BIRDNET_SPEC_WIDTH) /* 281 */
#define BIRDNET_STFT_PAD           (BIRDNET_FFT_SIZE / 2)         /* center pad 256 */

/* Model input element count: [1, FFT_BINS, SPEC_WIDTH, 1]. */
#define BIRDNET_MODEL_INPUT_LEN     (BIRDNET_FFT_BINS * BIRDNET_SPEC_WIDTH)

/* ----------------------------------------------------------------------------
 * Inference
 * ------------------------------------------------------------------------- */
/* Analyze the most recent 3 s every N ms (overlapping windows). */
#define BIRDNET_INFER_INTERVAL_MS   1500
#define BIRDNET_INFER_INTERVAL_SAMPLES \
    ((BIRDNET_SAMPLE_RATE_HZ * BIRDNET_INFER_INTERVAL_MS) / 1000)

/*
 * Tensor arena (PSRAM). DS-CNN at alpha=1.0 with a 257x256 input can need
 * ~1-3 MB of activations. Sized generously; the wrapper logs arena_used_bytes
 * at init so you can tune. Training at --alpha 0.5 is recommended for the S3
 * (smaller + faster, no NPU here).
 */
#define BIRDNET_TFLITE_ARENA_BYTES  (3 * 1024 * 1024)

/* sigmoid (multi-label) head: report the top class above this score. */
#define BIRDNET_DETECT_THRESHOLD    0.50f

/* Upper bound for the per-class score scratch buffer. */
#define BIRDNET_NUM_CLASSES_MAX     128

/* ----------------------------------------------------------------------------
 * Display (LCD spectrogram)
 * ------------------------------------------------------------------------- */
#define BIRDNET_DISP_W              240     /* canvas width  (px)              */
#define BIRDNET_DISP_H              200     /* canvas height (px)              */

/*
 * Microphone level meter color thresholds, in dBFS (20*log10(RMS)). The meter
 * is a segmented bargraph: FLOOR..0 dBFS maps to 0..METER_SEGMENTS lit segments
 * and each lit segment is colored by usability zone:
 *   below QUIET            -> grey   (too quiet, nothing discernible)
 *   QUIET .. GOOD_HI       -> green  (usable range)
 *   GOOD_HI .. HOT         -> yellow (loud, getting hard to analyze)
 *   above HOT              -> red    (too loud / clipping, unusable)
 *
 * FLOOR is the dBFS that reads as an empty meter. Lower it to make the meter
 * fill longer at quiet levels (so there is a visible grey run before green).
 */
#define BIRDNET_LEVEL_DB_FLOOR     (-90.0f)
#define BIRDNET_LEVEL_DB_QUIET     (-66.0f)
#define BIRDNET_LEVEL_DB_GOOD_HI   (-15.0f)
#define BIRDNET_LEVEL_DB_HOT       (-6.0f)

/*
 * Live scrolling spectrogram: the display is computed continuously (one new
 * column per BIRDNET_DISP_HOP samples) in the audio task, independently of the
 * 3 s model block, so it scrolls smoothly. Must divide BIRDNET_READ_SAMPLES.
 * 256 -> BIRDNET_READ_SAMPLES/256 = 4 new columns per mic read (~43 ms),
 * a full 240-column sweep showing ~2.6 s of history.
 */
#define BIRDNET_DISP_HOP            256

/*
 * UI frame period (ms) for the LCD render loop AND the LVGL refresh timer.
 * The display path is fully decoupled from inference (separate task, buffers
 * and STFT; detection runs on a fixed sample-based cadence at higher priority),
 * so a faster frame rate does NOT change detection speed. Ceilings: the audio
 * task emits ~94 columns/s (one per ~10.7 ms) and the 80 MHz SPI panel flushes
 * a full frame in ~11.5 ms, so ~60 fps is smooth without wasting work. Both the
 * render loop and LVGL's refresh timer use this; lowering only one would be
 * capped by the other.
 */
#define BIRDNET_UI_FRAME_MS         16      /* ~60 fps (was 40 ms / 25 fps)    */

/* ----------------------------------------------------------------------------
 * SD-backed multi-model swap (Option B)
 * ------------------------------------------------------------------------- */
/*
 * Multiple expert DS-CNN models live on the microSD card; one is resident in
 * PSRAM at a time and swapped in based on a manifest + selection (button cycle,
 * or a default at boot). All experts share the SAME hybrid frontend/contract
 * above, so only the weights, class head and labels differ between them.
 *
 * SD layout:
 *   <mount>/models/manifest.txt
 *   <mount>/models/<model>.tflite
 *   <mount>/models/<labels>.txt          (one "Common|Scientific" per line)
 *
 * BSP_SD_MOUNT_POINT comes from bsp/esp32_s3_eye.h (default "/sdcard"); these
 * macros are only expanded inside model_registry.c, which includes the BSP.
 */
#define BIRDNET_MODELS_SUBDIR       "/models"
#define BIRDNET_MANIFEST_NAME       "manifest.txt"

/* Max size of a single .tflite read from SD into the resident PSRAM buffer. */
#define BIRDNET_MODEL_MAX_BYTES     (1536 * 1024)

/* Max number of models listed in the manifest. */
#define BIRDNET_MAX_MODELS          16

/* ----------------------------------------------------------------------------
 * Model selection
 * ------------------------------------------------------------------------- */
/*
 * BIRDNET_HAS_REAL_MODEL controls only the FLASH-EMBEDDED fallback model:
 *   - If the SD card mounts and the manifest has >=1 model, SD models are used.
 *   - Else if BIRDNET_HAS_REAL_MODEL == 1, the embedded model_data.cc is used.
 *   - Else the MOCK classifier runs (full pipeline + UI still work).
 */
#ifndef BIRDNET_HAS_REAL_MODEL
#define BIRDNET_HAS_REAL_MODEL      0
#endif
