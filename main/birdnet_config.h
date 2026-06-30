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
/*
 * Extra microphone gain in dB. Default 0 -> the model is fed the mic's raw,
 * native level. A digital gain applied AFTER the ADC scales the signal and the
 * mic's self-noise together (no SNR gain) and only risks clipping loud sources;
 * since the classifier min-max normalizes every spectrogram, absolute level is
 * irrelevant to it anyway. The level gate + on-screen meter (BIRDNET_LEVEL_DB_*
 * below) are calibrated to the native level instead. A non-zero value is applied
 * as a real HARDWARE gain ahead of the ADC where the codec supports it (which
 * does improve SNR); on the S3-EYE, which has no codec gain stage, it falls back
 * to a digital gain with a saturation clamp.
 */
#define BIRDNET_MIC_GAIN_DB         0.0f
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

/*
 * High-pass cutoff (Hz) applied to the audio before the STFT. The I2S MEMS mic
 * adds strong sub-audible rumble + low-frequency self-noise (HVAC, handling,
 * 1/f) that clean training recordings do not have. Because the spectrogram is
 * globally min-max normalized, that low-frequency energy dominates the frame
 * and biases the model toward low-pitched species (hawks/owls -> "Cooper's
 * Hawk"). A 2nd-order Butterworth high-pass removes it. 200 Hz sits below the
 * lowest target-bird fundamentals (owl hoots ~300 Hz, dove coos ~400 Hz) so it
 * trims rumble without touching bird calls. Set to 0 to disable.
 */
#define BIRDNET_HPF_HZ              200


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

/*
 * Temporal voting (debounce): a detection is only published once the SAME
 * species clears BIRDNET_DETECT_THRESHOLD on this many CONSECUTIVE inferences.
 * Windows overlap every BIRDNET_INFER_INTERVAL_MS, so 2 means the call must
 * persist ~1.5 s. The model keeps a residual tendency to spike to class 0
 * ("Cooper's Hawk") on the odd near-silent frame, but those spikes oscillate
 * (e.g. 0.72 -> 0.36 -> 0.69) and never clear the threshold twice in a row, so
 * requiring 2 consecutive frames suppresses them while a real, sustained call
 * still fires. Set to 1 for the old behavior (every single-frame crossing
 * fires; more sensitive, more false positives); raise to 3 if false positives
 * persist.
 */
#define BIRDNET_DETECT_CONSECUTIVE  2

/* ----------------------------------------------------------------------------
 * Seasonal occurrence prior (location + time of year)
 * ------------------------------------------------------------------------- */
/*
 * Bias detection toward species that are actually around right now. An offline
 * table (model/occurrence_prior.h, generated by make_occurrence_prior.py from
 * BirdNET's location+time meta-model for Plainfield IN) gives each species'
 * relative occurrence for each of 48 weeks. The firmware fetches the date over
 * WiFi/SNTP (set the credentials in the "BirdNET detector" menuconfig entry),
 * works out the current week, and multiplies each class score by a soft weight
 * before thresholding:
 *
 *   norm   = prior[week][i] / max_week prior[*][i]      (0..1, 1 at peak season)
 *   weight = (1 - STRENGTH) + STRENGTH*(FLOOR + (1 - FLOOR)*norm)
 *   adjusted_score[i] = score[i] * weight
 *
 * The weight is 1.0 at a species' seasonal peak (no penalty) and falls toward
 * FLOOR out of season -- it only ever DOWN-weights, never boosts past the raw
 * score and never hard-excludes. If the time is unknown (no WiFi / not yet
 * synced) or the active model is not the 74-class set, the prior is skipped and
 * detection is unchanged. Set BIRDNET_USE_PRIOR to 0 to compile it out entirely
 * (no WiFi, no NVS, byte-for-byte the old behavior).
 */
#ifndef BIRDNET_USE_PRIOR
#define BIRDNET_USE_PRIOR           1
#endif
/* Overall effect [0..1]: 0 = no change (weight 1 everywhere), 1 = full prior.
 * Set to 1.0 after the 2026-06 field test: run10 collapses onto out-of-season
 * winter species (Red-breasted Nuthatch) on outdoor noise, and 0.5 left the
 * strong (~0.9) false hits above threshold (0.94*0.63 = 0.59 still fires). At
 * 1.0 they drop to ~0.14. Dial back toward 0.5 if a genuinely-present species
 * the meta-model under-rates ever gets suppressed. */
#define BIRDNET_PRIOR_STRENGTH      1.0f
/* Lowest weight a fully out-of-season species can receive at STRENGTH=1 [0..1]. */
#define BIRDNET_PRIOR_FLOOR         0.15f
/* Re-sync the clock over SNTP this often (hours); the prior only needs the week. */
#define BIRDNET_TIME_RESYNC_HOURS   12

/* Debug: log input spectrogram range + top-3 scoring classes each inference.
 * Set to 0 for normal operation. */
#ifndef BIRDNET_DEBUG_SCORES
#define BIRDNET_DEBUG_SCORES        1
#endif

/*
 * Diagnostic file logging: append one CSV row per inference to BIRDNET_LOG_PATH
 * on the SD card (ms, level dB, spectrogram mean/lowband, top-3 idx:score,
 * frame-hit, vote count, detected). This lets an UNTETHERED outdoor session be
 * captured to the card and analyzed later without a serial connection. Runs
 * alongside normal detection (additive, one small SD write per ~1.5 s). Set
 * back to 0 when done. Level-gated (too-quiet) frames are logged with sentinel
 * top indices (-1) so "no audio reached the model" is distinguishable from
 * "the model rejected the audio".
 */
#ifndef BIRDNET_LOG_TO_FILE
#define BIRDNET_LOG_TO_FILE         1
#endif
#define BIRDNET_LOG_PATH            "/sdcard/birdnet_log.csv"

/*
 * Audio capture alongside the file log: when set, the 3 s chunk fed to each
 * non-gated inference is saved as a 24 kHz mono WAV in BIRDNET_LOG_AUDIO_DIR,
 * and the CSV row's "clip" column names that file -- so an outdoor recording can
 * be played back and correlated to its exact scores. HEAVY on SD space
 * (~144 KB/clip every ~1.5 s); keep sessions short and set to 0 when done. Only
 * clips whose top-1 score >= BIRDNET_LOG_AUDIO_MIN_SCORE are saved (0.0 = every
 * non-gated inference; raise to keep only stronger candidates). Requires
 * BIRDNET_LOG_TO_FILE (the CSV is what the clip correlates to).
 */
#ifndef BIRDNET_LOG_AUDIO
#define BIRDNET_LOG_AUDIO           1
#endif
#if BIRDNET_LOG_AUDIO && !BIRDNET_LOG_TO_FILE
#undef BIRDNET_LOG_AUDIO
#define BIRDNET_LOG_AUDIO           0   /* audio needs the CSV to correlate to */
#endif
#define BIRDNET_LOG_AUDIO_DIR       "/sdcard/birdaudio"
#define BIRDNET_LOG_AUDIO_MIN_SCORE 0.0f

/*
 * Record mode: instead of running detection, capture the device's own audio to
 * the SD card as WAV clips (<mount>/rec/devNNN.wav, 24 kHz mono). These are used
 * as device-domain "background" negatives to fine-tune the model so it stops
 * defaulting to a bird on this microphone's room tone / self-noise. Currently
 * OFF (normal detection). To record more negatives: set this to 1, build +
 * flash, let it record, copy the clips off the card, then set back to 0.
 * Repeat sessions ACCUMULATE (each flash continues numbering after the existing
 * clips), so you can record several rooms/times without losing earlier ones.
 */
#ifndef BIRDNET_RECORD_MODE
#define BIRDNET_RECORD_MODE         0
#endif
#define BIRDNET_RECORD_SECS         10      /* length of each clip (s)             */
#define BIRDNET_RECORD_CLIPS        48      /* clips per session (48 x 10 s = 8 min) */

/* Upper bound for the per-class score scratch buffer. */
#define BIRDNET_NUM_CLASSES_MAX     128

/*
 * Detection overlay: when a species is detected, a full-screen card (common
 * name, photo, color-coded confidence) is shown over the spectrogram + meter
 * for this long. A new detection within the window restarts the timer.
 */
#define BIRDNET_DETECT_OVERLAY_MS   5000

/*
 * Overlay demo mode (verification aid). When set to 1, the firmware does NOT
 * run the mic + model; instead it cycles the detection overlay through every
 * species in the active label table, loading each one's photo from the SD card,
 * so the card + photo path can be checked end-to-end before testing real
 * detections. The confidence sweeps low->high across the species so the bar's
 * red->green gradient is exercised too. Leave at 0 for normal operation.
 */
#ifndef BIRDNET_DEMO_OVERLAY
#define BIRDNET_DEMO_OVERLAY        0
#endif

/*
 * Time each species card is shown in demo mode (ms). Keep below
 * BIRDNET_DETECT_OVERLAY_MS so the card stays up continuously (its content
 * swaps in place) instead of hiding between species.
 */
#define BIRDNET_DEMO_OVERLAY_MS     3000

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
 *
 * These are calibrated to the mic's NATIVE (un-gained) level, since
 * BIRDNET_MIC_GAIN_DB defaults to 0. If you add N dB of mic gain, raise QUIET /
 * GOOD_HI / HOT by N to keep the same zones. FLOOR stays near the int16 noise
 * floor (~-90 dBFS) regardless.
 */
#define BIRDNET_LEVEL_DB_FLOOR     (-90.0f)
#define BIRDNET_LEVEL_DB_QUIET     (-78.0f)   /* native usable floor (was -66 at +12 dB gain)          */
#define BIRDNET_LEVEL_DB_GOOD_HI   (-27.0f)   /* was -15 at +12 dB gain                                */
#define BIRDNET_LEVEL_DB_HOT       (-18.0f)   /* was  -6 at +12 dB gain; relax toward -6 once verified */

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
 * Detection overlay assets (bird photos on SD)
 * ------------------------------------------------------------------------- */
/*
 * On a detection the overlay shows a square RGB565 photo of the species, read
 * from the SD card at:  <mount>/birds/<sanitized scientific name>.bin
 * The name is sanitized by lowercasing and mapping every non [a-z0-9] byte to
 * '_' (the tools/birds converter uses the identical rule). Each file is a tiny
 * 8-byte header ("BN16", uint16 LE width, uint16 LE height) followed by
 * width*height RGB565 pixels (uint16 LE, same bit order as the UI's rgb565()).
 *
 * The converter emits images at BIRDNET_BIRD_IMG_PX; the firmware accepts any
 * size up to BIRDNET_BIRD_IMG_MAX_PX (its load buffer is sized from the max).
 *
 * The detection overlay shows the photo full-bleed (the whole 240x240 card minus
 * its 4 px border = 232 px), so images are emitted at 232 px to fill that area
 * pixel-for-pixel with no upscaling. The load buffer allows up to 240 px (the
 * full screen). At 240 px the PSRAM buffer is 240*240*2 = 112.5 KB.
 */
#define BIRDNET_BIRDS_SUBDIR        "/birds"
#define BIRDNET_BIRD_IMG_PX         232
#define BIRDNET_BIRD_IMG_MAX_PX     240

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
