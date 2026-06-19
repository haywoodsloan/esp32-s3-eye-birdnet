/**
 * @file stft_frontend.c
 * @brief Block-mode linear |STFT| magnitude frontend matching birdnet-stm32.
 *
 * Replicates, on-device, the preprocessing that birdnet-stm32 performs offline
 * for the "hybrid" frontend:
 *
 *   librosa.stft(y, n_fft=512, win_length=512, window="hann",
 *                center=True, hop_length=len(y)//spec_width)
 *   S = |STFT|                      # linear magnitude
 *   S = (S - S.min()) / (S.max() - S.min() + 1e-10)   # normalize() to [0,1]
 *
 * The model itself then applies the learned mel mixer, its own max-norm and PWL
 * scaling, so small differences here are largely absorbed. We still match
 * librosa closely: Hann window, center=True via reflect padding, hop = 281.
 */
#include "stft_frontend.h"

#include <math.h>
#include <stdbool.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

#include "birdnet_config.h"
#include "esp_dsp.h"
#include "esp_heap_caps.h"
#include "esp_log.h"

static const char *TAG = "stft_frontend";

/* Reflect-padded chunk: PAD + CHUNK_SAMPLES + PAD. */
#define PADDED_LEN (BIRDNET_CHUNK_SAMPLES + 2 * BIRDNET_STFT_PAD)

static float  s_window[BIRDNET_FFT_SIZE];            /* Hann window            */
static float  s_fft[2 * BIRDNET_FFT_SIZE];           /* interleaved complex    */
static float *s_padded = NULL;                       /* PADDED_LEN floats      */

/* Pre-STFT high-pass biquad (RBJ cookbook, Butterworth Q). Coefficients are
 * built once from BIRDNET_HPF_HZ; s_hp_on is false when the cutoff is 0. */
static float s_hp_b0, s_hp_b1, s_hp_b2, s_hp_a1, s_hp_a2;
static bool  s_hp_on = false;

static void *psram_or_internal(size_t bytes)
{
    void *p = heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (p == NULL) {
        p = heap_caps_malloc(bytes, MALLOC_CAP_8BIT);
    }
    return p;
}

/* Design a 2nd-order Butterworth high-pass at BIRDNET_HPF_HZ (RBJ high-pass). */
static void design_highpass(void)
{
    if (BIRDNET_HPF_HZ <= 0) {
        s_hp_on = false;
        return;
    }
    const float f0 = (float)BIRDNET_HPF_HZ;
    const float fs = (float)BIRDNET_SAMPLE_RATE_HZ;
    const float w0 = 2.0f * (float)M_PI * f0 / fs;
    const float cw = cosf(w0);
    const float sw = sinf(w0);
    const float q  = 0.70710678f;          /* Butterworth */
    const float alpha = sw / (2.0f * q);
    const float b0 = (1.0f + cw) * 0.5f;
    const float b1 = -(1.0f + cw);
    const float b2 = (1.0f + cw) * 0.5f;
    const float a0 = 1.0f + alpha;
    const float a1 = -2.0f * cw;
    const float a2 = 1.0f - alpha;
    s_hp_b0 = b0 / a0;
    s_hp_b1 = b1 / a0;
    s_hp_b2 = b2 / a0;
    s_hp_a1 = a1 / a0;
    s_hp_a2 = a2 / a0;
    s_hp_on = true;
}

esp_err_t stft_frontend_init(void)
{
    s_padded = (float *)psram_or_internal(sizeof(float) * PADDED_LEN);
    if (s_padded == NULL) {
        ESP_LOGE(TAG, "failed to allocate padded buffer (%u floats)", (unsigned)PADDED_LEN);
        return ESP_ERR_NO_MEM;
    }

    esp_err_t err = dsps_fft2r_init_fc32(NULL, BIRDNET_FFT_SIZE);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "dsps_fft2r_init_fc32 failed: %s", esp_err_to_name(err));
        return err;
    }

    dsps_wind_hann_f32(s_window, BIRDNET_FFT_SIZE);
    design_highpass();

    ESP_LOGI(TAG, "ready: SR=%d FFT=%d bins=%d frames=%d hop=%d (%.1fs chunk)",
             BIRDNET_SAMPLE_RATE_HZ, BIRDNET_FFT_SIZE, BIRDNET_FFT_BINS,
             BIRDNET_SPEC_WIDTH, BIRDNET_STFT_HOP, (double)BIRDNET_CHUNK_SECONDS);
    return ESP_OK;
}

/* Build the reflect-padded float signal (numpy 'reflect': edge not repeated). */
static void build_padded(const int16_t *chunk)
{
    const int pad = BIRDNET_STFT_PAD;
    const int n   = BIRDNET_CHUNK_SAMPLES;

    /* Remove the per-chunk DC offset, then high-pass, before the STFT. The I2S
     * MEMS mic carries a DC bias plus strong sub-audible rumble / low-frequency
     * self-noise that the model's training data -- clean field recordings --
     * does not. Because the spectrogram is then GLOBALLY min-max normalized,
     * that low-frequency energy dominates the frame and saturates the low rows,
     * which the model reads as a low-pitched call and maps onto low-frequency
     * species (hawks/owls/woodpeckers -> "Cooper's Hawk"). Subtracting the mean
     * kills DC (FFT bin 0); the Butterworth high-pass (BIRDNET_HPF_HZ) removes
     * the rest of the rumble below the lowest bird fundamentals. This is
     * gain-invariant, which is why input-gain changes never affected it. */
    /* Exact integer accumulation: 72000 int16 samples can sum past 2^24, where
     * a float loses integer precision, and the S3 has no hardware double. */
    int64_t sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += chunk[i];
    }
    const float dc = (float)sum / (float)n;

    /* Direct Form II Transposed biquad; state resets per chunk (each 3 s chunk
     * is an independent analysis window, so the startup transient -- a few ms
     * on a zero-mean signal -- is negligible over 72000 samples). */
    float z1 = 0.0f, z2 = 0.0f;
    for (int i = 0; i < n; ++i) {
        float x = ((float)chunk[i] - dc) / 32768.0f;
        if (s_hp_on) {
            float y = s_hp_b0 * x + z1;
            z1 = s_hp_b1 * x - s_hp_a1 * y + z2;
            z2 = s_hp_b2 * x - s_hp_a2 * y;
            x = y;
        }
        s_padded[pad + i] = x;
    }
    /* Left/right reflect (mirror around the edge sample, excluding it). */
    for (int i = 0; i < pad; ++i) {
        s_padded[pad - 1 - i] = s_padded[pad + 1 + i];                 /* left  */
        s_padded[pad + n + i] = s_padded[pad + n - 2 - i];             /* right */
    }
}

void stft_frontend_process(const int16_t *chunk, float *model_input, float *display)
{
    if (chunk == NULL || model_input == NULL || s_padded == NULL) {
        return;
    }

    build_padded(chunk);

    /* STFT magnitude into model_input[bin * SPEC_WIDTH + frame]. */
    float gmin = 1e30f, gmax = -1e30f;
    for (int t = 0; t < BIRDNET_SPEC_WIDTH; ++t) {
        const int start = t * BIRDNET_STFT_HOP;

        for (int k = 0; k < BIRDNET_FFT_SIZE; ++k) {
            s_fft[2 * k]     = s_padded[start + k] * s_window[k];
            s_fft[2 * k + 1] = 0.0f;
        }
        dsps_fft2r_fc32(s_fft, BIRDNET_FFT_SIZE);
        dsps_bit_rev_fc32(s_fft, BIRDNET_FFT_SIZE);

        for (int b = 0; b < BIRDNET_FFT_BINS; ++b) {
            float re = s_fft[2 * b];
            float im = s_fft[2 * b + 1];
            float mag = sqrtf(re * re + im * im);
            model_input[b * BIRDNET_SPEC_WIDTH + t] = mag;
            if (mag < gmin) gmin = mag;
            if (mag > gmax) gmax = mag;
        }
    }

    /* Per-chunk min-max normalize to [0,1] (librosa normalize()). */
    const float inv_range = 1.0f / (gmax - gmin + 1e-10f);
    for (int i = 0; i < BIRDNET_MODEL_INPUT_LEN; ++i) {
        model_input[i] = (model_input[i] - gmin) * inv_range;
    }

    if (display == NULL) {
        return;
    }

    /* Downsample to the display canvas; row 0 = highest frequency; gamma for
     * visual contrast (does not affect the model input). */
    for (int r = 0; r < BIRDNET_DISP_H; ++r) {
        int blo = ((BIRDNET_DISP_H - 1 - r) * BIRDNET_FFT_BINS) / BIRDNET_DISP_H;
        int bhi = ((BIRDNET_DISP_H - r) * BIRDNET_FFT_BINS) / BIRDNET_DISP_H;
        if (bhi <= blo) bhi = blo + 1;

        for (int c = 0; c < BIRDNET_DISP_W; ++c) {
            int frame = (c * BIRDNET_SPEC_WIDTH) / BIRDNET_DISP_W;
            float acc = 0.0f;
            for (int b = blo; b < bhi; ++b) {
                acc += model_input[b * BIRDNET_SPEC_WIDTH + frame];
            }
            float v = acc / (float)(bhi - blo);
            display[r * BIRDNET_DISP_W + c] = powf(v, 0.45f);
        }
    }
}

/* ===========================================================================
 * Live scrolling display spectrogram
 *
 * Runs in the audio task: each BIRDNET_DISP_HOP samples produces one column,
 * stored in a ring of columns. The UI copies it out (oldest column first).
 * Auto-scales to the signal with a slow peak tracker, so quiet (low-sensitivity
 * mic) and loud sounds both render with good contrast. Only the audio task
 * calls _push (single producer), so the per-column scratch is plain static.
 * ======================================================================== */

#define DISP_COLS BIRDNET_DISP_W
#define DISP_ROWS BIRDNET_DISP_H

/*
 * Spectrogram display following standard conventions (cf. Wikipedia
 * "Spectrogram"; librosa.power_to_db with ref/top_db):
 *   - intensity is the POWER spectrum |X|^2 of the STFT, shown on a DECIBEL
 *     scale (10*log10(power)) -- audio amplitude is conventionally logarithmic;
 *   - a fixed-width dB WINDOW [floor, floor+span] is mapped to the colormap and
 *     clipped, exactly like real-time "waterfall" displays (span ~= top_db).
 *     The floor follows a slow noise-floor estimate so the view self-calibrates
 *     to the microphone while keeping CONSTANT contrast (fixed dB span);
 *   - frequency axis is LINEAR with low frequencies at the BOTTOM (the
 *     ornithology "sonogram" convention used for bird song);
 *   - time flows left -> right with the newest column at the right edge;
 *   - a perceptually ordered heat-map colormap (Turbo) renders intensity.
 */
#define DISP_DB_SPAN     70.0f    /* dB from window floor to ceiling (~top_db)  */
#define DISP_NF_MARGIN    6.0f    /* keep the noise estimate this far below floor */
#define DISP_NF_ATTACK   0.25f    /* floor tracker: descent coefficient         */
#define DISP_NF_RELEASE  0.02f    /* floor tracker: slow rise (per column)      */
#define DISP_NF_MAX_DROP  0.30f   /* floor tracker: max descent (dB) per column */

static float  d_window[BIRDNET_FFT_SIZE];     /* Hann window (display)         */
static float  d_frame[BIRDNET_FFT_SIZE];      /* overlapped analysis frame     */
static float  d_fft[2 * BIRDNET_FFT_SIZE];    /* interleaved complex (display) */
static float  d_pow[BIRDNET_FFT_BINS];        /* per-bin power scratch         */
static float  d_col[DISP_ROWS];               /* one computed column scratch   */
static float  d_nfloor = -90.0f;              /* tracked noise floor (dB)      */

static float *d_cols = NULL;                  /* ring: d_cols[col*ROWS + row]  */
static int    d_head = 0;                     /* next column write index       */
static SemaphoreHandle_t d_mtx = NULL;

esp_err_t stft_display_init(void)
{
    d_cols = (float *)psram_or_internal(sizeof(float) * DISP_COLS * DISP_ROWS);
    if (d_cols == NULL) {
        ESP_LOGE(TAG, "failed to allocate display spectrogram buffer");
        return ESP_ERR_NO_MEM;
    }
    memset(d_cols, 0, sizeof(float) * DISP_COLS * DISP_ROWS);
    memset(d_frame, 0, sizeof(d_frame));

    d_mtx = xSemaphoreCreateMutex();
    if (d_mtx == NULL) {
        return ESP_ERR_NO_MEM;
    }

    /* Shares the FFT tables initialized by stft_frontend_init(). */
    dsps_wind_hann_f32(d_window, BIRDNET_FFT_SIZE);

    ESP_LOGI(TAG, "live display ready: %dx%d, hop=%d (%d cols/read)",
             DISP_COLS, DISP_ROWS, BIRDNET_DISP_HOP,
             BIRDNET_READ_SAMPLES / BIRDNET_DISP_HOP);
    return ESP_OK;
}

/* Window + FFT the current d_frame, then reduce to one display column. */
static void compute_display_column(void)
{
    /* Remove DC (the MEMS mic has a bias) so it doesn't dominate the low rows. */
    float mean = 0.0f;
    for (int i = 0; i < BIRDNET_FFT_SIZE; ++i) {
        mean += d_frame[i];
    }
    mean /= (float)BIRDNET_FFT_SIZE;

    for (int i = 0; i < BIRDNET_FFT_SIZE; ++i) {
        d_fft[2 * i]     = (d_frame[i] - mean) * d_window[i];
        d_fft[2 * i + 1] = 0.0f;
    }
    dsps_fft2r_fc32(d_fft, BIRDNET_FFT_SIZE);
    dsps_bit_rev_fc32(d_fft, BIRDNET_FFT_SIZE);

    /* Power spectrum |X|^2; also track this column's minimum (noise estimate). */
    float colmin_pow = 1e30f;
    for (int b = 0; b < BIRDNET_FFT_BINS; ++b) {
        float re = d_fft[2 * b];
        float im = d_fft[2 * b + 1];
        float p = re * re + im * im;
        d_pow[b] = p;
        if (p < colmin_pow) colmin_pow = p;
    }

    /* Track the noise floor in dB. It rises slowly (so steady tones don't drag
     * it up) and DESCENDS in slew-limited steps: a loud transient (a snap) makes
     * a windowed impulse with deep spectral nulls -- the minimum bin momentarily
     * collapses -- and capping the per-column drop stops that from yanking the
     * floor far below the true level and leaving the view saturated ("stuck
     * red"). Genuine background changes still recalibrate in ~1 s. */
    float colmin_db = 10.0f * log10f(colmin_pow + 1e-12f);
    if (colmin_db < d_nfloor) {
        float drop = (d_nfloor - colmin_db) * DISP_NF_ATTACK;
        if (drop > DISP_NF_MAX_DROP) drop = DISP_NF_MAX_DROP;
        d_nfloor -= drop;
    } else {
        d_nfloor += (colmin_db - d_nfloor) * DISP_NF_RELEASE;
    }

    const float floor_db = d_nfloor + DISP_NF_MARGIN;
    const float inv_span = 1.0f / DISP_DB_SPAN;

    /* Each row is a frequency band: average POWER across its bins, then convert
     * to dB (the correct order) and map [floor, floor+span] -> [0,1]. Low
     * frequencies at the bottom, high at the top. */
    for (int r = 0; r < DISP_ROWS; ++r) {
        int blo = ((DISP_ROWS - 1 - r) * BIRDNET_FFT_BINS) / DISP_ROWS;  /* top = high freq */
        int bhi = ((DISP_ROWS - r) * BIRDNET_FFT_BINS) / DISP_ROWS;
        if (bhi <= blo) bhi = blo + 1;

        float acc = 0.0f;
        for (int b = blo; b < bhi; ++b) {
            acc += d_pow[b];
        }
        float db = 10.0f * log10f(acc / (float)(bhi - blo) + 1e-12f);

        float v = (db - floor_db) * inv_span;
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        d_col[r] = v;
    }
}

void stft_display_push(const int16_t *samples, size_t num_samples)
{
    if (d_cols == NULL || samples == NULL) {
        return;
    }

    const int hop = BIRDNET_DISP_HOP;
    const int keep = BIRDNET_FFT_SIZE - hop;
    size_t off = 0;

    while (off + (size_t)hop <= num_samples) {
        /* Slide the frame left by one hop and append the new samples. */
        memmove(d_frame, d_frame + hop, sizeof(float) * (size_t)keep);
        for (int i = 0; i < hop; ++i) {
            d_frame[keep + i] = (float)samples[off + i] / 32768.0f;
        }

        compute_display_column();   /* writes d_col (audio-task only, no lock) */

        /* Publish the column into the ring (brief lock vs. the UI copy). */
        xSemaphoreTake(d_mtx, portMAX_DELAY);
        memcpy(&d_cols[d_head * DISP_ROWS], d_col, sizeof(float) * DISP_ROWS);
        d_head = (d_head + 1) % DISP_COLS;
        xSemaphoreGive(d_mtx);

        off += (size_t)hop;
    }
}

void stft_display_copy(float *out_display)
{
    if (d_cols == NULL || out_display == NULL) {
        return;
    }

    xSemaphoreTake(d_mtx, portMAX_DELAY);
    const int head = d_head;
    for (int x = 0; x < DISP_COLS; ++x) {
        const float *src = &d_cols[((head + x) % DISP_COLS) * DISP_ROWS];
        for (int r = 0; r < DISP_ROWS; ++r) {
            out_display[r * DISP_COLS + x] = src[r];   /* oldest column on left */
        }
    }
    xSemaphoreGive(d_mtx);
}
