/**
 * @file audio_capture.c
 * @brief I2S MEMS microphone capture for the ESP32-S3-EYE via the BSP.
 *
 * The ESP32-S3-EYE has a single digital I2S MEMS microphone wired to:
 *   BCLK = GPIO41, WS = GPIO42, DIN = GPIO2  (see bsp/esp32_s3_eye.h)
 *
 * The BSP sets up the I2S channel (bsp_audio_init) and wraps the mic as an
 * esp_codec_dev handle (bsp_audio_codec_microphone_init); we then use the
 * esp_codec_dev read API to pull PCM frames.
 */
#include "audio_capture.h"

#include <math.h>
#include <string.h>

#include "birdnet_config.h"

#include "bsp/esp32_s3_eye.h"
#include "esp_codec_dev.h"
#include "esp_log.h"

static const char *TAG = "audio_capture";

static esp_codec_dev_handle_t s_mic = NULL;

/*
 * The S3-EYE I2S MEMS mic returns data that, decimated naively, leaves a
 * spectral image mirrored around fs/4: any real or repeated content above the
 * 12 kHz half-band folds back on a plain 2:1 drop (visible as a second line
 * that moves opposite the real one). To remove it properly we OVERSAMPLE --
 * open the mic at 2x the target rate (48 kHz) and decimate to 24 kHz through a
 * linear-phase low-pass FIR, instead of dropping every other sample. The FIR
 * attenuates everything above ~12 kHz so nothing can alias into the band,
 * which kills the mirror regardless of whether the source is sample
 * duplication or genuine high-frequency energy.
 */
#define MIC_OVERSAMPLE      2
#define MIC_OPEN_RATE_HZ    (BIRDNET_SAMPLE_RATE_HZ * MIC_OVERSAMPLE)

/* Number of decimated read blocks discarded at startup (mic settle + DMA flush). */
#define MIC_PRIME_READS     5

/* Anti-alias decimation FIR: Hamming-windowed sinc, ~10.5 kHz cutoff at 48 kHz.
 * Odd tap count -> linear phase (group delay = (DECIM_TAPS-1)/2 input samples). */
#define DECIM_TAPS          63
#define DECIM_FC_HZ         10500

static float   s_decim_h[DECIM_TAPS];               /* filter coefficients      */
static int16_t s_decim_hist[DECIM_TAPS - 1];        /* carry-over between reads  */
/* Working buffer: overlap history + one oversampled read block (single-producer). */
static int16_t s_work[(DECIM_TAPS - 1) + BIRDNET_READ_SAMPLES * MIC_OVERSAMPLE];

/* Build the Hamming-windowed-sinc low-pass used for 2:1 anti-alias decimation. */
static void design_decimator(void)
{
    const float fc = (float)DECIM_FC_HZ / (float)MIC_OPEN_RATE_HZ;  /* 0..0.5 */
    const int   M  = DECIM_TAPS - 1;
    float sum = 0.0f;
    for (int k = 0; k < DECIM_TAPS; ++k) {
        float n    = (float)k - (float)M / 2.0f;
        float sinc = (n == 0.0f) ? (2.0f * fc)
                                 : sinf(2.0f * (float)M_PI * fc * n) / ((float)M_PI * n);
        float win  = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * (float)k / (float)M);
        s_decim_h[k] = sinc * win;
        sum += s_decim_h[k];
    }
    for (int k = 0; k < DECIM_TAPS; ++k) {
        s_decim_h[k] /= sum;          /* unity DC gain -> preserve signal level */
    }
}

esp_err_t audio_capture_init(void)
{
    /* Bring up the I2S peripheral with the BSP defaults. */
    esp_err_t err = bsp_audio_init(NULL);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "bsp_audio_init failed: %s", esp_err_to_name(err));
        return err;
    }

    s_mic = bsp_audio_codec_microphone_init();
    if (s_mic == NULL) {
        ESP_LOGE(TAG, "bsp_audio_codec_microphone_init returned NULL");
        return ESP_FAIL;
    }

    /* Apply input gain (digital). Non-fatal if unsupported. */
    int rc = esp_codec_dev_set_in_gain(s_mic, BIRDNET_MIC_GAIN_DB);
    if (rc != ESP_CODEC_DEV_OK) {
        ESP_LOGW(TAG, "set_in_gain rc=%d (continuing)", rc);
    }

    esp_codec_dev_sample_info_t fs = {
        .bits_per_sample = 16,
        .channel         = 1,
        .channel_mask    = 0,
        .sample_rate     = MIC_OPEN_RATE_HZ,
        .mclk_multiple   = 0,
    };

    rc = esp_codec_dev_open(s_mic, &fs);
    if (rc != ESP_CODEC_DEV_OK) {
        ESP_LOGE(TAG, "esp_codec_dev_open failed rc=%d", rc);
        return ESP_FAIL;
    }

    design_decimator();

    /* Prime the capture: discard the first reads so (1) the MEMS mic's power-on
     * settling transient never reaches the display, (2) any stale I2S DMA is
     * flushed, and (3) the decimation FIR starts with valid history instead of
     * zeros. ~5 reads of 1024 decimated samples ~= 200 ms at 24 kHz. */
    static int16_t prime[BIRDNET_READ_SAMPLES];
    for (int i = 0; i < MIC_PRIME_READS; ++i) {
        if (audio_capture_read(prime, BIRDNET_READ_SAMPLES) != ESP_OK) {
            break;
        }
    }

    ESP_LOGI(TAG, "microphone ready: %d Hz capture -> %d Hz (FIR decimate %dx, %d taps), 16-bit mono, gain %.0f dB",
             MIC_OPEN_RATE_HZ, BIRDNET_SAMPLE_RATE_HZ, MIC_OVERSAMPLE, DECIM_TAPS, (double)BIRDNET_MIC_GAIN_DB);
    return ESP_OK;
}

esp_err_t audio_capture_read(int16_t *dst, size_t num_samples)
{
    if (s_mic == NULL || dst == NULL) {
        return ESP_ERR_INVALID_STATE;
    }

    const int H = DECIM_TAPS - 1;   /* overlap / history length */

    /* Produce decimated samples in blocks. Each output needs MIC_OVERSAMPLE raw
     * input samples; we read 2x and low-pass + decimate through the FIR so no
     * energy above the 12 kHz half-band can fold back (removes the fs/4 image). */
    size_t done = 0;
    while (done < num_samples) {
        size_t want = num_samples - done;
        if (want > BIRDNET_READ_SAMPLES) {
            want = BIRDNET_READ_SAMPLES;
        }
        const int raw = (int)(want * MIC_OVERSAMPLE);

        /* [ history(H) | freshly read raw(raw) ] in one contiguous buffer. */
        memcpy(s_work, s_decim_hist, (size_t)H * sizeof(int16_t));
        int rc = esp_codec_dev_read(s_mic, s_work + H, raw * (int)sizeof(int16_t));
        if (rc != ESP_CODEC_DEV_OK) {
            ESP_LOGE(TAG, "esp_codec_dev_read failed rc=%d", rc);
            return ESP_FAIL;
        }

        for (size_t j = 0; j < want; ++j) {
            const int base = (int)j * MIC_OVERSAMPLE;   /* 2:1 decimation phase */
            float acc = 0.0f;
            for (int k = 0; k < DECIM_TAPS; ++k) {
                acc += s_decim_h[k] * (float)s_work[base + k];
            }
            int v = (int)lrintf(acc);
            if (v >  32767) v =  32767;
            if (v < -32768) v = -32768;
            dst[done + j] = (int16_t)v;
        }

        /* Carry the last H raw samples into the next block's history. */
        memcpy(s_decim_hist, s_work + raw, (size_t)H * sizeof(int16_t));
        done += want;
    }
    return ESP_OK;
}
