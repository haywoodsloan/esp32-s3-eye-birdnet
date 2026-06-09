/**
 * @file stft_frontend.h
 * @brief Block-mode linear |STFT| magnitude frontend (birdnet-stm32 "hybrid").
 *
 * Computes a librosa-compatible linear magnitude STFT over a whole audio chunk
 * and produces:
 *   1. model_input: [BIRDNET_FFT_BINS][BIRDNET_SPEC_WIDTH] normalized magnitude
 *      (row-major, bin-major) fed to the model as [1, FFT_BINS, SPEC_WIDTH, 1].
 *   2. display: [BIRDNET_DISP_H][BIRDNET_DISP_W] gamma-scaled spectrogram for UI
 *      (row 0 = highest frequency).
 */
#pragma once

#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Allocate buffers, build the Hann window and init the esp-dsp FFT.
 * @return ESP_OK on success, ESP_ERR_NO_MEM on allocation failure.
 */
esp_err_t stft_frontend_init(void);

/**
 * @brief Compute the spectrogram for one audio chunk.
 *
 * @param[in]  chunk         BIRDNET_CHUNK_SAMPLES int16 mono samples (chronological).
 * @param[out] model_input   BIRDNET_MODEL_INPUT_LEN floats, normalized [0,1].
 * @param[out] display       BIRDNET_DISP_H*BIRDNET_DISP_W floats, [0,1] (or NULL).
 */
void stft_frontend_process(const int16_t *chunk, float *model_input, float *display);

/* ----------------------------------------------------------------------------
 * Live scrolling display spectrogram (decoupled from the model block)
 *
 * Computed continuously in the audio task so the LCD scrolls smoothly, while
 * the model still runs on the 3 s block. Auto-scales to the signal level.
 * Requires stft_frontend_init() to have run first (shares the FFT tables).
 * ------------------------------------------------------------------------- */

/**
 * @brief Allocate the rolling display buffer and its window/mutex.
 * @return ESP_OK on success, ESP_ERR_NO_MEM on allocation failure.
 */
esp_err_t stft_display_init(void);

/**
 * @brief Feed mic samples; emits new columns every BIRDNET_DISP_HOP samples.
 *        Call only from a single producer (the audio task).
 *
 * @param[in] samples      int16 mono samples (chronological).
 * @param[in] num_samples  count; should be a multiple of BIRDNET_DISP_HOP.
 */
void stft_display_push(const int16_t *samples, size_t num_samples);

/**
 * @brief Copy the current scrolling spectrogram (oldest column on the left).
 *
 * @param[out] out_display  BIRDNET_DISP_H*BIRDNET_DISP_W floats, row-major,
 *                          row 0 = highest frequency, values in [0, 1].
 */
void stft_display_copy(float *out_display);

#ifdef __cplusplus
}
#endif
