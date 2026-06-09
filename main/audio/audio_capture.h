/**
 * @file audio_capture.h
 * @brief I2S MEMS microphone capture for the ESP32-S3-EYE.
 */
#pragma once

#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the board audio interface and open the microphone.
 *
 * Configures the I2S bus through the BSP and opens the digital MEMS mic. The
 * mic is opened at twice BIRDNET_SAMPLE_RATE_HZ (16-bit mono); audio_capture_read
 * decimates by 2 to undo the mic's duplicated-sample output (see the .c file).
 *
 * @return ESP_OK on success, or an error from the BSP / codec layer.
 */
esp_err_t audio_capture_init(void);

/**
 * @brief Read exactly @p num_samples mono int16 samples at BIRDNET_SAMPLE_RATE_HZ.
 *
 * Reads 2x @p num_samples raw samples from the mic and decimates 2:1 through a
 * linear-phase anti-aliasing FIR (see the .c file), yielding a clean,
 * non-aliased stream. Blocks until the requested number of samples is available.
 *
 * @param[out] dst          Destination buffer (at least @p num_samples int16).
 * @param[in]  num_samples  Number of (decimated) samples to read.
 * @return ESP_OK on success, or an error from the codec read.
 */
esp_err_t audio_capture_read(int16_t *dst, size_t num_samples);

#ifdef __cplusplus
}
#endif
