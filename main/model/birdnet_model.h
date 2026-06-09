/**
 * @file birdnet_model.h
 * @brief Classifier wrapper (TFLite-Micro) with hot model swapping + mock.
 *
 * Targets the birdnet-stm32 DS-CNN ("hybrid" frontend):
 *   input  : [1, BIRDNET_FFT_BINS, BIRDNET_SPEC_WIDTH, 1] float32 (or int8),
 *            a normalized linear |STFT| magnitude spectrogram.
 *   output : [1, num_classes] float32 sigmoid scores (multi-label).
 *
 * One model is resident at a time. Call birdnet_model_set() to (re)build the
 * interpreter over the shared arena when swapping models from SD. Until a model
 * is set (or if a load fails), birdnet_model_run() returns mock results so the
 * pipeline and UI keep working.
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Result of a single classification. */
typedef struct {
    int   top_index;   /**< Argmax class index, or -1 if none.     */
    float top_score;   /**< Top class score in [0, 1].             */
    bool  is_mock;     /**< True if produced by the mock model.    */
} birdnet_result_t;

/**
 * @brief Allocate the tensor arena and build the op resolver (no model yet).
 * @return ESP_OK on success.
 */
esp_err_t birdnet_model_init(void);

/**
 * @brief (Re)build the interpreter for a model flatbuffer.
 *
 * Tears down any previous interpreter and rebuilds over the shared arena. The
 * @p model_data buffer must remain valid/resident for as long as the model is
 * in use (the registry keeps it in a resident PSRAM buffer).
 *
 * @param model_data  Pointer to a 16-byte-aligned .tflite flatbuffer.
 * @param len         Length in bytes (informational).
 * @return ESP_OK on success; on failure the model is marked unloaded (mock).
 */
esp_err_t birdnet_model_set(const uint8_t *model_data, size_t len);

/** @brief True if a real model is currently loaded. */
bool birdnet_model_is_loaded(void);

/**
 * @brief Run the classifier on a normalized linear |STFT| spectrogram.
 *
 * @param[in]  spectro  BIRDNET_MODEL_INPUT_LEN floats, layout bin*SPEC_WIDTH+frame.
 * @param[out] scores   Optional per-class scores (length birdnet_num_labels), or NULL.
 * @param[out] result   Top-1 result summary.
 * @return ESP_OK on success.
 */
esp_err_t birdnet_model_run(const float *spectro, float *scores,
                            birdnet_result_t *result);

#ifdef __cplusplus
}
#endif
