/**
 * @file model_registry.h
 * @brief SD-backed registry of swappable BirdNET expert models (Option B).
 *
 * Mounts the microSD card, parses /models/manifest.txt, and loads a selected
 * .tflite (+ its labels) from SD into a resident PSRAM buffer, handing it to
 * the model wrapper. One model is active at a time; selection can be changed
 * (e.g. by a button) and is applied between inferences by the inference task.
 *
 * Threading: model_registry_request() is callback/ISR-light (just stores an
 * index). model_registry_apply() performs SD I/O + interpreter rebuild and MUST
 * be called from the inference task only.
 */
#pragma once

#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Mount SD, parse the manifest, allocate the resident model buffer.
 *
 * Does NOT load a model yet (call model_registry_apply()).
 *
 * @return ESP_OK if SD mounted and >=1 model was found in the manifest;
 *         an error otherwise (caller should fall back to embedded/mock).
 */
esp_err_t model_registry_init(void);

/** @brief Number of models listed in the manifest (0 if unavailable). */
int model_registry_count(void);

/** @brief Index chosen as default at boot (manifest "default", else 0). */
int model_registry_default_index(void);

/** @brief Name of the model at @p index, or "" if out of range. */
const char *model_registry_name(int index);

/** @brief Name of the currently applied model, or "" if none. */
const char *model_registry_active_name(void);

/**
 * @brief Load model @p index (+ its labels) from SD and make it active.
 *
 * Inference task only. On success the wrapper holds the new model and the label
 * table is replaced. On failure the previous state is left as-is.
 *
 * @return ESP_OK on success.
 */
esp_err_t model_registry_apply(int index);

/**
 * @brief Request a swap to @p index (safe to call from a button callback).
 * The inference task applies it before the next inference.
 */
void model_registry_request(int index);

/**
 * @brief Request a swap to the next model (wraps around).
 */
void model_registry_request_next(void);

/**
 * @brief If a swap was requested, return its index and clear it; else -1.
 */
int model_registry_take_pending(void);

#ifdef __cplusplus
}
#endif
