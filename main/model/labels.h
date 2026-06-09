/**
 * @file labels.h
 * @brief Active species label table (dynamic; loaded per model from SD).
 */
#pragma once

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Number of labels in the active table (changes when a model is loaded). */
extern int birdnet_num_labels;

/** @brief Common name for a class index, or "?" if out of range. */
const char *birdnet_label_name(int index);

/** @brief Scientific name for a class index, or "?" if out of range. */
const char *birdnet_label_sci(int index);

/** @brief Reset the table to the built-in default species list. */
void birdnet_labels_reset_default(void);

/**
 * @brief Load labels from a text file (one entry per line).
 *
 * Each line is "Common|Scientific" or "Common\tScientific"; a line with no
 * delimiter is used as both common and scientific name. Lines beyond
 * BIRDNET_NUM_CLASSES_MAX are ignored.
 *
 * @param path  Absolute VFS path (e.g. "/sdcard/models/foo.txt").
 * @return ESP_OK on success (table replaced), error otherwise (table kept).
 */
esp_err_t birdnet_labels_load_file(const char *path);

#ifdef __cplusplus
}
#endif
