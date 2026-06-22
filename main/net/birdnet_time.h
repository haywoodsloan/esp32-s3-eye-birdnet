/**
 * @file birdnet_time.h
 * @brief WiFi + SNTP wall-clock time with NVS persistence, for the seasonal
 *        occurrence prior.
 *
 * The seasonal prior needs to know roughly what week of the year it is. This
 * module joins the configured WiFi network (set via menuconfig ->
 * "BirdNET detector"), syncs the clock over SNTP, persists the epoch to NVS so
 * an approximate date survives reboots without WiFi, and re-syncs periodically.
 *
 * Everything here is gated on BIRDNET_USE_PRIOR; with the prior compiled out the
 * functions are not defined and nothing starts WiFi.
 */
#pragma once

#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Spawn the background task that connects WiFi, syncs SNTP, persists to NVS and
 * re-syncs. Non-blocking; returns once the task is created. Safe to call with no
 * SSID configured (it logs and leaves the time "unknown"). */
esp_err_t birdnet_time_start(void);

/* Current week as a 0-based row index into BIRDNET_OCCURRENCE_PRIOR [0,47]
 * (BirdNET's 4-weeks-per-month convention), or -1 if the time is not yet known.
 * A -1 tells the detector to skip the prior (no behavior change). */
int birdnet_time_prior_week(void);

/* True once a usable wall-clock exists (synced this boot or restored from NVS). */
bool birdnet_time_valid(void);

#ifdef __cplusplus
}
#endif
