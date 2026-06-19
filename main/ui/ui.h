/**
 * @file ui.h
 * @brief LVGL UI: scrolling spectrogram + color-coded mic level meter.
 *
 * Call with the BSP display (LVGL) lock held: bsp_display_lock() / unlock().
 */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Immutable snapshot of state to render for one UI frame. */
typedef struct {
    const float *display;   /**< BIRDNET_DISP_H*BIRDNET_DISP_W in [0,1], row0=top. */
    float        level;     /**< Microphone RMS level in [0, 1].                   */

    /* Detection overlay event. The card appears whenever det_seq changes to a
     * new non-zero value and stays up for BIRDNET_DETECT_OVERLAY_MS. The main
     * render loop owns these fields (populated from cross-core detection state
     * published by the inference task). */
    uint32_t     det_seq;   /**< Monotonic detection counter; 0 = none yet.        */
    const char  *det_name;  /**< Common name for the card title (may be NULL).     */
    const char  *det_sci;   /**< Scientific name, for the SD photo lookup.         */
    float        det_score; /**< Detection confidence in [0, 1].                   */
} ui_snapshot_t;

/** @brief Build the widget tree and the spectrogram canvas. @return 0 on success. */
int ui_init(void);

/** @brief Render one frame from @p snap. */
void ui_update(const ui_snapshot_t *snap);

#ifdef __cplusplus
}
#endif
