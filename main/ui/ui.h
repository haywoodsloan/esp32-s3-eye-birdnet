/**
 * @file ui.h
 * @brief LVGL UI: scrolling spectrogram + color-coded mic level meter.
 *
 * Call with the BSP display (LVGL) lock held: bsp_display_lock() / unlock().
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/** Immutable snapshot of state to render for one UI frame. */
typedef struct {
    const float *display;   /**< BIRDNET_DISP_H*BIRDNET_DISP_W in [0,1], row0=top. */
    float        level;     /**< Microphone RMS level in [0, 1].                   */
} ui_snapshot_t;

/** @brief Build the widget tree and the spectrogram canvas. @return 0 on success. */
int ui_init(void);

/** @brief Render one frame from @p snap. */
void ui_update(const ui_snapshot_t *snap);

#ifdef __cplusplus
}
#endif
