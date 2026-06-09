/**
 * @file model_data.h
 * @brief Embedded TFLite model byte array (placeholder until trained).
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* Raw flatbuffer bytes of the quantized .tflite model. */
extern const unsigned char birdnet_model_tflite[];
extern const unsigned int  birdnet_model_tflite_len;

#ifdef __cplusplus
}
#endif
