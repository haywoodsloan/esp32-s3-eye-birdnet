/**
 * @file model_data.cc
 * @brief Embedded TFLite model byte array.
 *
 * PLACEHOLDER: this is NOT a real model. While BIRDNET_HAS_REAL_MODEL == 0
 * (see birdnet_config.h) these bytes are never handed to TFLite-Micro and the
 * firmware uses the mock classifier instead.
 *
 * To embed a real model:
 *   1. Train + quantize a model with BirdNET-Tiny-Forge (see README).
 *   2. Run:  python tools/convert_model.py path/to/model.tflite \
 *                   main/model/model_data.cc
 *      which regenerates this file with the model bytes.
 *   3. Set BIRDNET_HAS_REAL_MODEL to 1 (or -DBIRDNET_HAS_REAL_MODEL=1).
 *
 * The 16-byte alignment matches TFLite-Micro's expectation for the model
 * flatbuffer.
 */
#include "model_data.h"

alignas(16) const unsigned char birdnet_model_tflite[] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};

const unsigned int birdnet_model_tflite_len =
    (unsigned int)sizeof(birdnet_model_tflite);
