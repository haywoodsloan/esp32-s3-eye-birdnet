# Component registration for the BirdNET application.
#
# The TFLite-Micro inference wrapper and the model data are C++ (.cc); the
# rest of the firmware is C. C++17 is required by esp-tflite-micro headers.
idf_component_register(
    SRCS
        "app_main.c"
        "audio/audio_capture.c"
        "dsp/stft_frontend.c"
        "model/birdnet_model.cc"
        "model/model_data.cc"
        "model/labels.c"
        "model/model_registry.c"
        "ui/ui.c"
    INCLUDE_DIRS
        "."
        "audio"
        "dsp"
        "model"
        "ui"
)

# esp-tflite-micro requires C++17.
target_compile_features(${COMPONENT_LIB} PRIVATE cxx_std_17)
