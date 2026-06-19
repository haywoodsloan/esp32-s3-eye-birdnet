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
        "ui/assets/roboto_28.c"
        "ui/assets/roboto_18.c"
        "ui/assets/roboto_30.c"
        "ui/assets/roboto_pct_32.c"
    INCLUDE_DIRS
        "."
        "audio"
        "dsp"
        "model"
        "ui"
)

# esp-tflite-micro requires C++17.
target_compile_features(${COMPONENT_LIB} PRIVATE cxx_std_17)

# The generated LVGL font files include <lvgl.h> via the "simple" path.
target_compile_definitions(${COMPONENT_LIB} PRIVATE LV_LVGL_H_INCLUDE_SIMPLE)
