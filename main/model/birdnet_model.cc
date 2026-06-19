/**
 * @file birdnet_model.cc
 * @brief TFLite-Micro classifier wrapper with hot model swapping + mock.
 *
 * TFLM is always compiled now (SD models are loaded at runtime regardless of
 * the embedded-model flag). The interpreter is rebuilt on each swap via
 * placement-new over a fixed storage block, reusing one PSRAM tensor arena and
 * one shared op resolver.
 *
 * Input contract (see birdnet_config.h): a normalized linear |STFT| magnitude
 * spectrogram, [1, BIRDNET_FFT_BINS, BIRDNET_SPEC_WIDTH, 1], float32 or int8.
 * The model embeds its own mel mixer, max-norm, PWL scaling and DS-CNN, which
 * is why the op resolver is broad. If AllocateTensors() reports a missing op,
 * add the matching Add*() below.
 */
#include "birdnet_model.h"

#include <math.h>
#include <new>
#include <string.h>

#include "birdnet_config.h"
#include "labels.h"
#include "esp_heap_caps.h"
#include "esp_log.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char *TAG = "birdnet_model";

/* ------------------------------------------------------------------------- */
/* Mock classifier                                                            */
/* ------------------------------------------------------------------------- */
static void mock_classify(const float *spectro, float *scores, birdnet_result_t *result)
{
    const int n = birdnet_num_labels;
    if (n <= 0) {
        result->top_index = -1;
        result->top_score = 0.0f;
        result->is_mock = true;
        return;
    }

    float total = 0.0f, centroid_num = 0.0f;
    for (int b = 0; b < BIRDNET_FFT_BINS; ++b) {
        for (int t = 0; t < BIRDNET_SPEC_WIDTH; ++t) {
            float v = spectro[b * BIRDNET_SPEC_WIDTH + t];
            total += v;
            centroid_num += v * (float)b;
        }
    }
    float mean = total / (float)BIRDNET_MODEL_INPUT_LEN;
    float centroid = (total > 1e-6f)
                         ? (centroid_num / total) / (float)BIRDNET_FFT_BINS
                         : 0.0f;

    float sum = 0.0f;
    int top = 0;
    float top_raw = -1.0f;
    for (int i = 0; i < n; ++i) {
        float phase = centroid * (float)n + mean * 3.0f + (float)i * 0.7f;
        float s = 0.5f + 0.5f * sinf(phase);
        if (scores) scores[i] = s;
        sum += s;
        if (s > top_raw) { top_raw = s; top = i; }
    }
    if (scores && sum > 1e-6f) {
        for (int i = 0; i < n; ++i) scores[i] /= sum;
    }

    float conf = top_raw * (0.30f + 6.0f * mean);
    if (conf < 0.0f) conf = 0.0f;
    if (conf > 0.99f) conf = 0.99f;

    result->top_index = top;
    result->top_score = (sum > 1e-6f) ? conf : 0.0f;
    result->is_mock   = true;
}

/* ------------------------------------------------------------------------- */
/* TFLite-Micro state                                                         */
/* ------------------------------------------------------------------------- */
namespace {
constexpr int kArenaSize = BIRDNET_TFLITE_ARENA_BYTES;
uint8_t *g_arena = nullptr;

/* Superset covering the in-graph hybrid frontend (Transpose, mel-mixer Conv2D,
 * ReLU, ReduceMax/Div max-norm, Concatenation channel-pad, StridedSlice, PWL
 * Add/Mul) plus the DS-CNN body (DepthwiseConv2D, Add residuals, Mul SE,
 * Relu6, Mean GAP, FullyConnected, Logistic) and float<->int8 boundary.
 *
 * NOTE: the model must be exported with a STATIC batch of 1 (see the training
 * repo's conversion step). A dynamic batch makes the frontend's channel-pad
 * emit tf.zeros(tf.shape(x)) -> Shape/Pack/Fill ops; TFLite-Micro's Fill kernel
 * rejects a non-constant dims tensor, so AllocateTensors() fails on-device. A
 * static batch constant-folds those away, so they are intentionally NOT here. */
using BirdNetOpResolver = tflite::MicroMutableOpResolver<28>;
BirdNetOpResolver g_resolver;
bool g_resolver_ready = false;

/* Storage for the interpreter, reconstructed in place on each swap. */
alignas(8) uint8_t g_interp_storage[sizeof(tflite::MicroInterpreter)];
tflite::MicroInterpreter *g_interp = nullptr;
TfLiteTensor *g_input = nullptr;
TfLiteTensor *g_output = nullptr;
bool g_loaded = false;

TfLiteStatus register_ops(BirdNetOpResolver &r)
{
    if (r.AddConv2D() != kTfLiteOk) return kTfLiteError;
    if (r.AddDepthwiseConv2D() != kTfLiteOk) return kTfLiteError;
    if (r.AddFullyConnected() != kTfLiteOk) return kTfLiteError;
    if (r.AddAdd() != kTfLiteOk) return kTfLiteError;
    if (r.AddMul() != kTfLiteOk) return kTfLiteError;
    if (r.AddSub() != kTfLiteOk) return kTfLiteError;
    if (r.AddMean() != kTfLiteOk) return kTfLiteError;
    if (r.AddLogistic() != kTfLiteOk) return kTfLiteError;
    if (r.AddSoftmax() != kTfLiteOk) return kTfLiteError;
    if (r.AddRelu() != kTfLiteOk) return kTfLiteError;
    if (r.AddRelu6() != kTfLiteOk) return kTfLiteError;
    if (r.AddReshape() != kTfLiteOk) return kTfLiteError;
    if (r.AddTranspose() != kTfLiteOk) return kTfLiteError;
    if (r.AddReduceMax() != kTfLiteOk) return kTfLiteError;
    if (r.AddDiv() != kTfLiteOk) return kTfLiteError;
    if (r.AddPad() != kTfLiteOk) return kTfLiteError;
    if (r.AddStridedSlice() != kTfLiteOk) return kTfLiteError;
    if (r.AddConcatenation() != kTfLiteOk) return kTfLiteError;
    if (r.AddMaximum() != kTfLiteOk) return kTfLiteError;
    if (r.AddMinimum() != kTfLiteOk) return kTfLiteError;
    if (r.AddMaxPool2D() != kTfLiteOk) return kTfLiteError;
    if (r.AddAveragePool2D() != kTfLiteOk) return kTfLiteError;
    if (r.AddQuantize() != kTfLiteOk) return kTfLiteError;
    if (r.AddDequantize() != kTfLiteOk) return kTfLiteError;
    return kTfLiteOk;
}

void teardown_interpreter()
{
    if (g_interp != nullptr) {
        g_interp->~MicroInterpreter();
        g_interp = nullptr;
    }
    g_input = nullptr;
    g_output = nullptr;
    g_loaded = false;
}

/* Abandon a half-built interpreter WITHOUT running its destructor. After a
 * failed AllocateTensors() the subgraph table is left partially initialized, so
 * ~MicroInterpreter()->FreeSubgraphs() dereferences an uninitialized pointer and
 * crashes (LoadProhibited) -- which turns a recoverable load failure into a boot
 * loop. The interpreter lives in a fixed storage block and owns nothing outside
 * the reused PSRAM arena, so simply forgetting it leaks nothing; the next load
 * placement-news over the same storage. */
void forget_interpreter()
{
    g_interp = nullptr;
    g_input = nullptr;
    g_output = nullptr;
    g_loaded = false;
}
}  // namespace

/* ------------------------------------------------------------------------- */
/* Public API                                                                 */
/* ------------------------------------------------------------------------- */
esp_err_t birdnet_model_init(void)
{
    if (g_arena == nullptr) {
        g_arena = (uint8_t *)heap_caps_aligned_alloc(
            16, kArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (g_arena == nullptr) {
            g_arena = (uint8_t *)heap_caps_aligned_alloc(16, kArenaSize, MALLOC_CAP_8BIT);
        }
        if (g_arena == nullptr) {
            ESP_LOGE(TAG, "failed to allocate %d byte tensor arena", kArenaSize);
            return ESP_ERR_NO_MEM;
        }
    }

    if (!g_resolver_ready) {
        if (register_ops(g_resolver) != kTfLiteOk) {
            ESP_LOGE(TAG, "failed to register ops");
            return ESP_FAIL;
        }
        g_resolver_ready = true;
    }

    ESP_LOGI(TAG, "ready: arena=%d KB (no model loaded; mock until set)",
             kArenaSize / 1024);
    return ESP_OK;
}

esp_err_t birdnet_model_set(const uint8_t *model_data, size_t len)
{
    if (model_data == nullptr || g_arena == nullptr) {
        return ESP_ERR_INVALID_STATE;
    }

    const tflite::Model *model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "model schema %lu != supported %d",
                 (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
        return ESP_FAIL;
    }

    /* Log the size up front: AllocateTensors() can fail before any other line
     * prints, so this is the only on-device confirmation of WHICH .tflite was
     * read from SD (size is the quickest way to tell stale vs current). */
    ESP_LOGI(TAG, "loading model: %u bytes", (unsigned)len);

    /* Drop the old interpreter before building the new one (destruction does
     * not read the model flatbuffer, so buffer reuse upstream is safe). */
    teardown_interpreter();

    g_interp = new (g_interp_storage)
        tflite::MicroInterpreter(model, g_resolver, g_arena, kArenaSize);

    if (g_interp->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed (arena too small or missing op?)");
        /* Forget (do NOT destruct) the half-built interpreter: its destructor
         * crashes after a failed allocation. Returning cleanly lets the caller
         * fall back to the embedded/mock model instead of panicking. */
        forget_interpreter();
        return ESP_FAIL;
    }

    g_input  = g_interp->input(0);
    g_output = g_interp->output(0);
    g_loaded = true;

    ESP_LOGI(TAG, "model set: %u bytes, arena used=%u KB, in=[%d,%d,%d,%d] type=%d out=%d",
             (unsigned)len,
             (unsigned)(g_interp->arena_used_bytes() / 1024),
             g_input->dims->size > 0 ? g_input->dims->data[0] : 0,
             g_input->dims->size > 1 ? g_input->dims->data[1] : 0,
             g_input->dims->size > 2 ? g_input->dims->data[2] : 0,
             g_input->dims->size > 3 ? g_input->dims->data[3] : 0,
             (int)g_input->type,
             g_output->dims->data[g_output->dims->size - 1]);
    return ESP_OK;
}

bool birdnet_model_is_loaded(void)
{
    return g_loaded;
}

static esp_err_t real_run(const float *spectro, float *scores, birdnet_result_t *result)
{
    const int n_in = BIRDNET_MODEL_INPUT_LEN;

    if (g_input->type == kTfLiteInt8) {
        const float scale = g_input->params.scale;
        const int   zp    = g_input->params.zero_point;
        int8_t *in = g_input->data.int8;
        for (int i = 0; i < n_in; ++i) {
            int q = (int)lroundf(spectro[i] / scale) + zp;
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            in[i] = (int8_t)q;
        }
    } else {
        memcpy(g_input->data.f, spectro, sizeof(float) * (size_t)n_in);
    }

    if (g_interp->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed");
        return ESP_FAIL;
    }

    const int n = g_output->dims->data[g_output->dims->size - 1];
    int top = -1;
    float top_v = -1.0f;
    for (int i = 0; i < n; ++i) {
        float v;
        if (g_output->type == kTfLiteInt8) {
            v = (g_output->data.int8[i] - g_output->params.zero_point) *
                g_output->params.scale;
        } else {
            v = g_output->data.f[i];
        }
        if (scores && i < birdnet_num_labels) scores[i] = v;
        if (v > top_v) { top_v = v; top = i; }
    }

    result->top_index = top;
    result->top_score = top_v;
    result->is_mock   = false;
    return ESP_OK;
}

esp_err_t birdnet_model_run(const float *spectro, float *scores, birdnet_result_t *result)
{
    if (spectro == nullptr || result == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }
    if (g_loaded) {
        return real_run(spectro, scores, result);
    }
    mock_classify(spectro, scores, result);
    return ESP_OK;
}
