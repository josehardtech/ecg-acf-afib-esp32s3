/**
 * @file  tflite_inference.cc
 * @brief Generic INT8 inference (deployed v3 model or MobileNet-1D baseline) — TFLite Micro
 *        ESP32-S3 / XIAO Sense
 *
 *   Butterworth → ACF (mean-centered) → StandardScaler → INT8 → TFLite Micro
 *
 * v3 (revisão IEEE Access, rodada 2):
 *   - modelo, escalonador e τ gerados por export_firmware.py (model.h,
 *     model_params.h, model_ops.inc, scaler_params.h); SHA-256 do modelo registrado no log;
 *   - parâmetros de quantização lidos dos tensores do modelo em tempo de execução
 *     e conferidos contra model_params.h (não há mais constantes digitadas à mão);
 *   - decisão pelo limiar inteiro MODEL_THRESHOLD_Q na saída INT8 (idêntica ao Python);
 *   - API para o tamanho usado e reservado da tensor arena.
 */

#include "tflite_inference.h"
#include "model.h"          /* ecg_hybrid_v15_sem_loess_nlm_int8_tflite[] */
#include "model_params.h"   /* τ, limiar inteiro, quantização esperada, SHA-256 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "esp_log.h"
#include "esp_heap_caps.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

/* ─── Configuration ──────────────────────────────────────────────────────── */

static const char *TAG = "TFLITE";

static constexpr size_t TFLITE_ARENA_SIZE = 512 * 1024;

/* Número de operações: gerado (model_params.h) */
static constexpr int TFLITE_NUM_OPS = MODEL_NUM_OPS;

/* ─── Scaler arrays ─────────────────────────────────────────────────────────
 * StandardScaler da ACF, gerado por export_firmware.py.                       */
#include "scaler_params.h"   /* vazio (1 elemento) se o modelo não usa a ACF */

/* ─── Op resolver ────────────────────────────────────────────────────────── */

/* Operações do modelo, geradas por export_firmware.py a partir do .tflite (model_ops.inc) */
static tflite::MicroMutableOpResolver<TFLITE_NUM_OPS> *build_op_resolver(void)
{
    auto *resolver = new tflite::MicroMutableOpResolver<TFLITE_NUM_OPS>();
#include "model_ops.inc"
    return resolver;
}

/* ─── Context struct ─────────────────────────────────────────────────────── */

struct tflite_inference_s {
    uint8_t                                        *arena;
    tflite::MicroMutableOpResolver<TFLITE_NUM_OPS> *resolver;
    tflite::MicroInterpreter                       *interpreter;
    const tflite::Model                            *model;
    int8_t *q_raw;
    int8_t *q_acf;
    /* Quantização lida do modelo (v3) */
    float  raw_scale, acf_scale, out_scale;
    int    raw_zp, acf_zp, out_zp;
};

/* ─── Helpers ────────────────────────────────────────────────────────────── */

static inline int8_t clamp_i8(float x)
{
    if (x >  127.0f) return  127;
    if (x < -128.0f) return (int8_t)(-128);
    return (int8_t)(int)x;
}

static void quantise(const float *x, int8_t *q, size_t n,
                     float scale, int zero_point)
{
    const float inv_scale = 1.0f / scale;
    for (size_t i = 0; i < n; i++)
        q[i] = clamp_i8(roundf(x[i] * inv_scale + (float)zero_point));
}

/* ─── Public API ─────────────────────────────────────────────────────────── */

void tflite_acf_normalise(const float *acf_raw, float *acf_norm)
{
#if MODEL_USES_ACF
    for (size_t i = 0; i < TFLITE_ACF_LAGS; i++)
        acf_norm[i] = (acf_raw[i] - kScalerMean[i]) / kScalerScale[i];
#else
    (void)acf_raw; (void)acf_norm;   /* modelo sem ramo ACF */
#endif
}

/* ─────────────────────────────────────────────────────────────────────────── */

tflite_inference_t *tflite_inference_alloc(void)
{
    tflite_inference_t *ctx =
        (tflite_inference_t *)heap_caps_calloc(1, sizeof(tflite_inference_t),
                                               MALLOC_CAP_SPIRAM);
    if (!ctx) { ESP_LOGE(TAG, "alloc: context malloc failed"); return nullptr; }

    ctx->arena = (uint8_t *)heap_caps_malloc(TFLITE_ARENA_SIZE, MALLOC_CAP_SPIRAM);
    if (!ctx->arena) { ESP_LOGE(TAG, "alloc: arena malloc failed"); goto fail; }

    ctx->q_raw = (int8_t *)heap_caps_malloc(TFLITE_ECG_SAMPLES * sizeof(int8_t),
                                             MALLOC_CAP_SPIRAM);
    ctx->q_acf = (int8_t *)heap_caps_malloc(TFLITE_ACF_LAGS    * sizeof(int8_t),
                                             MALLOC_CAP_SPIRAM);
    if (!ctx->q_raw || !ctx->q_acf) {
        ESP_LOGE(TAG, "alloc: scratch buffer malloc failed");
        goto fail;
    }

    ctx->model = tflite::GetModel(ecg_hybrid_v15_sem_loess_nlm_int8_tflite);
    if (ctx->model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "alloc: model schema mismatch");
        goto fail;
    }

    ctx->resolver = build_op_resolver();
    if (!ctx->resolver) { ESP_LOGE(TAG, "alloc: resolver failed"); goto fail; }

    ctx->interpreter = new tflite::MicroInterpreter(
        ctx->model, *ctx->resolver, ctx->arena, TFLITE_ARENA_SIZE);

    if (ctx->interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "alloc: AllocateTensors failed");
        /* NÃO chamar delete ctx->interpreter — objeto parcialmente inicializado
         * causa LoadProhibited panic no destrutor (aprendido no modelo anterior) */
        goto fail;
    }

    {
        if (ctx->interpreter->inputs_size() != MODEL_NUM_INPUTS) {
            ESP_LOGE(TAG, "alloc: modelo tem %u entradas, model_params.h diz %d",
                     (unsigned)ctx->interpreter->inputs_size(), MODEL_NUM_INPUTS);
            goto fail;
        }
        TfLiteTensor *t_raw = ctx->interpreter->input(MODEL_RAW_INPUT_INDEX);
        TfLiteTensor *t_acf = (MODEL_ACF_INPUT_INDEX >= 0) ? ctx->interpreter->input(MODEL_ACF_INPUT_INDEX) : nullptr;
        TfLiteTensor *t_out = ctx->interpreter->output(0);
        if (!t_raw || !t_out || (MODEL_ACF_INPUT_INDEX >= 0 && !t_acf)) { ESP_LOGE(TAG, "alloc: tensor NULL"); goto fail; }
        ctx->raw_scale = t_raw->params.scale; ctx->raw_zp = t_raw->params.zero_point;
        ctx->acf_scale = t_acf ? t_acf->params.scale : MODEL_ACF_SCALE;
        ctx->acf_zp    = t_acf ? t_acf->params.zero_point : MODEL_ACF_ZP;
        ctx->out_scale = t_out->params.scale; ctx->out_zp = t_out->params.zero_point;
        bool ok = fabsf(ctx->raw_scale - MODEL_RAW_SCALE) < 1e-6f && ctx->raw_zp == MODEL_RAW_ZP &&
                  fabsf(ctx->acf_scale - MODEL_ACF_SCALE) < 1e-6f && ctx->acf_zp == MODEL_ACF_ZP &&
                  fabsf(ctx->out_scale - MODEL_OUT_SCALE) < 1e-6f && ctx->out_zp == MODEL_OUT_ZP;
        if (!ok) {
            ESP_LOGE(TAG, "alloc: quantização do modelo difere de model_params.h — regenerar os .h");
            goto fail;
        }
    }

    ESP_LOGI(TAG, "Model %s | SHA-256 %s | %d bytes", MODEL_VERSION_TAG, MODEL_SHA256,
             (int)MODEL_TFLITE_BYTES);
    ESP_LOGI(TAG, "raw: scale=%.8f zp=%d | acf: scale=%.8f zp=%d | out: scale=%.8f zp=%d",
             ctx->raw_scale, ctx->raw_zp, ctx->acf_scale, ctx->acf_zp, ctx->out_scale, ctx->out_zp);
    ESP_LOGI(TAG, "Decision: AFIB if int8_out >= %d (tau = %.6f) | inputs=%d | uses ACF=%d | ops=%d",
             MODEL_THRESHOLD_Q, MODEL_TAU, MODEL_NUM_INPUTS, MODEL_USES_ACF, MODEL_NUM_OPS);
    ESP_LOGI(TAG, "[MEM] tensor arena: used %u B of %u B reserved (PSRAM)",
             (unsigned)ctx->interpreter->arena_used_bytes(), (unsigned)TFLITE_ARENA_SIZE);
    return ctx;

fail:
    heap_caps_free(ctx->q_raw);
    heap_caps_free(ctx->q_acf);
    heap_caps_free(ctx->arena);
    delete ctx->resolver;
    /* delete ctx->interpreter — OMITIDO intencionalmente (ver comentário acima) */
    heap_caps_free(ctx);
    return nullptr;
}

/* ─────────────────────────────────────────────────────────────────────────── */

bool tflite_inference_run(tflite_inference_t *ctx,
                          const float        *ecg_raw,
                          const float        *acf_raw,
                          tflite_result_t    *result)
{
    if (!ctx || !ecg_raw || !result || (MODEL_USES_ACF && !acf_raw)) {
        ESP_LOGE(TAG, "run: NULL argument");
        return false;
    }

    /* ── 1-2. Quantise inputs and set tensors ──────────────────────────
     * acf_raw deve chegar normalizado (tflite_acf_normalise); é ignorado
     * (pode ser NULL) quando o modelo não usa a ACF.                      */
    quantise(ecg_raw, ctx->q_raw, TFLITE_ECG_SAMPLES, ctx->raw_scale, ctx->raw_zp);
    TfLiteTensor *t_raw = ctx->interpreter->input(MODEL_RAW_INPUT_INDEX);
    if (!t_raw) { ESP_LOGE(TAG, "run: input_raw NULL"); return false; }
    memcpy(t_raw->data.int8, ctx->q_raw, TFLITE_ECG_SAMPLES * sizeof(int8_t));
#if MODEL_ACF_INPUT_INDEX >= 0
    {
        TfLiteTensor *t_acf = ctx->interpreter->input(MODEL_ACF_INPUT_INDEX);
        if (!t_acf) { ESP_LOGE(TAG, "run: input_acf NULL"); return false; }
#if MODEL_USES_ACF
        quantise(acf_raw, ctx->q_acf, TFLITE_ACF_LAGS, ctx->acf_scale, ctx->acf_zp);
        memcpy(t_acf->data.int8, ctx->q_acf, TFLITE_ACF_LAGS * sizeof(int8_t));
#else
        memset(t_acf->data.int8, ctx->acf_zp, t_acf->bytes);   /* entrada não usada pelo grafo */
#endif
    }
#endif

    /* ── 3. Invoke ───────────────────────────────────────────────────── */
    if (ctx->interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "run: Invoke() failed");
        return false;
    }

    /* ── 4. Dequantise output ─────────────────────────────────────────── */
    TfLiteTensor *t_out = ctx->interpreter->output(0);
    if (!t_out) { ESP_LOGE(TAG, "run: output tensor NULL"); return false; }

    int8_t out_i8 = t_out->data.int8[0];
    float  prob   = ((float)out_i8 - (float)ctx->out_zp) * ctx->out_scale;
    if (prob < 0.0f) prob = 0.0f;
    if (prob > 1.0f) prob = 1.0f;

    result->raw_output  = out_i8;
    result->probability = prob;
    result->is_afib     = ((int)out_i8 >= MODEL_THRESHOLD_Q);   /* idêntico ao Python */

    return true;
}

/* ─────────────────────────────────────────────────────────────────────────── */

size_t tflite_inference_arena_used(const tflite_inference_t *ctx)
{
    return (ctx && ctx->interpreter) ? ctx->interpreter->arena_used_bytes() : 0;
}

size_t tflite_inference_arena_size(void) { return TFLITE_ARENA_SIZE; }

/* ─────────────────────────────────────────────────────────────────────────── */

void tflite_inference_free(tflite_inference_t *ctx)
{
    if (!ctx) return;
    delete ctx->interpreter;
    delete ctx->resolver;
    heap_caps_free(ctx->q_raw);
    heap_caps_free(ctx->q_acf);
    heap_caps_free(ctx->arena);
    heap_caps_free(ctx);
}
