/**
 * @file  tflite_inference.h
 * @brief ECGHybridV15 No-LOESS/NLM (v3) — TFLite Micro inference interface
 *        ESP32-S3 / XIAO Sense
 *
 * ── Model (v3, IEEE Access revision) ─────────────────────────────────────────
 *   Architecture : ECGHybridV15 (ResNet1D + ACF Conv1D + AttentionGate)
 *   Pipeline     : Butterworth only (no LOESS, no NLM)
 *   Training     : PTB-XL + CSN, patient-exclusive (80.7% train / 5.0% calib / 14.3% test),
 *                  deterministic training, seed 42
 *   Task         : Binary AFIB vs SR, Lead I, 10 s @ 500 Hz
 *
 * Model bytes, StandardScaler, threshold and quantisation parameters are generated
 * by export_v3_firmware.py (model.h, scaler_params.h, model_params.h) from the
 * deployed artefacts; quantisation is read from the model tensors at run time.
 *
 * ── Pre-processing expected by this module ───────────────────────────────────
 *   input_raw  : ECG signal after Butterworth ONLY (no LOESS, no NLM), in mV.
 *   input_acf  : Normalised ACF (5000 lags); call tflite_acf_normalise() first.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Constants ──────────────────────────────────────────────────────────── */

#define TFLITE_ECG_SAMPLES       5000
#define TFLITE_ACF_LAGS          5000

/* Threshold and quantisation parameters: see model_params.h (generated). */

/* ─── Result ─────────────────────────────────────────────────────────────── */

typedef struct {
    bool    is_afib;      /**< true = AFIB, false = SR                      */
    float   probability;  /**< Dequantised probability in [0.0, 1.0]        */
    int8_t  raw_output;   /**< Raw INT8 output tensor value (debug)          */
} tflite_result_t;

/* ─── Opaque context ─────────────────────────────────────────────────────── */

typedef struct tflite_inference_s tflite_inference_t;

/* ─── API ────────────────────────────────────────────────────────────────── */

/**
 * @brief  Allocate and initialise the TFLite Micro inference context.
 *         Tensor arena (512 KB reserved) allocated from PSRAM.
 * @return Pointer to context, or NULL on failure.
 */
tflite_inference_t *tflite_inference_alloc(void);

/**
 * @brief  Normalise raw ACF using the embedded StandardScaler.
 *         Call BEFORE tflite_inference_run().
 * @param[in]  acf_raw   Raw ACF from autocorr_compute() [TFLITE_ACF_LAGS].
 * @param[out] acf_norm  Normalised ACF, ready for tflite_inference_run().
 */
void tflite_acf_normalise(const float *acf_raw, float *acf_norm);

/**
 * @brief  Run a single ECG inference.
 * @param[in]  ctx      Context from tflite_inference_alloc().
 * @param[in]  ecg_raw  ECG after Butterworth only — NOT normalised.
 *                      Must be TFLITE_ECG_SAMPLES floats.
 * @param[in]  acf_raw  ACF already normalised by tflite_acf_normalise().
 *                      Must be TFLITE_ACF_LAGS floats.
 * @param[out] result   Classification result.
 * @return true on success, false on error.
 */
bool tflite_inference_run(tflite_inference_t *ctx,
                          const float        *ecg_raw,
                          const float        *acf_raw,
                          tflite_result_t    *result);

/** @brief Bytes of the tensor arena actually used by the interpreter. */
size_t tflite_inference_arena_used(const tflite_inference_t *ctx);

/** @brief Bytes reserved for the tensor arena. */
size_t tflite_inference_arena_size(void);

/**
 * @brief  Free all resources. Safe to call with NULL.
 */
void tflite_inference_free(tflite_inference_t *ctx);

#ifdef __cplusplus
}
#endif
