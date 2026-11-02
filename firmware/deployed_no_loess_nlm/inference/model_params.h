/* model_params.h — gerado por export_v3_firmware.py em 2026-11-02T08:56:21 */
#pragma once
#define MODEL_SHA256          "a7582a6d1dd772754886b242c2c23a9c1fd36a7dbf7d261e4761e14b5e086b2e"
#define MODEL_VERSION_TAG     "v3 No-LOESS/NLM patient-exclusive"
#define MODEL_TFLITE_BYTES    193504
/* Decisão: AFIB se saída INT8 >= MODEL_THRESHOLD_Q  (equivale a p >= MODEL_TAU) */
#define MODEL_TAU             0.81250000f
#define MODEL_THRESHOLD_Q     (80)
/* Parâmetros de quantização esperados (conferidos em tempo de execução contra o modelo) */
#define MODEL_RAW_INPUT_INDEX 0
#define MODEL_ACF_INPUT_INDEX 1
#define MODEL_RAW_SCALE       9.0415552258e-03f
#define MODEL_RAW_ZP          (0)
#define MODEL_ACF_SCALE       1.0761160403e-01f
#define MODEL_ACF_ZP          (-30)
#define MODEL_OUT_SCALE       3.9062500000e-03f
#define MODEL_OUT_ZP          (-128)
