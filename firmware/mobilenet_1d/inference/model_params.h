/* model_params.h — MobileNet-1D baseline — gerado por export_firmware.py em 2026-11-02T10:16:49 */
#pragma once
#define MODEL_SHA256          "8de128cdf55878cc3425b698ed3fa56400a2a9d8a9b96863fc5a31f213f2b080"
#define MODEL_VERSION_TAG     "MobileNet-1D baseline"
#define MODEL_TFLITE_BYTES    105768
#define MODEL_NUM_INPUTS      1
#define MODEL_USES_ACF        0
#define MODEL_NUM_OPS         8
/* Decisão: AFIB se saída INT8 >= MODEL_THRESHOLD_Q  (equivale a p >= MODEL_TAU) */
#define MODEL_TAU             0.64843750f
#define MODEL_THRESHOLD_Q     (38)
#define MODEL_RAW_INPUT_INDEX 0
#define MODEL_ACF_INPUT_INDEX -1
#define MODEL_RAW_SCALE       9.0415552258e-03f
#define MODEL_RAW_ZP          (0)
#define MODEL_ACF_SCALE       1.0000000000e+00f
#define MODEL_ACF_ZP          (0)
#define MODEL_OUT_SCALE       3.9062500000e-03f
#define MODEL_OUT_ZP          (-128)
