cmake_minimum_required(VERSION 3.16)

# ── Extra component directories ────────────────────────────────────────────
# filters/  : Butterworth, LOESS, NLM, autocorrelation, pipeline
# inference/: TFLite Micro wrapper (tflite_inference.cc)
set(EXTRA_COMPONENT_DIRS
    "${CMAKE_SOURCE_DIR}/filters"
    "${CMAKE_SOURCE_DIR}/inference"
)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(ecg_denoising)
