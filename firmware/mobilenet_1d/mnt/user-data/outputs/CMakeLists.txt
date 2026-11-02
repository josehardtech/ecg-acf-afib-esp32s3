cmake_minimum_required(VERSION 3.16)

# ── Project name ──────────────────────────────────────────────────────────────
set(PROJECT_NAME "ecg_denoising")

# ── Extra component directories ───────────────────────────────────────────────
#  Tell ESP-IDF where to find our custom 'filters' component.
#  The path is relative to this CMakeLists.txt file.
set(EXTRA_COMPONENT_DIRS
    "${CMAKE_SOURCE_DIR}"   # makes 'filters/' visible as a component
)

# ── ESP-IDF project macro ─────────────────────────────────────────────────────
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(${PROJECT_NAME})
