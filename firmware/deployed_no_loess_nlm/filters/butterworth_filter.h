#pragma once
/**
 * @file butterworth_filter.h
 * @brief Butterworth IIR Bandpass Filter — ESP-DSP accelerated
 *
 * Implements a cascaded 2nd-order Butterworth IIR filter pipeline:
 *   Stage 1 — High-pass  fc = 0.5 Hz  (removes baseline wander)
 *   Stage 2 — Low-pass   fc = 40 Hz   (removes EMG / powerline noise)
 *
 * Each stage is represented as one biquad section in Direct-Form II
 * Transposed layout and executed by the ESP-DSP library function:
 *
 *   esp_err_t dsps_biquad_f32(const float *input,
 *                              float       *output,
 *                              int          len,
 *                              float       *coeff,   // [b0,b1,b2,a1,a2]
 *                              float       *w);      // [w0, w1] delay line
 *
 * dsps_biquad_f32 is hand-optimised for the Xtensa LX7 FPU present in the
 * ESP32S3, using fused-multiply-add (MADD) instructions and minimising
 * load/store stalls.  It is the canonical way to run biquad filters on this
 * chip and replaces any hand-written scalar loop.
 *
 * Both stages are applied with forward-backward (zero-phase) filtering
 * (paper Eq. 2), doubling the effective order to 4th per stage.
 *
 * Reference: Paper Section II-C-1, Equations (1)-(3).
 *            ESP-DSP docs: dsps_biquad_f32 (esp_dsp/modules/iir).
 * Hardware:  XIAO ESP32S3 Sense, Xtensa LX7 @ 240 MHz, 500 Hz fs.
 *
 * Memory:
 *   Coefficients and delay-line arrays: internal SRAM
 *   Scratch buffer for signal reversal: internal SRAM (fallback PSRAM)
 */

#include <stdint.h>
#include <stddef.h>
#include "esp_heap_caps.h"
#include "dsps_biquad.h"   /* ESP-DSP: dsps_biquad_f32 */

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Configuration ──────────────────────────────────────────────────────── */

#define BW_FS       500.0f   /**< Sampling frequency [Hz]       */
#define BW_HP_FC      0.5f   /**< High-pass cut-off  [Hz]       */
#define BW_LP_FC     40.0f   /**< Low-pass  cut-off  [Hz]       */

/**
 * @brief Number of biquad sections per filter stage.
 *
 *  A 2nd-order Butterworth requires exactly 1 biquad section.
 *  Defined here so dsps_biquad_f32 coefficient/state arrays are
 *  correctly sized; expand to 2 if a 4th-order design is ever needed.
 */
#define BW_NUM_BIQUADS  1

/**
 * @brief Coefficient array length required by dsps_biquad_f32.
 *
 *  dsps_biquad_f32 expects 5 floats per section: [b0, b1, b2, a1, a2].
 *  Note: a1 and a2 must be stored NEGATED relative to the standard
 *  difference-equation convention:
 *
 *    y[n] = b0·x[n] + b1·x[n-1] + b2·x[n-2]
 *                   − a1·y[n-1]  − a2·y[n-2]
 *
 *  dsps_biquad_f32 stores the feedback coefficients with sign flipped
 *  internally, so we pass them as-is from the bilinear-transform design
 *  (a1, a2 already have positive sign after BLT normalisation); the
 *  library negates them internally.  Always verify against the ESP-DSP
 *  source (dsps_biquad_f32.c) when updating SDK versions.
 */
#define BW_COEFF_LEN    (BW_NUM_BIQUADS * 5)   /* = 5 */

/**
 * @brief Delay-line (state) array length required by dsps_biquad_f32.
 *
 *  2 floats per biquad section: [w0, w1].
 */
#define BW_STATE_LEN    (BW_NUM_BIQUADS * 2)   /* = 2 */

/* ─── Data Structure ─────────────────────────────────────────────────────── */

/**
 * @brief Butterworth filter context.
 *
 *  All arrays sized to the exact layout expected by dsps_biquad_f32
 *  and allocated in internal SRAM for minimum access latency.
 */
typedef struct {
    size_t signal_len;   /**< Samples per ECG segment (e.g. 5000) */

    /**
     * HP biquad coefficients [b0, b1, b2, a1, a2] — SRAM.
     * Computed once during _alloc via the bilinear transform.
     */
    float hp_coeff[BW_COEFF_LEN];

    /**
     * LP biquad coefficients [b0, b1, b2, a1, a2] — SRAM.
     */
    float lp_coeff[BW_COEFF_LEN];

    /**
     * HP biquad delay-line state [w0, w1] — SRAM.
     * Reset to zero before each forward or backward pass.
     */
    float hp_state[BW_STATE_LEN];

    /**
     * LP biquad delay-line state [w0, w1] — SRAM.
     */
    float lp_state[BW_STATE_LEN];

    /**
     * Scratch buffer (length = signal_len) used as the output array
     * for dsps_biquad_f32 during each sub-pass, then copied back.
     * Allocated in internal SRAM; falls back to PSRAM if unavailable.
     */
    float *scratch;
} butterworth_filter_t;

/* ─── Public API ─────────────────────────────────────────────────────────── */

/**
 * @brief Allocate and initialise a Butterworth filter instance.
 *
 *  Computes HP and LP biquad coefficients via the bilinear transform
 *  and stores them in the dsps_biquad_f32-compatible layout.
 *  Allocates all buffers in internal SRAM where possible.
 *
 * @param signal_len  Samples per ECG window (e.g. 5000 @ 500 Hz).
 * @return Pointer to the context, or NULL on failure.
 */
butterworth_filter_t *butterworth_filter_alloc(size_t signal_len);

/**
 * @brief Release all resources held by a filter instance.
 * @param filt  Context from butterworth_filter_alloc.
 */
void butterworth_filter_free(butterworth_filter_t *filt);

/**
 * @brief Apply the cascaded zero-phase bandpass filter (in-place).
 *
 *  Executes dsps_biquad_f32 four times (HP forward, LP forward,
 *  HP backward, LP backward) to achieve zero-phase response.
 *
 * @param filt    Initialised filter context.
 * @param signal  Float array of length filt->signal_len (modified in-place).
 */
void butterworth_filter_apply(butterworth_filter_t *filt, float *signal);

#ifdef __cplusplus
}
#endif
