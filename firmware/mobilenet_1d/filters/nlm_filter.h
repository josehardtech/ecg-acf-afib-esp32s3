#pragma once
/**
 * @file nlm_filter.h
 * @brief Non-Local Means (NLM) Denoising Filter for ECG Signals
 *
 * Implements the 1-D Non-Local Means algorithm (Buades et al., 2005) using
 * Darbon's fast cumulative-sum acceleration (Darbon et al., 2008).
 *
 * ESP-DSP acceleration (Xtensa LX7 FPU):
 *   dsps_sub_f32  — vectorised subtraction for the shifted-difference array
 *   dsps_mul_f32  — vectorised element-wise squaring
 *   Both replace the inner scalar loops of Step 1 in the Darbon algorithm.
 *
 * Filtered output at sample i (paper Eq. 7):
 *
 *   NLM(x)[i] = (1/Z_i) · Σ_{j∈S_i} w(i,j)·x[j]
 *   Z_i       = Σ_{j∈S_i} w(i,j)
 *
 * Similarity weights (paper Eq. 8):
 *   w(i,j) = exp(−d(P_i,P_j) / h)
 *   d(P_i,P_j) = Σ_{k=−p}^{p} (x[i+k]−x[j+k])²
 *
 * Smoothing parameter h (paper Eq. 9):
 *   h = 2·N_patch·λ²,  λ = k_λ·σ̂,  σ̂ = 1.4826·median(|Δx|)/√2
 *
 * Parameters @ 500 Hz:
 *   p = 13 samples  (patch half-width, ≈26 ms)
 *   M = 694 samples (search half-width, ≈1.39 s)
 *
 * References:
 *   Buades, Coll, Morel. CVPR 2005.
 *   Darbon et al. ISBI 2008.
 *   Jiang, Gao, Wang. CinC 2013.
 *   ESP-DSP: dsps_math.h (dsps_sub_f32, dsps_mul_f32).
 */

#include <stdint.h>
#include <stddef.h>
#include "dsps_math.h"   /* dsps_sub_f32, dsps_mul_f32 */

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Configuration ──────────────────────────────────────────────────────── */

#define NLM_FS               500
#define NLM_PATCH_HALF        13
#define NLM_PATCH_LEN        (2 * NLM_PATCH_HALF + 1)   /* 27 */
#define NLM_SEARCH_HALF      694
#define NLM_K_LAMBDA         1.5f
#define NLM_MAD_FACTOR       1.4826f
#define NLM_EXP_TABLE_SIZE   512
#define NLM_EXP_MAX          12.0f

/* ─── Data Structure ─────────────────────────────────────────────────────── */

/**
 * @brief NLM filter context.
 */
typedef struct {
    size_t signal_len;

    /**
     * diff_sq: squared-difference array D_δ[], length = signal_len.
     * Computed by dsps_sub_f32 + dsps_mul_f32 for each shift δ.
     * Allocated in PSRAM (large, written once per shift).
     */
    float  *diff_sq;

    /** Cumulative-sum array C_δ[], length = signal_len. SRAM (float). */
    float  *cum_sum;

    /** Accumulated weighted signal, length = signal_len. PSRAM. */
    float  *output;

    /** Weight normaliser, length = signal_len. PSRAM. */
    float  *norm;

    /**
     * diff_buf: first-difference scratch for σ̂ estimation, length = signal_len.
     * Also computed via dsps_sub_f32.  PSRAM.
     */
    float  *diff_buf;

    /**
     * exp_lut: 512-entry exp(−t) table for t∈[0,12].
     * Allocated in internal SRAM for fast access during the inner loop.
     */
    float  *exp_lut;
} nlm_filter_t;

/* ─── Public API ─────────────────────────────────────────────────────────── */

/**
 * @brief Allocate and initialise an NLM filter instance.
 * @param signal_len  Samples per ECG window (e.g. 5000).
 * @return Pointer to the context, or NULL on failure.
 */
nlm_filter_t *nlm_filter_alloc(size_t signal_len);

/**
 * @brief Free all resources held by an NLM filter instance.
 */
void nlm_filter_free(nlm_filter_t *filt);

/**
 * @brief Apply NLM denoising to an ECG signal (in-place).
 * @param filt    Initialised NLM context.
 * @param signal  Float array of length filt->signal_len.
 */
void nlm_filter_apply(nlm_filter_t *filt, float *signal);

#ifdef __cplusplus
}
#endif
