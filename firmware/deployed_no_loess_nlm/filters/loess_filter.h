#pragma once
/**
 * @file loess_filter.h
 * @brief Robust LOESS (Locally Estimated Scatterplot Smoothing) Filter
 *
 * Implements the two-pass robust LOESS variant described in the paper
 * (Section II-C-2, Equations 4–6) for baseline drift suppression and
 * transient outlier rejection on ECG signals.
 *
 * Algorithm:
 *   For each sample i, a local 1st-degree polynomial is fitted by
 *   weighted least-squares over a neighbourhood N(i) of span W samples:
 *
 *     ŷ_i = argmin_β Σ_{j∈N(i)} w_ij · (y_j − β₀ − β₁·(x_j − x_i))²
 *
 *   Tricube neighbourhood weights (Eq. 5):
 *
 *     w(u) = (1 − |u|³)³   for |u| < 1,  else 0
 *     u    = d_ij / d_max  (normalised distance from window centre)
 *
 *   Robust extension (two-pass MAD outlier rejection, Eq. 6):
 *     Pass 1: compute residuals r_i = y_i − ŷ_i
 *             σ̂ = 1.4826 · median(|r_i − median(r_i)|)   (MAD estimator)
 *     Pass 2: samples with |r_i| > 6·σ̂ receive zero weight (bisquare mask)
 *
 * Parameters (paper defaults, 500 Hz):
 *   Window span  W = 151 samples  (~300 ms)
 *   Exceeds QRS duration (60-100 ms) → QRS morphology preserved.
 *
 * Memory:
 *   All working arrays (trend, residuals, outlier mask) are allocated in
 *   internal SRAM to avoid PSRAM cache coherency issues during iterative
 *   computation.
 *
 * Reference: Paper Section II-C-2; Cleveland (1979); Cleveland & Devlin (1988).
 */

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Configuration ──────────────────────────────────────────────────────── */

/**
 * @brief LOESS window span in samples.
 *
 *  151 samples @ 500 Hz ≈ 302 ms.  Paper specifies ≈300 ms (Section II-C-2).
 *  Must be an odd number so the window is symmetric around each centre point.
 */
#define LOESS_SPAN       151

/** Half-width of the LOESS window (samples on each side of centre). */
#define LOESS_HALF_SPAN  ((LOESS_SPAN - 1) / 2)   /* = 75 */

/**
 * @brief MAD outlier rejection threshold multiplier.
 *
 *  Samples whose |residual| > LOESS_OUTLIER_SIGMA · σ̂ are assigned
 *  zero weight in the second pass. Paper specifies 6·σ̂.
 */
#define LOESS_OUTLIER_SIGMA   6.0f

/**
 * @brief MAD consistency factor for Gaussian assumption.
 *
 *  σ̂ = MAD_FACTOR · median(|r_i − median(r_i)|)
 *  1.4826 makes σ̂ a consistent estimator of σ under normality.
 */
#define LOESS_MAD_FACTOR      1.4826f

/* ─── Data Structures ────────────────────────────────────────────────────── */

/**
 * @brief LOESS filter context.
 *
 *  All working buffers are pre-allocated at construction time to avoid
 *  dynamic allocation during real-time processing.
 */
typedef struct {
    size_t signal_len;   /**< Number of samples per ECG segment            */

    /** Tricube weights for a full window (length = LOESS_SPAN). */
    float *tricube_weights;   /* SRAM */

    /** Intermediate trend estimate (length = signal_len). */
    float *trend;             /* SRAM */

    /** Residuals buffer (length = signal_len). */
    float *residuals;         /* SRAM */

    /** Sorted residual copy for median computation (length = signal_len). */
    float *sorted_buf;        /* SRAM */

    /** Per-sample outlier mask: 1 = outlier (skip), 0 = valid (length = signal_len). */
    unsigned char *outlier_mask;  /* SRAM */
} loess_filter_t;

/* ─── Public API ─────────────────────────────────────────────────────────── */

/**
 * @brief Allocate and initialise a LOESS filter instance.
 *
 *  Pre-computes the tricube weight table for the configured span.
 *  All working buffers are allocated in internal SRAM when available.
 *
 * @param signal_len  Number of samples per ECG segment (e.g. 5000).
 * @return Pointer to the context, or NULL on allocation failure.
 */
loess_filter_t *loess_filter_alloc(size_t signal_len);

/**
 * @brief Free all resources held by a LOESS filter instance.
 * @param filt  Context returned by loess_filter_alloc.
 */
void loess_filter_free(loess_filter_t *filt);

/**
 * @brief Apply two-pass robust LOESS smoothing to an ECG signal (in-place).
 *
 *  The function removes the estimated local trend from the signal:
 *
 *    signal_out[i] = signal_in[i] − trend[i]
 *
 *  so that the output is a baseline-corrected, detrended signal.
 *
 * @param filt    Initialised LOESS context.
 * @param signal  Input/output array of length filt->signal_len.
 */
void loess_filter_apply(loess_filter_t *filt, float *signal);

#ifdef __cplusplus
}
#endif
