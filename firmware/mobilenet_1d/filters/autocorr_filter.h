#pragma once
/**
 * @file autocorr_filter.h
 * @brief Discrete Autocorrelation — ESP-DSP accelerated (no FFT)
 *
 * Computes the normalized discrete autocorrelation of a MEAN-CENTERED signal,
 * identical to the Python implementation used during model training:
 *
 *   s[n]     = x[n] - mean(x)                          (centering)
 *   R[k]     = Σ_{n=0}^{N-1-k} s[n] · s[n+k]          (unnormalized)
 *   R_norm[k] = R[k] / R[0]                             (R[0] = 1 by def.)
 *
 * IMPORTANT — alignment with Python training code:
 *   The Python extract_acf() subtracts the mean before computing the
 *   autocorrelation.  Without this step the ACF values differ significantly
 *   from the training distribution, causing incorrect inference results.
 *
 * Acceleration strategy:
 *   Each lag k requires the dot product of s[0..N-1-k] with s[k..N-1].
 *   dsps_dotprod_f32 computes this using Xtensa LX7 MADD instructions,
 *   running ~4x faster than a scalar loop at 240 MHz.
 *
 * Memory:
 *   acorr  : N floats — output buffer (PSRAM)
 *   centered: N floats — mean-centered scratch buffer (PSRAM)
 *   Both allocated at autocorr_alloc() time and reused across calls.
 *
 * Reference: Oppenheim & Schafer, "Discrete-Time Signal Processing", Eq. 2.155.
 *            ESP-DSP: dsps_dotprod_f32 (esp_dsp/modules/dotprod).
 */

#include <stddef.h>
#include "esp_heap_caps.h"
#include "dsps_dotprod.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Data Structure ─────────────────────────────────────────────────────── */

/**
 * @brief Autocorrelation context.
 */
typedef struct {
    size_t  signal_len;   /**< N — number of samples / number of lags         */
    float  *acorr;        /**< Output: R_norm[0..N-1], length N  (PSRAM)      */
    float  *centered;     /**< Scratch: mean-centered signal     (PSRAM)      */
} autocorr_t;

/* ─── Public API ─────────────────────────────────────────────────────────── */

/**
 * @brief Allocate and initialise an autocorrelation context.
 *
 * Allocates two PSRAM buffers: acorr (output) and centered (scratch).
 *
 * @param signal_len  Number of input samples (= number of output lags).
 * @return Pointer to context, or NULL on allocation failure.
 */
autocorr_t *autocorr_alloc(size_t signal_len);

/**
 * @brief Free all resources held by an autocorrelation context.
 * @param ac  Context from autocorr_alloc().
 */
void autocorr_free(autocorr_t *ac);

/**
 * @brief Compute the normalised discrete autocorrelation of a mean-centred signal.
 *
 *  Steps:
 *    1. Compute mean of signal[0..N-1]
 *    2. Store s[n] = signal[n] - mean  in ac->centered[]
 *    3. Compute R_norm[k] = dot(s, s+k) / dot(s, s)  for k = 0..N-1
 *    4. Write result to ac->acorr[0..N-1]  (ac->acorr[0] = 1.0)
 *
 *  The input signal is NOT modified.
 *
 * @param ac      Initialised context.
 * @param signal  Input signal of length ac->signal_len (read-only).
 */
void autocorr_compute(autocorr_t *ac, const float *signal);

#ifdef __cplusplus
}
#endif
