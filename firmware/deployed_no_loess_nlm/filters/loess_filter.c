/**
 * @file loess_filter.c
 * @brief Robust LOESS Smoothing Filter — Implementation
 *
 * For efficiency on the ESP32S3, the local polynomial fit at each point
 * exploits the fact that the tricube weights are the same for every interior
 * window (only the boundary windows require truncation).
 *
 * Weighted least-squares solution for local linear fit:
 *
 *   Given samples y_j (j in window), positions t_j = j − i (centred),
 *   weights w_j (tricube × outlier_mask):
 *
 *     S0 = Σ w_j            S1 = Σ w_j·t_j        S2 = Σ w_j·t_j²
 *     T0 = Σ w_j·y_j        T1 = Σ w_j·t_j·y_j
 *
 *   Normal equations (2×2 system):
 *     β₀ = (S2·T0 − S1·T1) / (S0·S2 − S1²)
 *     β₁ = (S0·T1 − S1·T0) / (S0·S2 − S1²)
 *
 *   Fitted value at centre:  ŷ_i = β₀   (t=0 → β₁ term vanishes)
 */

#include "loess_filter.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "esp_log.h"
#include "esp_heap_caps.h"

static const char *TAG = "LOESS";

/* ─── Internal utilities ─────────────────────────────────────────────────── */

/** Tricube kernel: w(u) = (1 − |u|³)³ for |u| < 1. */
static inline float tricube(float u)
{
    if (u >= 1.0f || u <= -1.0f) return 0.0f;
    float t = 1.0f - fabsf(u) * fabsf(u) * fabsf(u);
    return t * t * t;
}

/** Compare function for qsort (ascending float). */
static int float_cmp(const void *a, const void *b)
{
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

/**
 * @brief Compute the median of an array (non-destructive via sorted_buf).
 *
 * @param buf        Input array
 * @param sorted_buf Scratch buffer of length >= n for sorting
 * @param n          Number of elements
 * @return Median value
 */
static float compute_median(const float *buf, float *sorted_buf, size_t n)
{
    memcpy(sorted_buf, buf, n * sizeof(float));
    qsort(sorted_buf, n, sizeof(float), float_cmp);
    if (n % 2 == 1) {
        return sorted_buf[n / 2];
    } else {
        return 0.5f * (sorted_buf[n / 2 - 1] + sorted_buf[n / 2]);
    }
}

/**
 * @brief Estimate the robust scale via Median Absolute Deviation (Eq. 6).
 *
 *  σ̂ = 1.4826 · median(|r_i − median(r_i)|)
 *
 * @param residuals  Array of length n
 * @param sorted_buf Scratch buffer of length >= n
 * @param n          Number of elements
 * @return Robust scale estimate σ̂
 */
static float compute_mad(const float *residuals, float *sorted_buf, size_t n)
{
    float med = compute_median(residuals, sorted_buf, n);

    /* Compute absolute deviations |r_i - median| in-place in sorted_buf */
    for (size_t i = 0; i < n; i++) {
        sorted_buf[i] = fabsf(residuals[i] - med);
    }

    float mad_val = compute_median(sorted_buf, sorted_buf, n);
    /* sorted_buf is now overwritten by the second median call — that's fine */

    return LOESS_MAD_FACTOR * mad_val;
}

/**
 * @brief Pre-compute tricube weights for a full-width interior window.
 *
 *  The weight for position offset t ∈ [−H, H] is:
 *    w[t + H] = tricube(t / H)
 *  where H = LOESS_HALF_SPAN.
 *
 * @param weights Output array of length LOESS_SPAN.
 */
static void precompute_tricube_weights(float *weights)
{
    float H = (float)LOESS_HALF_SPAN;
    for (int t = -LOESS_HALF_SPAN; t <= LOESS_HALF_SPAN; t++) {
        weights[t + LOESS_HALF_SPAN] = tricube((float)t / H);
    }
}

/**
 * @brief Fit the local linear model for sample i and return ŷ_i.
 *
 *  Uses float arithmetic throughout (matching the optimised ecg_filters.c
 *  implementation) to avoid double-precision overhead on Xtensa LX7.
 *  The outlier_mask is a byte array: 1 = outlier (skip), 0 = valid.
 */
static float loess_fit_at(const float        *signal,
                          size_t              signal_len,
                          size_t              i,
                          const float        *tw,          /* unused — kept for API compat */
                          const unsigned char *outlier_mask)
{
    int half_span = LOESS_HALF_SPAN;
    int start = (int)i - half_span; if (start < 0) start = 0;
    int end   = (int)i + half_span; if (end >= (int)signal_len) end = (int)signal_len - 1;

    int max_dist = (i - start > (size_t)(end - (int)i))
                   ? (int)(i - start) : (end - (int)i);
    if (max_dist == 0) max_dist = 1;

    float sum_w   = 0.0f, sum_wx  = 0.0f, sum_wy  = 0.0f;
    float sum_wxx = 0.0f, sum_wxy = 0.0f;

    for (int j = start; j <= end; j++) {
        if (outlier_mask != NULL && outlier_mask[j] == 1) continue;

        float dist = (float)abs(j - (int)i) / (float)max_dist;
        float w    = tricube(dist);
        float x_local = (float)(j - (int)i);
        float y_val   = signal[j];

        sum_w   += w;
        sum_wx  += w * x_local;
        sum_wy  += w * y_val;
        sum_wxx += w * x_local * x_local;
        sum_wxy += w * x_local * y_val;
    }

    float denom = sum_w * sum_wxx - sum_wx * sum_wx;
    if (denom < 1e-9f) {
        return (sum_w > 0.0f) ? sum_wy / sum_w : signal[i];
    }
    return (sum_wy * sum_wxx - sum_wx * sum_wxy) / denom;
}

/* ─── Public API implementation ──────────────────────────────────────────── */

loess_filter_t *loess_filter_alloc(size_t signal_len)
{
    uint32_t sram_flags = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;

    loess_filter_t *filt =
        (loess_filter_t *)heap_caps_malloc(sizeof(loess_filter_t), sram_flags);
    if (!filt) {
        ESP_LOGE(TAG, "Failed to allocate LOESS context");
        return NULL;
    }
    filt->signal_len = signal_len;

    /* Allocate working buffers in internal SRAM */
    filt->tricube_weights = (float *)heap_caps_malloc(LOESS_SPAN * sizeof(float), sram_flags);
    filt->trend           = (float *)heap_caps_malloc(signal_len  * sizeof(float), sram_flags);
    filt->residuals       = (float *)heap_caps_malloc(signal_len  * sizeof(float), sram_flags);
    filt->sorted_buf      = (float *)heap_caps_malloc(signal_len  * sizeof(float), sram_flags);
    filt->outlier_mask    = (unsigned char *)heap_caps_malloc(signal_len * sizeof(unsigned char), sram_flags);

    if (!filt->tricube_weights || !filt->trend || !filt->residuals ||
        !filt->sorted_buf      || !filt->outlier_mask) {
        ESP_LOGE(TAG, "Buffer allocation failed — SRAM may be exhausted");
        loess_filter_free(filt);
        return NULL;
    }

    /* Pre-compute the full-span tricube weight table */
    precompute_tricube_weights(filt->tricube_weights);

    ESP_LOGI(TAG, "LOESS filter initialised: span=%d samples (~%d ms @ 500 Hz)",
             LOESS_SPAN, (LOESS_SPAN * 1000) / 500);
    return filt;
}

void loess_filter_free(loess_filter_t *filt)
{
    if (!filt) return;
    if (filt->tricube_weights) heap_caps_free(filt->tricube_weights);
    if (filt->trend)           heap_caps_free(filt->trend);
    if (filt->residuals)       heap_caps_free(filt->residuals);
    if (filt->sorted_buf)      heap_caps_free(filt->sorted_buf);
    if (filt->outlier_mask)    heap_caps_free(filt->outlier_mask);
    heap_caps_free(filt);
}

void loess_filter_apply(loess_filter_t *filt, float *signal)
{
    size_t N = filt->signal_len;

    /* ── Pass 1: Fit trend without outlier mask ─────────────────────────── */
    for (size_t i = 0; i < N; i++) {
        filt->trend[i] = loess_fit_at(signal, N, i, NULL, NULL);
    }

    /* Compute residuals r_i = y_i − ŷ_i */
    for (size_t i = 0; i < N; i++) {
        filt->residuals[i] = signal[i] - filt->trend[i];
    }

    /* Estimate MAD: copy |residuals| into sorted_buf, qsort, take median */
    for (size_t i = 0; i < N; i++) {
        filt->sorted_buf[i] = fabsf(filt->residuals[i]);
    }
    qsort(filt->sorted_buf, N, sizeof(float), float_cmp);
    float mad       = filt->sorted_buf[N / 2];
    float threshold = LOESS_OUTLIER_SIGMA * mad;

    /* Build outlier mask: 1 = outlier (skip), 0 = valid */
    for (size_t i = 0; i < N; i++) {
        filt->outlier_mask[i] = (fabsf(filt->residuals[i]) > threshold) ? 1 : 0;
    }

    ESP_LOGD(TAG, "Pass 1: mad=%.6f, threshold=%.6f", mad, threshold);

    /* ── Pass 2: Refit with outlier mask ────────────────────────────────── */
    for (size_t i = 0; i < N; i++) {
        filt->trend[i] = loess_fit_at(signal, N, i, NULL, filt->outlier_mask);
    }

    /* ── Baseline correction: subtract trend from signal ────────────────── */
    for (size_t i = 0; i < N; i++) {
        signal[i] -= filt->trend[i];
    }
}
