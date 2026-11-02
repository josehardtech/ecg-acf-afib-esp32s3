/**
 * @file autocorr_filter.c
 * @brief Discrete Autocorrelation — Implementation (mean-centered, v2)
 *
 * Computes the same autocorrelation as the Python training code:
 *
 *   Python (extract_acf):
 *     s    = signal - signal.mean()
 *     r0   = dot(s, s)
 *     R[k] = dot(s[:N-k], s[k:]) / r0
 *
 *   This C implementation:
 *     1. mean  = (1/N) · Σ signal[n]
 *     2. s[n]  = signal[n] - mean          (stored in ac->centered[])
 *     3. r0    = dsps_dotprod(s, s, N)     = dot(s, s)
 *     4. R[0]  = 1.0
 *     5. R[k]  = dsps_dotprod(s, s+k, N-k) / r0   for k = 1..N-1
 *
 * CORRECTION vs previous version:
 *   The original code used the raw (non-centered) signal, producing ACF
 *   values that diverge from the training distribution when the signal
 *   has a non-zero mean (DC offset / baseline wander). This caused
 *   incorrect inference results on the ESP32-S3.
 *
 * Memory layout:
 *   ac->acorr    [PSRAM, N floats] — output R_norm[0..N-1]
 *   ac->centered [PSRAM, N floats] — scratch s[0..N-1], reused each call
 */

#include "autocorr_filter.h"

#include <string.h>
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "dsps_dotprod.h"

static const char *TAG = "AUTOCORR";

/* ─── Public API ─────────────────────────────────────────────────────────── */

autocorr_t *autocorr_alloc(size_t signal_len)
{
    uint32_t sram  = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
    uint32_t psram = MALLOC_CAP_SPIRAM;

    autocorr_t *ac = (autocorr_t *)heap_caps_malloc(sizeof(autocorr_t), sram);
    if (!ac) {
        ESP_LOGE(TAG, "Context alloc failed");
        return NULL;
    }
    ac->signal_len = signal_len;
    ac->acorr      = NULL;
    ac->centered   = NULL;

    /* Output buffer in PSRAM — N floats */
    ac->acorr = (float *)heap_caps_malloc(signal_len * sizeof(float), psram);
    if (!ac->acorr) {
        ESP_LOGE(TAG, "acorr buffer alloc failed (PSRAM, %zu bytes)",
                 signal_len * sizeof(float));
        heap_caps_free(ac);
        return NULL;
    }

    /* Scratch buffer for mean-centered signal — N floats in PSRAM */
    ac->centered = (float *)heap_caps_malloc(signal_len * sizeof(float), psram);
    if (!ac->centered) {
        ESP_LOGE(TAG, "centered buffer alloc failed (PSRAM, %zu bytes)",
                 signal_len * sizeof(float));
        heap_caps_free(ac->acorr);
        heap_caps_free(ac);
        return NULL;
    }

    ESP_LOGI(TAG, "Autocorr ready: N=%zu lags, 2x%.0f KB in PSRAM",
             signal_len, signal_len * sizeof(float) / 1024.0f);
    return ac;
}

void autocorr_free(autocorr_t *ac)
{
    if (!ac) return;
    if (ac->centered) { heap_caps_free(ac->centered); ac->centered = NULL; }
    if (ac->acorr)    { heap_caps_free(ac->acorr);    ac->acorr    = NULL; }
    heap_caps_free(ac);
}

void autocorr_compute(autocorr_t *ac, const float *signal)
{
    size_t       N  = ac->signal_len;
    float       *s  = ac->centered;   /* scratch — mean-centered signal */
    float        dot;

    /* ── 1. Calcular media ───────────────────────────────────────────── */
    float mean = 0.0f;
    for (size_t i = 0; i < N; i++) mean += signal[i];
    mean /= (float)N;

    /* ── 2. Centralizar: s[n] = signal[n] - mean ─────────────────────── */
    for (size_t i = 0; i < N; i++) s[i] = signal[i] - mean;

    /* ── 3. R[0] = dot(s, s) ─────────────────────────────────────────── */
    dsps_dotprod_f32(s, s, &dot, (int)N);
    float r0 = dot;

    /* Sinal de energia zero — retorna zeros com R[0]=1 */
    if (r0 < 1e-10f) {
        ESP_LOGW(TAG, "Signal energy ~0 after centering — autocorr set to zero");
        memset(ac->acorr, 0, N * sizeof(float));
        ac->acorr[0] = 1.0f;
        return;
    }

    /* ── 4. R_norm[0] = 1.0 por definicao ───────────────────────────── */
    ac->acorr[0] = 1.0f;

    /* ── 5. Lags k = 1..N-1 ─────────────────────────────────────────── */
    /*
     * R_norm[k] = dot(s[0..N-1-k], s[k..N-1]) / r0
     *
     * dsps_dotprod_f32(s, s+k, &dot, N-k) computa o produto interno
     * dos vetores deslocados — identico ao Python:
     *   np.dot(s[:N-k], s[k:]) / r0
     */
    for (size_t k = 1; k < N; k++) {
        dsps_dotprod_f32(s,       /* s[0..N-1-k] */
                         s + k,   /* s[k..N-1]   */
                         &dot,
                         (int)(N - k));
        ac->acorr[k] = dot / r0;
    }

    ESP_LOGD(TAG, "Autocorr done: mean=%.4f  R[0]=1.000000  R[1]=%.6f  R[N/2]=%.6f",
             mean, ac->acorr[1], ac->acorr[N / 2]);
}
