/**
 * @file butterworth_filter.c
 * @brief Butterworth IIR Bandpass Filter — Implementation using dsps_biquad_f32
 *
 * ── Coefficient design ──────────────────────────────────────────────────────
 *
 * Bilinear Transform (BLT) with frequency pre-warping:
 *
 *   ωa = 2·fs · tan(π·fc / fs)      ← analogue pre-warped angular frequency
 *   k  = (2·fs) / ωa                ← normalised frequency ratio
 *
 * 2nd-order Butterworth HIGH-PASS:
 *   b0 =  k²/(k²+k√2+1)
 *   b1 = -2k²/(k²+k√2+1)
 *   b2 =  k²/(k²+k√2+1)
 *   a1 =  2(1−k²)/(k²+k√2+1)
 *   a2 =  (k²−k√2+1)/(k²+k√2+1)
 *
 * 2nd-order Butterworth LOW-PASS:
 *   b0 =  1/(k²+k√2+1)
 *   b1 =  2/(k²+k√2+1)
 *   b2 =  1/(k²+k√2+1)
 *   a1 =  2(1−k²)/(k²+k√2+1)     (same denominator)
 *   a2 =  (k²−k√2+1)/(k²+k√2+1)
 *
 * ── dsps_biquad_f32 coefficient layout ─────────────────────────────────────
 *
 *  The function signature is:
 *    esp_err_t dsps_biquad_f32(const float *input, float *output, int len,
 *                               float *coeff, float *w);
 *
 *  coeff[] layout expected by dsps_biquad_f32 (5 elements):
 *    coeff[0] = b0
 *    coeff[1] = b1
 *    coeff[2] = b2
 *    coeff[3] = -a1   ← SIGN NEGATED (library adds internally)
 *    coeff[4] = -a2   ← SIGN NEGATED
 *
 *  w[] layout (2 elements, zeroed before each pass):
 *    w[0] = delay register z^-1
 *    w[1] = delay register z^-2
 *
 * ── Zero-phase filtering ────────────────────────────────────────────────────
 *
 *  Forward-backward application (paper Eq. 2):
 *    1. dsps_biquad_f32(HP)  → scratch
 *    2. dsps_biquad_f32(LP)  → signal   (forward pass complete)
 *    3. reverse signal
 *    4. dsps_biquad_f32(HP)  → scratch
 *    5. dsps_biquad_f32(LP)  → signal   (backward pass complete)
 *    6. reverse signal       (restore causal order)
 */

#include "butterworth_filter.h"

#include <math.h>
#include <string.h>

#include "esp_log.h"
#include "esp_heap_caps.h"
#include "dsps_biquad.h"

static const char *TAG = "BW_FILTER";

/* ─── Coefficient computation ────────────────────────────────────────────── */

/**
 * @brief Fill a dsps_biquad_f32-compatible coefficient array for a
 *        2nd-order Butterworth HIGH-PASS filter.
 *
 * @param fc    Cut-off frequency [Hz]
 * @param fs    Sampling frequency [Hz]
 * @param coeff Output array of length 5: [b0, b1, b2, −a1, −a2]
 */
static void compute_hp_coeffs(float fc, float fs, float *coeff)
{
    float omega_a = 2.0f * fs * tanf((float)M_PI * fc / fs);
    float k       = (2.0f * fs) / omega_a;
    float k2      = k * k;
    float sqrt2k  = sqrtf(2.0f) * k;
    float a0      = k2 + sqrt2k + 1.0f;  /* normalisation denominator */

    coeff[0] =  k2 / a0;                       /* b0           */
    coeff[1] = -2.0f * k2 / a0;               /* b1           */
    coeff[2] =  k2 / a0;                       /* b2           */
    coeff[3] =  ( 2.0f * (1.0f - k2) / a0);  /* a1 — dsps_biquad_f32 negates internally */
    coeff[4] =  ((k2 - sqrt2k + 1.0f) / a0); /* a2 — dsps_biquad_f32 negates internally */
}

/**
 * @brief Fill a dsps_biquad_f32-compatible coefficient array for a
 *        2nd-order Butterworth LOW-PASS filter.
 *
 * @param fc    Cut-off frequency [Hz]
 * @param fs    Sampling frequency [Hz]
 * @param coeff Output array of length 5: [b0, b1, b2, −a1, −a2]
 */
static void compute_lp_coeffs(float fc, float fs, float *coeff)
{
    float omega_a = 2.0f * fs * tanf((float)M_PI * fc / fs);
    float k       = (2.0f * fs) / omega_a;
    float k2      = k * k;
    float sqrt2k  = sqrtf(2.0f) * k;
    float a0      = k2 + sqrt2k + 1.0f;

    coeff[0] =  1.0f / a0;                     /* b0 */
    coeff[1] =  2.0f / a0;                     /* b1 */
    coeff[2] =  1.0f / a0;                     /* b2 */
    coeff[3] =  ( 2.0f * (1.0f - k2) / a0);  /* a1 — dsps_biquad_f32 negates internally */
    coeff[4] =  ((k2 - sqrt2k + 1.0f) / a0); /* a2 — dsps_biquad_f32 negates internally */
}

/* ─── Signal utilities ───────────────────────────────────────────────────── */

static void reverse_signal(float *sig, size_t len)
{
    size_t lo = 0, hi = len - 1;
    while (lo < hi) {
        float t   = sig[lo];
        sig[lo++] = sig[hi];
        sig[hi--] = t;
    }
}

/* ─── Public API ─────────────────────────────────────────────────────────── */

butterworth_filter_t *butterworth_filter_alloc(size_t signal_len)
{
    uint32_t sram = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;

    butterworth_filter_t *filt =
        (butterworth_filter_t *)heap_caps_malloc(sizeof(butterworth_filter_t), sram);
    if (!filt) {
        ESP_LOGE(TAG, "Failed to allocate filter context");
        return NULL;
    }
    filt->signal_len = signal_len;

    /* Scratch buffer: output of HP pass, input to LP pass during each sub-pass */
    filt->scratch = (float *)heap_caps_malloc(signal_len * sizeof(float), sram);
    if (!filt->scratch) {
        ESP_LOGW(TAG, "SRAM scratch unavailable — trying PSRAM");
        filt->scratch = (float *)heap_caps_malloc(signal_len * sizeof(float),
                                                  MALLOC_CAP_SPIRAM);
    }
    if (!filt->scratch) {
        ESP_LOGE(TAG, "Scratch buffer allocation failed");
        heap_caps_free(filt);
        return NULL;
    }

    /* Compute coefficients in dsps_biquad_f32 layout [b0,b1,b2,−a1,−a2] */
    compute_hp_coeffs(BW_HP_FC, BW_FS, filt->hp_coeff);
    compute_lp_coeffs(BW_LP_FC, BW_FS, filt->lp_coeff);

    ESP_LOGI(TAG, "Butterworth (dsps_biquad_f32): HP=%.2f Hz, LP=%.2f Hz, fs=%.0f Hz",
             BW_HP_FC, BW_LP_FC, BW_FS);
    ESP_LOGI(TAG, "  HP coeff: [%.6f, %.6f, %.6f, %.6f, %.6f]",
             filt->hp_coeff[0], filt->hp_coeff[1], filt->hp_coeff[2],
             filt->hp_coeff[3], filt->hp_coeff[4]);
    ESP_LOGI(TAG, "  LP coeff: [%.6f, %.6f, %.6f, %.6f, %.6f]",
             filt->lp_coeff[0], filt->lp_coeff[1], filt->lp_coeff[2],
             filt->lp_coeff[3], filt->lp_coeff[4]);

    return filt;
}

void butterworth_filter_free(butterworth_filter_t *filt)
{
    if (!filt) return;
    if (filt->scratch) heap_caps_free(filt->scratch);
    heap_caps_free(filt);
}

void butterworth_filter_apply(butterworth_filter_t *filt, float *signal)
{
    int len = (int)filt->signal_len;

    /*
     * Zero-phase forward-backward filtering — paper Eq. 2:
     *
     *   y_fb[n] = H{ H{x[n]}_rev }_rev
     *
     * Each half uses dsps_biquad_f32 twice: HP→scratch, LP→signal.
     * The delay-line state (hp_state, lp_state) is zeroed before
     * every sub-pass to prevent inter-pass state leakage.
     *
     * dsps_biquad_f32 leverages the Xtensa LX7 MADD/MSUB instructions
     * and is significantly faster than a scalar C loop at 240 MHz.
     */

    /* ── Forward pass ────────────────────────────────────────────────────── */
    memset(filt->hp_state, 0, sizeof(filt->hp_state));
    memset(filt->lp_state, 0, sizeof(filt->lp_state));

    /* HP: signal → scratch */
    dsps_biquad_f32(signal, filt->scratch, len, filt->hp_coeff, filt->hp_state);
    /* LP: scratch → signal */
    dsps_biquad_f32(filt->scratch, signal, len, filt->lp_coeff, filt->lp_state);

    /* ── Reverse ─────────────────────────────────────────────────────────── */
    reverse_signal(signal, (size_t)len);

    /* ── Backward pass ───────────────────────────────────────────────────── */
    memset(filt->hp_state, 0, sizeof(filt->hp_state));
    memset(filt->lp_state, 0, sizeof(filt->lp_state));

    dsps_biquad_f32(signal, filt->scratch, len, filt->hp_coeff, filt->hp_state);
    dsps_biquad_f32(filt->scratch, signal, len, filt->lp_coeff, filt->lp_state);

    /* ── Restore temporal order ──────────────────────────────────────────── */
    reverse_signal(signal, (size_t)len);
}
