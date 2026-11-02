#include "ecg_denoising_pipeline.h"
#include <stdlib.h>
#include "esp_log.h"

static const char *TAG = "ECG_PIPELINE";

ecg_pipeline_t *ecg_pipeline_alloc(size_t signal_len)
{
    ecg_pipeline_t *pipe = (ecg_pipeline_t *)malloc(sizeof(ecg_pipeline_t));
    if (!pipe) { ESP_LOGE(TAG, "Failed to allocate pipeline"); return NULL; }
    pipe->signal_len = signal_len;

    pipe->bw = butterworth_filter_alloc(signal_len);
    if (!pipe->bw) { free(pipe); return NULL; }

    pipe->loess = loess_filter_alloc(signal_len);
    if (!pipe->loess) { butterworth_filter_free(pipe->bw); free(pipe); return NULL; }

    pipe->nlm = nlm_filter_alloc(signal_len);
    if (!pipe->nlm) {
        butterworth_filter_free(pipe->bw);
        loess_filter_free(pipe->loess);
        free(pipe);
        return NULL;
    }
    ESP_LOGI(TAG, "Pipeline ready (%zu samples)", signal_len);
    return pipe;
}

void ecg_pipeline_free(ecg_pipeline_t *pipe)
{
    if (!pipe) return;
    butterworth_filter_free(pipe->bw);
    loess_filter_free(pipe->loess);
    nlm_filter_free(pipe->nlm);
    free(pipe);
}

void ecg_pipeline_apply(ecg_pipeline_t *pipe, float *signal)
{
    butterworth_filter_apply(pipe->bw, signal);
    loess_filter_apply(pipe->loess, signal);
    nlm_filter_apply(pipe->nlm, signal);
}
