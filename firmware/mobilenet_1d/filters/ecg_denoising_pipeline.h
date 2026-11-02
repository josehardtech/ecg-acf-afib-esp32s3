#pragma once
#include <stddef.h>
#include "butterworth_filter.h"
#include "loess_filter.h"
#include "nlm_filter.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    butterworth_filter_t *bw;
    loess_filter_t       *loess;
    nlm_filter_t         *nlm;
    size_t                signal_len;
} ecg_pipeline_t;

ecg_pipeline_t *ecg_pipeline_alloc(size_t signal_len);
void ecg_pipeline_free(ecg_pipeline_t *pipe);
void ecg_pipeline_apply(ecg_pipeline_t *pipe, float *signal);

#ifdef __cplusplus
}
#endif
