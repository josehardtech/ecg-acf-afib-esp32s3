idf_component_register(
    SRCS
        "butterworth_filter.c"
        "loess_filter.c"
        "nlm_filter.c"
        "ecg_denoising_pipeline.c"
        "autocorr_filter.c"
    INCLUDE_DIRS "."
    REQUIRES esp_system heap log espressif__esp-dsp
)
