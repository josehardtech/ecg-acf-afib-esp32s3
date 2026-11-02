idf_component_register(
    SRCS "main.cpp"
    INCLUDE_DIRS "."
    REQUIRES filters inference esp_timer fatfs sdmmc nvs_flash log heap freertos
)
