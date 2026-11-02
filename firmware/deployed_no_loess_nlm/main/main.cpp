/**
 * @file main.cpp
 * @brief ECG Batch Classification — XIAO ESP32S3 Sense
 *        Model: ECGHybridV15 SemLOESSNLM (ablation: apenas Butterworth)
 *
 * ── Overview ─────────────────────────────────────────────────────────────────
 *
 *  Scans ALL .csv files in the root of the SD card, applies the full pipeline
 *  on each one, and appends one result row per file to /sd/results.csv.
 *
 *  Pipeline per file:
 *    Stage 1  Butterworth IIR bandpass  (0.5–40 Hz, zero-phase)
 *    Stage 2  Discrete Autocorrelation  (N=5000 lags, normalised)
 *             [LOESS and NLM removed — ablation SemLOESSNLM]
 *    Stage 3  TFLite Micro inference    (ECGHybridV15 SemLOESSNLM INT8, AFIB vs SR)
 *
 * ── Output file (/sd/results.csv) ────────────────────────────────────────────
 *
 *  filename, label, probability, int8_out, <stage timings…>, total_us
 *  (int8_out = raw INT8 output; AFIB if int8_out >= MODEL_THRESHOLD_Q)
 *  ...
 *
 *  Files named "results.csv" or "timing_log.csv" are skipped automatically.
 *
 * ── SD card wiring (XIAO ESP32S3 Sense onboard slot) ─────────────────────────
 *  CLK → GPIO 7 | CMD → GPIO 9 | DATA0 → GPIO 8 | 1-bit bus | FAT32
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <cmath>

/* ESP-IDF core */
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_err.h"
#include "nvs_flash.h"

/* SD card */
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"
#include "driver/sdmmc_host.h"

/* Directory listing */
#include <dirent.h>

/* Our components */
#include "ecg_denoising_pipeline.h"
#include "autocorr_filter.h"
#include "tflite_inference.h"

/* Memory report (IEEE Access revision, Reviewer 1 item 6) */
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

/* ─── Configuration ──────────────────────────────────────────────────────── */

static const char *TAG = "ECG_BATCH";

/* MEM_REPORT_MAX_FILES: 0 = processa todos os arquivos do SD (rodada completa, 4.025).
 * Para só medir memória basta um lote pequeno (ex.: 20): o pico de heap é atingido
 * já na primeira inferência e não cresce depois. */
static constexpr int    MEM_REPORT_MAX_FILES = 0;

static constexpr char   SD_MOUNT[]          = "/sd";
static constexpr char   RESULTS_FILE[]      = "/sd/results.csv";
static constexpr char   TIMING_FILE[]       = "/sd/timing_log.csv";
static constexpr size_t ECG_SAMPLES         = 5000;
static constexpr size_t CSV_LINE_BUF        = 64;

/* ─── Signal buffers in PSRAM ────────────────────────────────────────────── */

static float ecg_buf [ECG_SAMPLES] EXT_RAM_BSS_ATTR;   /* ECG após pipeline    */
static float acf_norm[ECG_SAMPLES] EXT_RAM_BSS_ATTR;   /* ACF normalizada (StandardScaler) */

/* ─── Memory report ──────────────────────────────────────────────────────────
 * Heap (alocação dinâmica) por região, no instante e no pior caso desde o boot:
 *   used  = total - free agora
 *   peak  = total - minimum_free (marca d'água mínima desde o boot)
 * A memória estática (.data/.bss na SRAM e EXT_RAM_BSS na PSRAM) NÃO está no heap:
 * ela sai de `idf.py size` e deve ser somada à memória de heap no artigo.
 */
static void mem_report(const char *stage)
{
    const uint32_t CAP_SRAM  = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
    const uint32_t CAP_PSRAM = MALLOC_CAP_SPIRAM;
    size_t st = heap_caps_get_total_size(CAP_SRAM);
    size_t sf = heap_caps_get_free_size(CAP_SRAM);
    size_t sm = heap_caps_get_minimum_free_size(CAP_SRAM);
    size_t pt = heap_caps_get_total_size(CAP_PSRAM);
    size_t pf = heap_caps_get_free_size(CAP_PSRAM);
    size_t pm = heap_caps_get_minimum_free_size(CAP_PSRAM);
    UBaseType_t stack_hw = uxTaskGetStackHighWaterMark(NULL);  /* bytes no ESP-IDF */
    ESP_LOGI(TAG, "[MEM] %-24s | SRAM heap total=%u used=%u peak=%u | "
                  "PSRAM heap total=%u used=%u peak=%u | main stack min free=%u B",
             stage,
             (unsigned)st, (unsigned)(st - sf), (unsigned)(st - sm),
             (unsigned)pt, (unsigned)(pt - pf), (unsigned)(pt - pm),
             (unsigned)stack_hw);
}

/* ─── Timing struct ──────────────────────────────────────────────────────── */

struct PipelineTiming {
    int64_t butterworth_us = 0;
    int64_t loess_us       = 0;
    int64_t nlm_us         = 0;
    int64_t autocorr_us    = 0;
    int64_t normalise_us   = 0;   /* tflite_acf_normalise (StandardScaler) */
    int64_t tflite_us      = 0;
    int64_t total_us       = 0;   /* butterworth+loess+nlm+autocorr+normalise+tflite */
};

/* ─── SD card ────────────────────────────────────────────────────────────── */

static esp_err_t sd_mount(sdmmc_card_t **out_card)
{
    esp_vfs_fat_sdmmc_mount_config_t mount_cfg = {
        .format_if_mount_failed   = false,
        .max_files                = 5,
        .allocation_unit_size     = 16 * 1024,
        .disk_status_check_enable = false,
        .use_one_fat              = false,
    };
    sdmmc_host_t host    = SDMMC_HOST_DEFAULT();
    host.max_freq_khz    = SDMMC_FREQ_DEFAULT;

    sdmmc_slot_config_t slot_cfg = SDMMC_SLOT_CONFIG_DEFAULT();
    slot_cfg.clk   = GPIO_NUM_7;
    slot_cfg.cmd   = GPIO_NUM_9;
    slot_cfg.d0    = GPIO_NUM_8;
    slot_cfg.width = 1;
    slot_cfg.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;

    ESP_LOGI(TAG, "Mounting SD card at %s …", SD_MOUNT);
    esp_err_t ret = esp_vfs_fat_sdmmc_mount(SD_MOUNT, &host, &slot_cfg,
                                             &mount_cfg, out_card);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "SD mount failed: %s", esp_err_to_name(ret));
        return ret;
    }
    sdmmc_card_print_info(stdout, *out_card);
    return ESP_OK;
}

static void sd_unmount(sdmmc_card_t *card)
{
    esp_vfs_fat_sdcard_unmount(SD_MOUNT, card);
    ESP_LOGI(TAG, "SD card unmounted.");
}

/* ─── File I/O ───────────────────────────────────────────────────────────── */

static size_t read_ecg_csv(const char *path, float *buf, size_t n)
{
    FILE *f = fopen(path, "r");
    if (!f) { ESP_LOGE(TAG, "Cannot open: %s", path); return 0; }

    char line[CSV_LINE_BUF];
    size_t count = 0;
    while (count < n && fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        char *end;
        float val = strtof(line, &end);
        if (end == line) continue;
        buf[count++] = val;
    }
    fclose(f);

    if (count < n) {
        ESP_LOGW(TAG, "%s: only %zu/%zu samples, zero-padding.", path, count, n);
        memset(buf + count, 0, (n - count) * sizeof(float));
    }
    return count;
}

/* ─── Pipeline execution ─────────────────────────────────────────────────── */

static PipelineTiming run_pipeline_timed(ecg_pipeline_t *pipe, float *signal)
{
    PipelineTiming t;
    int64_t t0, t1;

    /* Ablation SemLOESSNLM: apenas Butterworth bandpass.
     * LOESS e NLM foram removidos — o modelo foi treinado sem esses filtros. */
    t0 = esp_timer_get_time();
    butterworth_filter_apply(pipe->bw, signal);
    t1 = esp_timer_get_time();
    t.butterworth_us = t1 - t0;

    /* loess_us e nlm_us permanecem zero — registrados no CSV para consistência */
    t.total_us = t.butterworth_us;
    return t;
}

/* ─── Results CSV ────────────────────────────────────────────────────────── */

/**
 * @brief Append one result row to /sd/results.csv.
 *        Creates the file with header if it does not exist.
 */
static void append_result(const char *filename,
                          const tflite_result_t &r,
                          const PipelineTiming  &t)
{
    /* Check if file exists to decide whether to write header */
    bool needs_header = false;
    FILE *f_check = fopen(RESULTS_FILE, "r");
    if (!f_check) {
        needs_header = true;
    } else {
        fclose(f_check);
    }

    FILE *f = fopen(RESULTS_FILE, "a");
    if (!f) { ESP_LOGE(TAG, "Cannot open results file"); return; }

    if (needs_header) {
        fprintf(f, "filename,label,probability,int8_out,"
                   "butterworth_us,loess_us,nlm_us,"
                   "autocorr_us,normalise_us,tflite_us,total_us\n");
    }

    fprintf(f, "%s,%s,%.8f,%d,%" PRId64 ",%" PRId64 ",%" PRId64
               ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "\n",
            filename,
            r.is_afib ? "AFIB" : "SR",
            r.probability,
            (int)r.raw_output,
            t.butterworth_us,
            t.loess_us,
            t.nlm_us,
            t.autocorr_us,
            t.normalise_us,
            t.tflite_us,
            t.total_us);

    fclose(f);
}

/* ─── Helper: check if filename should be skipped ───────────────────────── */

static bool should_skip(const char *name)
{
    /* Skip non-CSV files */
    size_t len = strlen(name);
    if (len < 4) return true;
    if (strcasecmp(name + len - 4, ".csv") != 0) return true;

    /* Skip output files generated by this firmware */
    if (strcasecmp(name, "results.csv")    == 0) return true;
    if (strcasecmp(name, "timing_log.csv") == 0) return true;
    if (strcasecmp(name, "ecg_output.csv") == 0) return true;
    if (strcasecmp(name, "ecg_autocorr.csv") == 0) return true;
    if (strcasecmp(name, "ecg_input.csv")  == 0) return true;
    if (strcasecmp(name, "ecg_result.csv") == 0) return true;

    return false;
}

/* ─── app_main ───────────────────────────────────────────────────────────── */

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "=== ECG Batch Classification — XIAO ESP32S3 Sense ===");

    /* ── 1. NVS init ───────────────────────────────────────────────────── */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
        ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
    mem_report("boot (after NVS)");

    /* ── 2. Mount SD ───────────────────────────────────────────────────── */
    sdmmc_card_t *sd_card = nullptr;
    ESP_ERROR_CHECK(sd_mount(&sd_card));
    mem_report("after SD mount");

    /* ── 3. Allocate pipeline (reused across all files) ────────────────── */
    ecg_pipeline_t *pipeline = ecg_pipeline_alloc(ECG_SAMPLES);
    if (!pipeline) {
        ESP_LOGE(TAG, "Pipeline allocation failed — halting.");
        sd_unmount(sd_card);
        return;
    }

    /* ── 4. Allocate autocorrelation context (reused) ───────────────────── */
    autocorr_t *ac = autocorr_alloc(ECG_SAMPLES);
    if (!ac) {
        ESP_LOGE(TAG, "Autocorr allocation failed — halting.");
        ecg_pipeline_free(pipeline);
        sd_unmount(sd_card);
        return;
    }

    /* ── 5. Allocate TFLite context — ECGHybridV15 SemLOESSNLM ───────────── */
    tflite_inference_t *tfl = tflite_inference_alloc();
    if (!tfl) {
        ESP_LOGE(TAG, "TFLite allocation failed — halting.");
        autocorr_free(ac);
        ecg_pipeline_free(pipeline);
        sd_unmount(sd_card);
        return;
    }
    mem_report("after pipeline+ACF+TFLite alloc");
    ESP_LOGI(TAG, "[MEM] tensor arena used = %u B of %u B reserved",
             (unsigned)tflite_inference_arena_used(tfl), (unsigned)tflite_inference_arena_size());

    /* ── 6. Scan SD root directory ─────────────────────────────────────── */
    DIR *dir = opendir(SD_MOUNT);
    if (!dir) {
        ESP_LOGE(TAG, "Cannot open SD root directory — halting.");
        tflite_inference_free(tfl);
        autocorr_free(ac);
        ecg_pipeline_free(pipeline);
        sd_unmount(sd_card);
        return;
    }

    int file_count   = 0;
    int afib_count   = 0;
    int sr_count     = 0;
    int error_count  = 0;

    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {

        /* Skip directories and non-CSV files */
        if (entry->d_type == DT_DIR) continue;
        if (should_skip(entry->d_name))  continue;
        if (MEM_REPORT_MAX_FILES > 0 && file_count >= MEM_REPORT_MAX_FILES) break;

        file_count++;
        char full_path[280];
        snprintf(full_path, sizeof(full_path), "%s/%s", SD_MOUNT, entry->d_name);

        ESP_LOGI(TAG, "── [%d] Processing: %s", file_count, entry->d_name);

        /* ── Read ECG ─────────────────────────────────────────────────── */
        PipelineTiming timing;
        size_t samples = read_ecg_csv(full_path, ecg_buf, ECG_SAMPLES);
        if (samples == 0) {
            ESP_LOGE(TAG, "  Skipping — no samples read.");
            error_count++;
            continue;
        }

        /* ── Pipeline ────────────────────────────────────────────────── */
        {
            PipelineTiming pt = run_pipeline_timed(pipeline, ecg_buf);
            timing.butterworth_us = pt.butterworth_us;
            timing.loess_us       = pt.loess_us;
            timing.nlm_us         = pt.nlm_us;
            timing.total_us      += pt.butterworth_us + pt.loess_us + pt.nlm_us;
        }

        /* ── Autocorrelation ─────────────────────────────────────────── */
        int64_t t0 = esp_timer_get_time();
        autocorr_compute(ac, ecg_buf);
        timing.autocorr_us = esp_timer_get_time() - t0;
        timing.total_us   += timing.autocorr_us;

        /* ── Normalise ACF (StandardScaler embedded in tflite_inference.cc) ── */
        int64_t t_norm0 = esp_timer_get_time();
        tflite_acf_normalise(ac->acorr, acf_norm);
        timing.normalise_us  = esp_timer_get_time() - t_norm0;
        timing.total_us     += timing.normalise_us;

        /* ── TFLite inference ────────────────────────────────────────── */
        tflite_result_t result = {};
        t0 = esp_timer_get_time();
        bool ok = tflite_inference_run(tfl, ecg_buf, acf_norm, &result);
        timing.tflite_us  = esp_timer_get_time() - t0;
        timing.total_us  += timing.tflite_us;

        if (!ok) {
            ESP_LOGE(TAG, "  Inference failed — skipping.");
            error_count++;
            continue;
        }

        /* ── Log result ──────────────────────────────────────────────── */
        ESP_LOGI(TAG, "  → %s  prob=%.4f  total=%.2f s",
                 result.is_afib ? "AFIB" : "SR",
                 result.probability,
                 timing.total_us / 1e6f);

        if (result.is_afib) afib_count++; else sr_count++;
        if (file_count == 1) mem_report("after 1st inference");

        /* ── Append to results.csv ───────────────────────────────────── */
        append_result(entry->d_name, result, timing);
    }

    closedir(dir);
    mem_report("end of batch");

    /* ── 7. Summary ────────────────────────────────────────────────────── */
    ESP_LOGI(TAG, "══════════════════════════════════════");
    ESP_LOGI(TAG, "  Batch complete");
    ESP_LOGI(TAG, "  Files processed : %d", file_count);
    ESP_LOGI(TAG, "  AFIB            : %d", afib_count);
    ESP_LOGI(TAG, "  SR              : %d", sr_count);
    ESP_LOGI(TAG, "  Errors          : %d", error_count);
    ESP_LOGI(TAG, "  Results saved   : %s", RESULTS_FILE);
    ESP_LOGI(TAG, "══════════════════════════════════════");

    /* ── 8. Cleanup ────────────────────────────────────────────────────── */
    tflite_inference_free(tfl);
    autocorr_free(ac);
    ecg_pipeline_free(pipeline);
    sd_unmount(sd_card);
}
