# ECG autocorrelation + dual-branch CNN for AFIB/SR screening on the ESP32-S3

Code, INT8 models, patient-exclusive partitions and firmware accompanying the manuscript

> J. A. Ferreira Filho *et al.*, "ECG Characterization via Autocorrelation and Embedded Machine Learning
> for Classification of Sinus Rhythm and Atrial Fibrillation," submitted to *IEEE Access*.

A dual-branch one-dimensional CNN combines the Lead I waveform (10 s, 500 Hz) with its discrete
autocorrelation function (ACF, 5,000 lags) to discriminate atrial fibrillation (AFIB) from sinus rhythm (SR)
without fiducial points. The model is quantized to INT8 and executed with LiteRT for Microcontrollers
(ESP-NN kernels) on a Seeed Studio XIAO ESP32-S3 Sense.

## Contents

| Folder | Content |
|---|---|
| `notebooks/main` | Training, threshold selection, INT8 conversion and evaluation of the deployed model (Butterworth only, no LOESS/NLM) |
| `notebooks/ablations` | Full pipeline and ablations #2 (ACF branch only) and #3 (no filtering) |
| `notebooks/sensitivity`, `notebooks/architectures`, `notebooks/baselines` | One-at-a-time hyperparameter sensitivity, alternative architectures and fusion strategies, MobileNet-1D, CNN–LSTM and R-R-feature baselines |
| `notebooks/cross_domain`, `notebooks/external_validation` | Cross-dataset experiments (CSN ↔ PTB-XL) and Georgia external validation |
| `models/` | Float32 PyTorch checkpoints (`*_final.pt`), INT8 `.tflite` models, ACF scaler, INT8 thresholds, per-record test outputs and metrics |
| `firmware/` | ESP-IDF projects for the deployed model and the MobileNet-1D baseline (batch classification from micro-SD, per-stage timing, memory report) |
| `splits/` | Patient-exclusive train / calibration / test partitions, internal CV folds and the Georgia record list |
| `analysis/` | Patient-clustered bootstrap statistics, threshold-transfer analysis, board-versus-Python comparison, figure scripts and results |

## Data

The ECG signals are **not** redistributed. Download them from PhysioNet under their respective licenses:

- PTB-XL v1.0.3 — https://physionet.org/content/ptb-xl/1.0.3/
- Chapman-Shaoxing-Ningbo (CSN) and Georgia 12-lead ECG — PhysioNet/CinC Challenge 2020/2021 — https://physionet.org/content/challenge-2021/

Lead I is extracted to one CSV per record (5,000 samples, millivolts, no resampling or per-record
normalization). The file names in `splits/` identify each record (`NNNNN_hr.csv` = PTB-XL `ecg_id`;
`JSxxxxx.csv` = CSN record; `Exxxxx_*.csv` = Georgia record).

## Reproducing the results

1. Extract Lead I CSVs and place them in the folders expected by the notebooks
   (`PATHS_CH`, `PATHS_PTB`; adjust the paths at the top of each notebook).
2. Run `notebooks/main` (PyTorch, deterministic training with `SEED = 42`). The partitions are regenerated
   with `StratifiedGroupKFold(random_state = 42)` and checked against `splits/`; the INT8 model must reproduce
   SHA-256 `a7582a6d1dd772754886b242c2c23a9c1fd36a7dbf7d261e4761e14b5e086b2e`.
3. Run the scripts in `analysis/` for the patient-clustered confidence intervals, paired comparisons and
   threshold-transfer analysis.

## Original model and revision analyses

All results in the manuscript refer to the deployed model in `models/deployed_no_loess_nlm`
(No-LOESS/NLM configuration, seed 42, ACF scaler fitted on the whole training partition, SHA-256 above).
The analyses added during the peer review (Access-2026-50898) do **not** replace this model; they are stored separately:

| Analysis (manuscript location) | Script | Result files in `analysis/results/` |
|---|---|---|
| Patient-clustered bootstrap CIs and paired ΔAUC / McNemar (Sec. III, Tables 5–7) | `analysis/paired_stats_patient.py`, `analysis/baselines_patient_ci.py` | `paired_stats_patient/` |
| ACF scaler refitted within each training fold (Sec. III-E) | `notebooks/architectures/ARCH_variants_v4.ipynb`, `notebooks/scripts/run_v4.sh`, `analysis/consolidate_v4.py` | `v4_fold_scaler_comparison.csv`, `v4_scaler_fold_diag.csv`; model in `models/fold_scaler_seed42` |
| Float32 versus INT8 (Sec. III) | same as above | `v4_float32_vs_int8.csv`, `v4_float32_vs_int8_table5.csv` |
| Concatenation without gate, seeds 42–44 (Sec. III-F, Table 7) | `notebooks/scripts/run_no_gate_seeds.sh`, `analysis/consolidate_no_gate_seeds.py` | `no_gate_seeds_runs.csv`, `no_gate_seeds_deltas.csv`, `arch_summary.csv` |
| Hyperparameter sensitivity and seed variation (Sec. III-E, Table 6) | `notebooks/sensitivity/OAT_sensitivity_v3.ipynb`, `notebooks/scripts/run_oat.sh` | `oat_summary.csv` |
| Georgia threshold transfer and local recalibration (Sec. III-H) | `analysis/georgia_threshold_transfer.py` | `georgia_probs_int8.csv`, `threshold_transfer/` |
| Board versus Python agreement, MobileNet-1D on the board (Sec. III, III-F) | `analysis/compare_board_python.py` | `results_board_v3.csv`, `results_board_mobilenet.csv`, `monitor_*.log` |
| Fig. 4 (ROC of the deployed model) | `analysis/make_fig4.py` | — |

## Firmware

ESP-IDF v5.5, target `esp32s3`, components `espressif/esp-tflite-micro` and `espressif/esp-dsp`.

```bash
cd firmware/deployed_no_loess_nlm
idf.py set-target esp32s3
idf.py build flash monitor
```

The micro-SD card (FAT32) holds the test CSVs in its root; the firmware writes `results.csv` with the
INT8 output, the decision and the per-stage execution time of every window. Model files for any INT8
model of the notebooks are generated with `analysis/export_firmware.py`.

## License

Code: MIT (see `LICENSE`). The datasets keep their original licenses.
