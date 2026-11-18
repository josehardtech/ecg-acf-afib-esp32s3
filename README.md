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

## Version cited in the article

The article (IEEE Access, Access-2026-50898) cites commit `3a0eb24` of this repository.
To obtain exactly that version:

```bash
git clone https://github.com/josehardtech/ecg-acf-afib-esp32s3
cd ecg-acf-afib-esp32s3
git checkout 3a0eb24
```

## Software environment

| Component | Pinned version | File |
|---|---|---|
| Python and packages (training, conversion, analysis) | see file | `environment/requirements.txt`, `environment/environment.yml` |
| ESP-IDF | v5.5.3 | `firmware/*/dependencies.lock` |
| `espressif/esp-tflite-micro` | 1.3.5 | `firmware/*/dependencies.lock` |
| `espressif/esp-nn` | 1.1.2 | `firmware/*/dependencies.lock` |
| `espressif/esp-dsp` | 1.7.1 | `firmware/*/dependencies.lock` |

```bash
conda env create -f environment/environment.yml      # or: pip install -r environment/requirements.txt
conda activate ecg-acf-afib
```

The environment files contain only cross-platform package versions (no build strings).
Training was run on Linux with an NVIDIA GPU; deterministic settings are fixed in the notebooks.

## Paths

The scripts were written for the authors' folders. They read two environment variables, whose
defaults are the authors' paths:

| Variable | Content |
|---|---|
| `ECG_DATA` | Lead I CSV folders (`Chinese_Lead_I_AFIB`, `Chinese_Lead_I_SR`, `PTB_XL_Lead_I_AFIB`, `PTB_XL_Lead_I_SR`) and `PTB_XL_Raw/ptbxl_database.csv` |
| `ECG_WORK` | Working folder where the notebooks write models and results |

```bash
export ECG_DATA=/path/to/Datasets
export ECG_WORK=/path/to/work
```

Scripts that only read released result files also accept the file as an argument and run directly on
this repository, for example:

```bash
python analysis/make_fig4.py models/deployed_no_loess_nlm/holdout_probs_int8.csv .
python analysis/compare_board_python.py analysis/results/results_board_v3.csv --py models/deployed_no_loess_nlm/holdout_probs_int8.csv
```

In the notebooks, the paths are set in the first code cell:

| Notebooks | Variables to edit |
|---|---|
| `main`, `ablations`, `cross_domain` | `BASE` (output folder), `PTBXL_DB_CSV`, and the Lead I folders |
| `sensitivity`, `architectures`, `baselines` | `OAT_ROOT` (output folder), `REF_DIR` (reference model), `PTBXL_DB_CSV` |
| `external_validation` | `BASE_MODEL`, `GEORGIA_DIR`, `OUT_DIR`, `TRAIN_DIRS` |

The shell scripts in `notebooks/scripts/` use `ECG_WORK` as their root folder.

## Reproducing the results

1. Extract Lead I CSVs (see *Data*) and set the paths (see *Paths*).
2. Run `notebooks/main` (PyTorch, deterministic training with `SEED = 42`). The partitions are regenerated
   with `StratifiedGroupKFold(random_state = 42)` and checked against `splits/`; the INT8 model must reproduce
   SHA-256 `a7582a6d1dd772754886b242c2c23a9c1fd36a7dbf7d261e4761e14b5e086b2e`.
3. Run the scripts in `analysis/` for the patient-clustered confidence intervals, paired comparisons and
   threshold-transfer analysis.

Board and host decisions: on the 4,025 test recordings the firmware reproduces the confusion matrix of the
LiteRT INT8 model and 4,023 of the 4,025 decisions; two SR recordings (`JS20623.csv`, `JS37008.csv`) swap classes.
`analysis/compare_board_python.py` reproduces this comparison from the released files.

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

ESP-IDF v5.5.3, target `esp32s3`; component versions are pinned in `dependencies.lock` (see *Software environment*).

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
