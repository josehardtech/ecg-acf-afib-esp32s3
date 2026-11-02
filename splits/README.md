# Patient-exclusive partitions

| File | Records | Use |
|---|---|---|
| `train.csv` | 22,722 | Training; `cv_fold` gives the fold (1–10) of the patient-grouped internal cross-validation |
| `calibration.csv` | 1,420 | Representative dataset for full-integer (INT8) post-training quantization only |
| `test.csv` | 4,025 | Patient-exclusive test set (3,720 patients) |
| `georgia.csv` | 2,069 | External validation (Georgia 12-lead ECG database, PhysioNet/CinC 2020) |

Columns: `filename` (Lead I CSV extracted from the source record), `dataset`, `label`,
`patient_group` (`CSN_<file>`: one patient per record; `PTB_<patient_id>` from `ptbxl_database.csv`).
No patient group appears in more than one of train, calibration and test.
PTB-XL file `NNNNN_hr.csv` corresponds to `ecg_id` NNNNN; CSN file `JSxxxxx.csv` to record JSxxxxx.
Partitions: StratifiedGroupKFold (shuffle, random_state = 42).
