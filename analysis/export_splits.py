"""
Exporta as partições exclusivas por paciente do modelo implantado (v3) para CSVs públicos.

Reconstrói a ordem interna dos 28.167 registros exatamente como os notebooks
(load_domain: CSN AFIB, CSN SR, PTB-XL AFIB, PTB-XL SR; arquivos em ordem alfabética),
confere a reconstrução contra os grupos de paciente gravados no .pkl e contra o
conjunto de teste (holdout_probs_int8.csv), e grava:

  splits/train.csv        registros de treino, com o fold da validação cruzada interna (1–10)
  splits/calibration.csv  registros usados só na quantização INT8
  splits/test.csv         registros de teste (n = 4.025)
  splits/georgia.csv      registros do Georgia usados na validação externa (n = 2.069)
  splits/README.md        descrição das colunas

Colunas: filename, dataset (CSN | PTB-XL | Georgia), label (AFIB | SR), patient_group
(CSN_<arquivo> | PTB_<patient_id>; no Georgia não há identificador de paciente) e, no treino, cv_fold.

Uso:
  python export_splits.py ~/ecg-acf-afib-esp32s3/splits
"""
import glob
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

OUT = sys.argv[1] if len(sys.argv) > 1 else 'splits'
R = '/home/jose_alberto/Modelos_IEEE_Artigo_Revisao'
PKL = f'{R}/Ablation_LOESS_NLM_v3_ok/ablation_loess_nlm_calib_indices.pkl'
TEST_CSV = f'{R}/Ablation_LOESS_NLM_v3_ok/holdout_probs_int8.csv'
GEORGIA_CSV = f'{R}/Georgia/eval_ablation_loess_nlm_v3/georgia_probs_int8.csv'
PTBXL_DB = '/mnt/windows/Doutorado/Datasets/PTB_XL_Raw/ptbxl_database.csv'
DS = '/mnt/windows/Doutorado/Datasets'
ORDER = [('CSN', 'AFIB', f'{DS}/Chinese_Lead_I_AFIB'), ('CSN', 'SR', f'{DS}/Chinese_Lead_I_SR'),
         ('PTB-XL', 'AFIB', f'{DS}/PTB_XL_Lead_I_AFIB'), ('PTB-XL', 'SR', f'{DS}/PTB_XL_Lead_I_SR')]
N_FOLDS, SPLIT_SEED = 10, 42
PTB_RE = re.compile(r'^(\d+)_(hr|lr)\.csv$')

os.makedirs(OUT, exist_ok=True)
d = pickle.load(open(PKL, 'rb'))
rows = []
for ds, lab, path in ORDER:
    for f in sorted(glob.glob(os.path.join(path, '*.csv'))):
        rows.append((os.path.basename(f), ds, lab))
pool = pd.DataFrame(rows, columns=['filename', 'dataset', 'label'])
assert len(pool) == len(d['groups_all']) == 28167, f'pool com {len(pool)} registros (esperado 28.167)'

db = pd.read_csv(PTBXL_DB, usecols=['ecg_id', 'patient_id'])
e2p = dict(zip(db.ecg_id.astype(int), db.patient_id.astype(float).astype(int)))
def grp(fn, ds):
    if ds == 'CSN':
        return f'CSN_{fn}'
    return f'PTB_{e2p[int(PTB_RE.match(fn).group(1))]}'
pool['patient_group'] = [grp(f, s) for f, s in zip(pool.filename, pool.dataset)]
assert (pool.patient_group.values == np.asarray(d['groups_all'], dtype=str)).all(), \
    'a ordem reconstruída não coincide com os grupos gravados no .pkl'

tr, ca, te = (np.asarray(d[k]) for k in ('idx_train', 'idx_calib', 'idx_ho'))
assert len(set(tr) | set(ca) | set(te)) == len(tr) + len(ca) + len(te) == 28167
g = pool.patient_group.values
assert not (set(g[tr]) & set(g[te])) and not (set(g[tr]) & set(g[ca])) and not (set(g[ca]) & set(g[te])), \
    'paciente compartilhado entre partições'
test_ref = pd.read_csv(TEST_CSV)
assert list(pool.filename.values[te]) == list(test_ref.filename), 'teste reconstruído difere do holdout_probs_int8.csv'

train = pool.iloc[tr].reset_index(drop=True)
y_tr = (train.label == 'AFIB').astype(int).values
fold = np.zeros(len(train), dtype=int)
kf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SPLIT_SEED)
for k, (_, va) in enumerate(kf.split(np.zeros(len(train)), y_tr, groups=train.patient_group.values), 1):
    fold[va] = k
train['cv_fold'] = fold

train.to_csv(f'{OUT}/train.csv', index=False)
pool.iloc[ca].to_csv(f'{OUT}/calibration.csv', index=False)
pool.iloc[te].to_csv(f'{OUT}/test.csv', index=False)
geo = pd.read_csv(GEORGIA_CSV)
pd.DataFrame({'filename': geo.filename, 'dataset': 'Georgia',
              'label': np.where(geo.label_int == 1, 'AFIB', 'SR'),
              'patient_group': 'not available'}).to_csv(f'{OUT}/georgia.csv', index=False)

with open(f'{OUT}/README.md', 'w') as f:
    f.write('''# Patient-exclusive partitions

| File | Records | Use |
|---|---|---|
| `train.csv` | {ntr:,} | Training; `cv_fold` gives the fold (1–10) of the patient-grouped internal cross-validation |
| `calibration.csv` | {nca:,} | Representative dataset for full-integer (INT8) post-training quantization only |
| `test.csv` | {nte:,} | Patient-exclusive test set ({npt:,} patients) |
| `georgia.csv` | {ngeo:,} | External validation (Georgia 12-lead ECG database, PhysioNet/CinC 2020) |

Columns: `filename` (Lead I CSV extracted from the source record), `dataset`, `label`,
`patient_group` (`CSN_<file>`: one patient per record; `PTB_<patient_id>` from `ptbxl_database.csv`).
No patient group appears in more than one of train, calibration and test.
PTB-XL file `NNNNN_hr.csv` corresponds to `ecg_id` NNNNN; CSN file `JSxxxxx.csv` to record JSxxxxx.
Partitions: StratifiedGroupKFold (shuffle, random_state = 42).
'''.format(ntr=len(tr), nca=len(ca), nte=len(te), npt=len(set(g[te])), ngeo=len(geo)))
print(f'OK: treino {len(tr)} | calibração {len(ca)} | teste {len(te)} ({len(set(g[te]))} pacientes) | '
      f'Georgia {len(geo)} -> {OUT}/')
print('folds de CV no treino:', np.bincount(fold)[1:].tolist())
