import os, shutil, sys
import pandas as pd
# Paths: set ECG_WORK (working/results folder) and ECG_DATA (Lead I CSVs and PTB-XL metadata),
# or edit the defaults below. See README, section 'Paths'.
WORK = os.environ.get('ECG_WORK', '/home/jose_alberto/Modelos_IEEE_Artigo_Revisao')
DATA = os.environ.get('ECG_DATA', '/mnt/windows/Doutorado/Datasets')

SD = sys.argv[1]                                    # ex.: /media/jose_alberto/ECG_SD
DIRS = [f'{DATA}/Chinese_Lead_I_AFIB',
        f'{DATA}/Chinese_Lead_I_SR',
        f'{DATA}/PTB_XL_Lead_I_AFIB',
        f'{DATA}/PTB_XL_Lead_I_SR']
t = pd.read_csv('Ablation_LOESS_NLM_v3_ok/holdout_probs_int8.csv')
idx = {}
for d in DIRS:
    for f in os.listdir(d):
        if f.lower().endswith('.csv'):
            assert f not in idx, f'nome repetido entre pastas: {f}'
            idx[f] = os.path.join(d, f)
falta = [f for f in t.filename if f not in idx]
assert not falta, f'{len(falta)} arquivos de teste não encontrados, ex.: {falta[:5]}'
for k, f in enumerate(t.filename, 1):
    shutil.copy2(idx[f], os.path.join(SD, f))
    if k % 500 == 0: print(k, 'copiados')
n = len([f for f in os.listdir(SD) if f.lower().endswith('.csv')])
print(f'Concluído: {len(t)} copiados | CSV no cartão: {n}')
