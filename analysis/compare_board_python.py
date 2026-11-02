"""
Compara o results.csv gravado pela placa (firmware v3) com as predições LiteRT INT8 do Python
para os 4.025 registros de teste, e resume as latências por estágio.

Uso:
  cd /home/jose_alberto/Modelos_IEEE_Artigo_Revisao/
  python compare_board_python.py results.csv [--py holdout_probs_int8.csv --q 80]
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import argparse
ap = argparse.ArgumentParser()
ap.add_argument('results_csv')
ap.add_argument('--py', default='/home/jose_alberto/Modelos_IEEE_Artigo_Revisao/Ablation_LOESS_NLM_v3_ok/holdout_probs_int8.csv',
                help='holdout_probs_int8.csv do mesmo modelo (Python)')
ap.add_argument('--q', type=int, default=80, help='MODEL_THRESHOLD_Q do firmware (model_params.h)')
args = ap.parse_args()
PY, Q_THR = args.py, args.q
b = pd.read_csv(args.results_csv).drop_duplicates('filename', keep='last')
p = pd.read_csv(PY)
m = p.merge(b, on='filename', how='left', suffixes=('_py', '_board'))
missing = m['int8_out_board'].isna().sum()
print(f'Registros de teste: {len(p)} | presentes no results.csv: {len(p) - missing} | ausentes: {missing}')
extra = set(b.filename) - set(p.filename)
if extra:
    print(f'Arquivos no results.csv fora do conjunto de teste: {len(extra)} (ignorados)')
m = m.dropna(subset=['int8_out_board'])
qb, qp = m['int8_out_board'].astype(int).values, m['int8_out_py'].astype(int).values
y = m['label_int'].values.astype(int)
d = qb - qp
print('\n=== Saída INT8: placa × Python ===')
print(f'idênticas: {np.sum(d == 0)} de {len(d)} ({np.mean(d == 0) * 100:.2f}%) | '
      f'|diferença| máx = {np.abs(d).max()} LSB | distribuição: '
      + ', '.join(f'{k:+d}: {v}' for k, v in sorted(pd.Series(d).value_counts().items())))
dec_b, dec_p = qb >= Q_THR, qp >= Q_THR
print(f'decisões idênticas: {np.sum(dec_b == dec_p)} de {len(d)} '
      f'({int(np.sum(dec_b != dec_p))} discordantes)')

print(f'\n=== Métricas da placa (limiar inteiro {Q_THR}) ===')
pb = (qb + 128) / 256.0
tp = int(np.sum(dec_b & (y == 1))); fn = int(np.sum(~dec_b & (y == 1)))
tn = int(np.sum(~dec_b & (y == 0))); fp = int(np.sum(dec_b & (y == 0)))
print(f'AUC {roc_auc_score(y, pb):.4f}  Sens {tp / (tp + fn):.4f}  Spec {tn / (tn + fp):.4f}  '
      f'PPV {tp / (tp + fp):.4f}  F1 {2 * tp / (2 * tp + fp + fn):.4f}  TP {tp} FN {fn} FP {fp} TN {tn}')
yp = qp >= Q_THR
print(f'(Python LiteRT INT8: AUC {roc_auc_score(y, (qp + 128) / 256.0):.4f}  '
      f'TP {int(np.sum(yp & (y == 1)))} FN {int(np.sum(~yp & (y == 1)))} '
      f'FP {int(np.sum(yp & (y == 0)))} TN {int(np.sum(~yp & (y == 0)))})')

print('\n=== Latência por estágio na placa (ms, média ± DP; % do total) ===')
cols = ['butterworth_us', 'autocorr_us', 'normalise_us', 'tflite_us', 'total_us']
tot = m['total_us'].mean()
for c in cols:
    v = m[c].values / 1000.0
    print(f'{c:<16} {v.mean():9.2f} ± {v.std(ddof=1):6.2f}   [{v.min():.2f}, {v.max():.2f}]   '
          f'{m[c].mean() / tot * 100:5.1f}%')
soma = m[['butterworth_us', 'loess_us', 'nlm_us', 'autocorr_us', 'normalise_us', 'tflite_us']].sum(axis=1)
print(f'conferência: soma dos estágios = total em {np.mean(soma == m.total_us) * 100:.1f}% dos registros')
print('Obs.: total_us não inclui a leitura do arquivo no micro-SD nem a gravação do resultado.')
