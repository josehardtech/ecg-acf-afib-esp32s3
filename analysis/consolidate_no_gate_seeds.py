"""
Gate vs. concatenação (no_gate) com três sementes de treino (42, 43, 44).

Compara, no mesmo conjunto de teste (n = 4.025):
  (a) variação natural entre sementes da referência: ref_s43 - ref_s42, ref_s44 - ref_s42
  (b) efeito da troca do gate, semente a semente:     no_gate_sS - ref_sS
Usa ΔAUC pareado por bootstrap (2.000 reamostragens, semente 42), como na Tabela 7.

Uso:
  cd /home/jose_alberto/Modelos_IEEE_Artigo_Revisao/Sensibilidade_OAT/
  python consolidate_no_gate_seeds.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = '/home/jose_alberto/Modelos_IEEE_Artigo_Revisao/Sensibilidade_OAT'
REF_DIR_V3 = '/home/jose_alberto/Modelos_IEEE_Artigo_Revisao/Ablation_LOESS_NLM_v3_ok'
N_BOOT, BOOT_SEED = 2000, 42

RUNS = {
    'ref_s42':     [f'{ROOT}/reference_seed42', REF_DIR_V3],
    'ref_s43':     [f'{ROOT}/reference_seed43'],
    'ref_s44':     [f'{ROOT}/reference_seed44'],
    'no_gate_s42': [f'{ROOT}/arch_no_gate'],
    'no_gate_s43': [f'{ROOT}/arch_no_gate_seed43'],
    'no_gate_s44': [f'{ROOT}/arch_no_gate_seed44'],
}


def load(name):
    for d in RUNS[name]:
        p = os.path.join(d, 'holdout_probs_int8.csv')
        if os.path.isfile(p):
            df = pd.read_csv(p)
            m = os.path.join(d, 'metrics_final.json')
            meta = json.load(open(m)) if os.path.isfile(m) else {}
            return df, d, meta
    sys.exit(f'ERRO: holdout_probs_int8.csv não encontrado para {name}: {RUNS[name]}')


data = {}
for k in RUNS:
    df, d, meta = load(k)
    data[k] = (df.set_index('filename').sort_index(), d, meta)
    print(f'{k:<12} n={len(df):>5}  pasta={d}')

# Alinhamento registro a registro (mesmo conjunto de teste em todas as execuções)
base_idx = data['ref_s42'][0].index
for k, (df, _, _) in data.items():
    assert len(df) == 4025, f'{k}: n={len(df)} (esperado 4.025)'
    assert df.index.equals(base_idx), f'{k}: registros diferentes da referência'
    assert (df['label_int'].values == data['ref_s42'][0]['label_int'].values).all(), f'{k}: rótulos diferem'
y = data['ref_s42'][0]['label_int'].values
P = {k: v[0]['p_int8'].values for k, v in data.items()}

# Mesmos índices de reamostragem para todas as comparações
rng = np.random.RandomState(BOOT_SEED)
boot_idx = []
while len(boot_idx) < N_BOOT:
    i = rng.randint(0, len(y), len(y))
    if 0 < y[i].sum() < len(i):
        boot_idx.append(i)


def auc(p):
    return roc_auc_score(y, p)


def paired_delta(pa, pb):
    """ΔAUC = AUC(pa) - AUC(pb), IC 95% percentil por bootstrap pareado."""
    d = np.array([roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]) for i in boot_idx])
    return auc(pa) - auc(pb), np.percentile(d, 2.5), np.percentile(d, 97.5)


def thr_metrics(k):
    df = data[k][0]
    pred = df['pred_int8'].values
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    return tp / (tp + fn), tn / (tn + fp), 2 * tp / (2 * tp + fp + fn), fn, fp


print('\n=== AUC e métricas no limiar INT8 de cada execução ===')
print(f"{'execução':<12} {'AUC':>7} {'Sens':>7} {'Spec':>7} {'F1':>7} {'FN':>4} {'FP':>4}")
rows = []
for k in RUNS:
    s, sp, f1, fn, fp = thr_metrics(k)
    rows.append(dict(run=k, auc=auc(P[k]), sens=s, spec=sp, f1=f1, fn=fn, fp=fp))
    print(f'{k:<12} {auc(P[k]):7.4f} {s:7.4f} {sp:7.4f} {f1:7.4f} {fn:4d} {fp:4d}')

ref_aucs = np.array([auc(P[f'ref_s{s}']) for s in (42, 43, 44)])
ng_aucs = np.array([auc(P[f'no_gate_s{s}']) for s in (42, 43, 44)])
print(f'\nReferência (gate):  AUC = {ref_aucs.mean():.4f} ± {ref_aucs.std(ddof=1):.4f}  '
      f'[{ref_aucs.min():.4f}, {ref_aucs.max():.4f}]')
print(f'no_gate (concat.):  AUC = {ng_aucs.mean():.4f} ± {ng_aucs.std(ddof=1):.4f}  '
      f'[{ng_aucs.min():.4f}, {ng_aucs.max():.4f}]')

print('\n=== (a) Variação natural entre sementes da referência ===')
comp = []
for s in (43, 44):
    d, lo, hi = paired_delta(P[f'ref_s{s}'], P['ref_s42'])
    comp.append(dict(comparison=f'ref_s{s} - ref_s42', delta=d, lo=lo, hi=hi))
    print(f'ref_s{s} - ref_s42      ΔAUC = {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]')

print('\n=== (b) Troca do gate pela concatenação, semente a semente ===')
deltas = []
for s in (42, 43, 44):
    d, lo, hi = paired_delta(P[f'no_gate_s{s}'], P[f'ref_s{s}'])
    deltas.append(d)
    comp.append(dict(comparison=f'no_gate_s{s} - ref_s{s}', delta=d, lo=lo, hi=hi))
    flag = 'IC exclui 0' if (lo > 0 or hi < 0) else 'IC inclui 0'
    print(f'no_gate_s{s} - ref_s{s}  ΔAUC = {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]  ({flag})')

# ΔAUC médio entre sementes, com IC pareado (mesmos índices em todas as sementes)
dm = np.array([np.mean([roc_auc_score(y[i], P[f'no_gate_s{s}'][i]) -
                        roc_auc_score(y[i], P[f'ref_s{s}'][i]) for s in (42, 43, 44)])
               for i in boot_idx])
dmean = float(np.mean(deltas))
lo, hi = np.percentile(dm, 2.5), np.percentile(dm, 97.5)
comp.append(dict(comparison='média (no_gate - ref), 3 sementes', delta=dmean, lo=lo, hi=hi))
print(f'\nΔAUC médio (3 sementes) = {dmean:+.4f}  [{lo:+.4f}, {hi:+.4f}]')

# Leitura para a decisão de redação
pos = sum(d > 0 for d in deltas)
max_seed_noise = max(abs(c['delta']) for c in comp[:2])
print('\n=== Leitura ===')
print(f'no_gate acima da referência em {pos}/3 sementes; '
      f'maior |Δ| entre sementes da referência = {max_seed_noise:.4f}')
if pos == 3 and lo > 0:
    print('-> Cenário 2: a concatenação fica consistentemente acima (IC do Δ médio exclui 0).')
else:
    print('-> Cenário 1: a vantagem da concatenação não se sustenta entre sementes.')

pd.DataFrame(rows).to_csv(f'{ROOT}/no_gate_seeds_runs.csv', index=False)
pd.DataFrame(comp).to_csv(f'{ROOT}/no_gate_seeds_deltas.csv', index=False)
print(f'\nSalvo: {ROOT}/no_gate_seeds_runs.csv | {ROOT}/no_gate_seeds_deltas.csv')
