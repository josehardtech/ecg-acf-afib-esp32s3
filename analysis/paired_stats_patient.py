"""
Estatística com reamostragem por PACIENTE (cluster bootstrap) — resposta ao Revisor 1, item 2.

Para cada configuração já treinada, recalcula a partir dos holdout_probs_int8.csv:
  1. Métricas com IC 95% por bootstrap de pacientes (e, para comparação, por registros).
  2. Diferenças pareadas (ΔAUC, ΔSens, ΔSpec, ΔPPV, ΔF1) com IC 95% por bootstrap de pacientes,
     reamostrando os MESMOS pacientes para as duas configurações comparadas.
  3. McNemar exato (decisões binárias) e contagens FN/FP.
  4. Contagem de pacientes e registros no conjunto de teste (pedido do item 4).

Agrupamento:
  PTB-XL: '00017_hr.csv' -> ecg_id 17 -> patient_id (ptbxl_database.csv)
  CSN e Georgia: cada registro é um paciente distinto.

Uso:
  cd /home/jose_alberto/Modelos_IEEE_Artigo_Revisao/
  conda activate pytorch_env
  python paired_stats_patient.py                    # tudo
  python paired_stats_patient.py --georgia PASTA    # inclui o Georgia (pasta com holdout_probs_int8.csv)

Saídas (em ./paired_stats_patient/):
  metrics_ci.csv     métricas por configuração, IC por paciente e por registro
  paired_deltas.csv  diferenças pareadas com IC por paciente + McNemar
  test_patients.csv  registros e pacientes por base em cada conjunto de teste
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata
from sklearn.metrics import roc_auc_score

ROOT = '/home/jose_alberto/Modelos_IEEE_Artigo_Revisao'
OAT = f'{ROOT}/Sensibilidade_OAT'
PTBXL_DB = '/mnt/windows/Doutorado/Datasets/PTB_XL_Raw/ptbxl_database.csv'
N_BOOT, SEED = 2000, 42

# ── Configurações (nome -> pasta) ────────────────────────────────────────────
TABLE5 = {
    'Full':     f'{ROOT}/Dual_cnn_branch_v3_ok',
    'Abl1':     f'{ROOT}/Ablation_LOESS_NLM_v3_ok',
    'Abl2':     f'{ROOT}/Ablation_CNN_LOESS_NLM_v3_ok',
    'Abl3':     f'{ROOT}/Ablation_All_Filters_RAW_v3_ok',
}
OAT_RUNS = ['reference_seed42', 'reference_seed43', 'reference_seed44',
            'gamma_1.0', 'gamma_3.0', 'alpha_0.5', 'alpha_0.9',
            'dropout_0.1-0.1', 'dropout_0.5-0.4', 'lr_2e-3', 'lr_8e-3', 'wd_1e-5', 'wd_1e-3']
ARCH_RUNS = ['arch_cnn_only', 'arch_no_gate', 'arch_no_gate_seed43', 'arch_no_gate_seed44',
             'arch_early_fusion', 'arch_acf_trunc']
CROSS = {
    'CSN->CSN': f'{ROOT}/Ablation_LOESS_NLM_csn_csn_v3_ok',
    'CSN->PTB': f'{ROOT}/Ablation_LOESS_NLM_csn_ptb_v3',
    'PTB->CSN': f'{ROOT}/Ablation_LOESS_NLM_ptb_csn_v3',
    'PTB->PTB': f'{ROOT}/Ablation_LOESS_NLM_ptb_ptb_v3',
}

# ── Comparações pareadas (A - B) ─────────────────────────────────────────────
PAIRS = [
    ('Table5', 'Abl1', 'Full'),
    ('Table5', 'Abl2', 'Abl1'),
    ('Table5', 'Abl3', 'Abl1'),
    ('Seeds', 'reference_seed43', 'reference_seed42'),
    ('Seeds', 'reference_seed44', 'reference_seed42'),
] + [('Table6', r, 'reference_seed42') for r in OAT_RUNS[3:]] \
  + [('Table7', 'arch_cnn_only', 'reference_seed42'),
     ('Table7', 'arch_no_gate', 'reference_seed42'),
     ('Table7', 'arch_early_fusion', 'reference_seed42'),
     ('Table7', 'arch_acf_trunc', 'reference_seed42'),
     ('Gate_seed', 'arch_no_gate_seed43', 'reference_seed43'),
     ('Gate_seed', 'arch_no_gate_seed44', 'reference_seed44')]


# ── Agrupamento por paciente ─────────────────────────────────────────────────
PTB_RE = re.compile(r'^(\d+)_(hr|lr)\.csv$')


def load_ptbxl_map(path):
    if not os.path.isfile(path):
        sys.exit(f'ERRO: ptbxl_database.csv não encontrado em {path}')
    db = pd.read_csv(path, usecols=['ecg_id', 'patient_id'])
    return dict(zip(db['ecg_id'].astype(int), db['patient_id'].astype(float).astype(int)))


def group_of(fname, ptb_map):
    m = PTB_RE.match(fname)
    if m:
        eid = int(m.group(1))
        if eid not in ptb_map:
            sys.exit(f'ERRO: ecg_id {eid} ({fname}) ausente do ptbxl_database.csv')
        return 'PTB', f'PTB_{ptb_map[eid]}'
    if fname.startswith('JS'):
        return 'CSN', f'CSN_{fname}'
    return 'OTHER', f'REC_{fname}'      # Georgia (E*.csv, um registro por paciente) e outros


def read_run(folder):
    p = os.path.join(folder, 'holdout_probs_int8.csv')
    if not os.path.isfile(p):
        return None
    df = pd.read_csv(p)
    df['filename'] = df['filename'].astype(str)
    return df.sort_values('filename').reset_index(drop=True)


# ── Métricas ─────────────────────────────────────────────────────────────────
def thr_metrics(y, pred):
    tp = np.sum((pred == 1) & (y == 1)); fn = np.sum((pred == 0) & (y == 1))
    tn = np.sum((pred == 0) & (y == 0)); fp = np.sum((pred == 1) & (y == 0))
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    acc = (tp + tn) / len(y)
    return dict(Sens=sens, Spec=spec, PPV=ppv, NPV=npv, F1=f1, Acc=acc, FN=int(fn), FP=int(fp))


def fast_auc(y, p):
    """AUC por postos (Mann-Whitney), idêntica a roc_auc_score inclusive com empates."""
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def all_metrics(y, p, pred):
    m = thr_metrics(y, pred)
    m['AUC'] = fast_auc(y, p)
    return m


KEYS = ['AUC', 'Sens', 'Spec', 'PPV', 'NPV', 'F1', 'Acc']


class Resampler:
    """Índices de bootstrap por paciente (cluster) e por registro, fixos para todo o conjunto de teste."""

    def __init__(self, groups, y, n_boot=N_BOOT, seed=SEED):
        codes, uniq = pd.factorize(groups)
        self.members = [np.flatnonzero(codes == g) for g in range(len(uniq))]
        self.n_pat = len(uniq)
        rng = np.random.RandomState(seed)
        self.clu, self.rec = [], []
        while len(self.clu) < n_boot:
            idx = np.concatenate([self.members[g] for g in rng.randint(0, self.n_pat, self.n_pat)])
            if 0 < y[idx].sum() < len(idx):
                self.clu.append(idx)
        rng = np.random.RandomState(seed)
        while len(self.rec) < n_boot:
            idx = rng.randint(0, len(y), len(y))
            if 0 < y[idx].sum() < len(idx):
                self.rec.append(idx)


def ci(vals):
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def metrics_with_ci(y, p, pred, rs):
    point = all_metrics(y, p, pred)
    out = {}
    for mode, idxs in (('pat', rs.clu), ('rec', rs.rec)):
        boots = [all_metrics(y[i], p[i], pred[i]) for i in idxs]
        for k in KEYS:
            lo, hi = ci([b[k] for b in boots])
            out[f'{k}_lo_{mode}'], out[f'{k}_hi_{mode}'] = lo, hi
    return point, out


def mcnemar_exact(y, pa, pb):
    ca, cb = (pa == y), (pb == y)
    b = int(np.sum(ca & ~cb)); c = int(np.sum(~ca & cb))
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p


def paired(y, A, B, rs):
    """A e B: (p, pred). Retorna Δ = A - B com IC por paciente."""
    ma, mb = all_metrics(y, *A), all_metrics(y, *B)
    res = {}
    dkeys = ['AUC', 'Sens', 'Spec', 'PPV', 'F1']
    boots = {k: [] for k in dkeys}
    for i in rs.clu:
        a, b = all_metrics(y[i], A[0][i], A[1][i]), all_metrics(y[i], B[0][i], B[1][i])
        for k in dkeys:
            boots[k].append(a[k] - b[k])
    for k in dkeys:
        lo, hi = ci(boots[k])
        res[f'd{k}'], res[f'd{k}_lo'], res[f'd{k}_hi'] = ma[k] - mb[k], lo, hi
    res['FN_A'], res['FN_B'], res['FP_A'], res['FP_B'] = ma['FN'], mb['FN'], ma['FP'], mb['FP']
    b_, c_, p_ = mcnemar_exact(y, A[1], B[1])
    res['mcnemar_A_only_correct'], res['mcnemar_B_only_correct'], res['mcnemar_p'] = b_, c_, p_
    return res


def fmt(x, nd=4):
    return f'{x:+.{nd}f}' if np.isfinite(x) else 'nan'


# ── Principal ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--georgia', default=None, help='pasta com holdout_probs_int8.csv do Georgia')
    ap.add_argument('--ptbxl', default=PTBXL_DB)
    ap.add_argument('--out', default=os.path.join(os.getcwd(), 'paired_stats_patient'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ptb_map = load_ptbxl_map(args.ptbxl)

    # Conjunto de teste interno compartilhado (Tabelas 5, 6, 7)
    runs = {k: v for k, v in TABLE5.items()}
    runs.update({r: f'{OAT}/{r}' for r in OAT_RUNS + ARCH_RUNS})
    data, missing = {}, []
    for k, d in runs.items():
        df = read_run(d)
        if df is None:
            missing.append(f'{k} ({d})')
        else:
            data[k] = df
    if missing:
        print('AVISO: sem holdout_probs_int8.csv, ignoradas:\n  ' + '\n  '.join(missing))
    if 'Abl1' not in data:
        sys.exit('ERRO: Ablation_LOESS_NLM_v3_ok é obrigatória (referência).')

    # Verificação: AUC por postos == sklearn
    _d = data['Abl1']
    assert abs(fast_auc(_d['label_int'].values, _d['p_int8'].values)
               - roc_auc_score(_d['label_int'].values, _d['p_int8'].values)) < 1e-12

    base = data['Abl1']
    for k, df in data.items():
        assert len(df) == len(base) and (df['filename'].values == base['filename'].values).all(), \
            f'{k}: conjunto de teste difere do da Ablação #1'
        assert (df['label_int'].values == base['label_int'].values).all(), f'{k}: rótulos diferem'

    y = base['label_int'].values.astype(int)
    db, grp = zip(*[group_of(f, ptb_map) for f in base['filename']])
    db, grp = np.array(db), np.array(grp)

    # Contagem de pacientes (item 4)
    tp_rows = []
    for name in ['CSN', 'PTB', 'OTHER']:
        mk = db == name
        if mk.any():
            g = pd.Series(grp[mk])
            multi = (g.value_counts() > 1).sum()
            tp_rows.append(dict(test_set='internal', dataset=name, records=int(mk.sum()),
                                patients=int(g.nunique()), patients_multi_record=int(multi),
                                AFIB=int(y[mk].sum()), SR=int((y[mk] == 0).sum())))
    tp_rows.append(dict(test_set='internal', dataset='ALL', records=len(y), patients=len(set(grp)),
                        patients_multi_record=int((pd.Series(grp).value_counts() > 1).sum()),
                        AFIB=int(y.sum()), SR=int((y == 0).sum())))
    print('\n=== Conjunto de teste interno ===')
    print(pd.DataFrame(tp_rows).to_string(index=False))

    rs = Resampler(grp, y)
    print(f'\nBootstrap: {N_BOOT} reamostragens de {rs.n_pat} pacientes (semente {SEED})')

    # 1. Métricas com IC (paciente vs registro)
    rows = []
    print('\n=== Métricas (IC 95% por paciente | por registro) ===')
    for k, df in data.items():
        p, pred = df['p_int8'].values, df['pred_int8'].values.astype(int)
        point, cis = metrics_with_ci(y, p, pred, rs)
        rows.append(dict(config=k, n=len(y), **point, **cis))
        print(f"{k:<22} AUC {point['AUC']:.4f} [{cis['AUC_lo_pat']:.4f}, {cis['AUC_hi_pat']:.4f}] "
              f"| [{cis['AUC_lo_rec']:.4f}, {cis['AUC_hi_rec']:.4f}]  "
              f"Sens {point['Sens']:.4f} [{cis['Sens_lo_pat']:.4f}, {cis['Sens_hi_pat']:.4f}]  "
              f"Spec {point['Spec']:.4f} [{cis['Spec_lo_pat']:.4f}, {cis['Spec_hi_pat']:.4f}]  "
              f"FN {point['FN']} FP {point['FP']}")

    # 2. Diferenças pareadas
    prow = []
    print('\n=== Diferenças pareadas A - B (IC 95% por paciente) ===')
    for tab, a, b in PAIRS:
        if a not in data or b not in data:
            continue
        A = (data[a]['p_int8'].values, data[a]['pred_int8'].values.astype(int))
        B = (data[b]['p_int8'].values, data[b]['pred_int8'].values.astype(int))
        r = paired(y, A, B, rs)
        prow.append(dict(table=tab, A=a, B=b, **r))
        print(f"{tab:<9} {a:<20} - {b:<17} ΔAUC {fmt(r['dAUC'])} [{fmt(r['dAUC_lo'])}, {fmt(r['dAUC_hi'])}]  "
              f"ΔSens {fmt(r['dSens'])} [{fmt(r['dSens_lo'])}, {fmt(r['dSens_hi'])}]  "
              f"ΔSpec {fmt(r['dSpec'])} [{fmt(r['dSpec_lo'])}, {fmt(r['dSpec_hi'])}]  "
              f"FN {r['FN_A']}/{r['FN_B']} FP {r['FP_A']}/{r['FP_B']}  McNemar p={r['mcnemar_p']:.3g}")

    # 3. Cross-domain e Georgia (conjuntos de teste próprios)
    extra = dict(CROSS)
    if args.georgia:
        extra['Georgia'] = args.georgia
    print('\n=== Cross-domain / Georgia (IC 95% por paciente | por registro) ===')
    for k, d in extra.items():
        df = read_run(d)
        if df is None:
            print(f'AVISO: {k}: sem holdout_probs_int8.csv em {d}')
            continue
        yk = df['label_int'].values.astype(int)
        dbk, gk = zip(*[group_of(f, ptb_map) for f in df['filename']])
        dbk, gk = np.array(dbk), np.array(gk)
        rsk = Resampler(gk, yk)
        point, cis = metrics_with_ci(yk, df['p_int8'].values, df['pred_int8'].values.astype(int), rsk)
        rows.append(dict(config=k, n=len(yk), **point, **cis))
        for name in sorted(set(dbk)):
            mk = dbk == name
            g = pd.Series(gk[mk])
            tp_rows.append(dict(test_set=k, dataset=name, records=int(mk.sum()), patients=int(g.nunique()),
                                patients_multi_record=int((g.value_counts() > 1).sum()),
                                AFIB=int(yk[mk].sum()), SR=int((yk[mk] == 0).sum())))
        print(f"{k:<10} n={len(yk):>5} pac={rsk.n_pat:>5}  AUC {point['AUC']:.4f} "
              f"[{cis['AUC_lo_pat']:.4f}, {cis['AUC_hi_pat']:.4f}] | [{cis['AUC_lo_rec']:.4f}, {cis['AUC_hi_rec']:.4f}]  "
              f"Sens {point['Sens']:.4f} [{cis['Sens_lo_pat']:.4f}, {cis['Sens_hi_pat']:.4f}]  "
              f"Spec {point['Spec']:.4f} [{cis['Spec_lo_pat']:.4f}, {cis['Spec_hi_pat']:.4f}]")

    pd.DataFrame(rows).to_csv(f'{args.out}/metrics_ci.csv', index=False)
    pd.DataFrame(prow).to_csv(f'{args.out}/paired_deltas.csv', index=False)
    pd.DataFrame(tp_rows).to_csv(f'{args.out}/test_patients.csv', index=False)
    print(f'\nSalvo em {args.out}/: metrics_ci.csv | paired_deltas.csv | test_patients.csv')


if __name__ == '__main__':
    main()
