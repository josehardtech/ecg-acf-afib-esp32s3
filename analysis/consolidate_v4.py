"""
Consolida as execuções v4 — Revisor 1, itens 4 (escalonador por fold) e 6 (float32 × INT8).

  A. Reprodutibilidade: v4_reference_seed42 deve reproduzir o modelo implantado
     (mesmo SHA-256 do .tflite e mesmas predições INT8 da Ablation_LOESS_NLM_v3_ok).
  B. Float32 × INT8 no teste (n = 4.025), modelo implantado: AUC, Sens, Spec, F1 com IC por
     paciente, ΔAUC pareado (INT8 − float32) e concordância das decisões.
     Também lista float32 × INT8 (pontuais) das quatro configurações da Tabela 5.
  C. Escalonador por fold: diagnóstico dos parâmetros e comparação do modelo retreinado
     com escalonador por fold contra o implantado (ΔAUC, ΔSens, ΔSpec por paciente, McNemar).

Uso:
  cd /home/jose_alberto/Modelos_IEEE_Artigo_Revisao/Sensibilidade_OAT/
  python consolidate_v4.py
"""
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata

ROOT = '/home/jose_alberto/Modelos_IEEE_Artigo_Revisao'
OAT = f'{ROOT}/Sensibilidade_OAT'
DEPLOYED = f'{ROOT}/Ablation_LOESS_NLM_v3_ok'
REF4 = f'{OAT}/v4_reference_seed42'
FS4 = f'{OAT}/v4_fold_scaler_seed42'
TABLE5 = {'Full': f'{ROOT}/Dual_cnn_branch_v3_ok', 'Abl1': DEPLOYED,
          'Abl2': f'{ROOT}/Ablation_CNN_LOESS_NLM_v3_ok', 'Abl3': f'{ROOT}/Ablation_All_Filters_RAW_v3_ok'}
PTBXL_DB = '/mnt/windows/Doutorado/Datasets/PTB_XL_Raw/ptbxl_database.csv'
EXPECTED_SHA = 'a7582a6d1dd772754886b242c2c23a9c1fd36a7dbf7d261e4761e14b5e086b2e'
N_BOOT, SEED = 2000, 42
PTB_RE = re.compile(r'^(\d+)_(hr|lr)\.csv$')


def need(p):
    if not os.path.exists(p):
        sys.exit(f'ERRO: não encontrado: {p}')
    return p


def sha_tflite(folder):
    f = sorted(glob.glob(os.path.join(folder, '*int8*.tflite')))
    return hashlib.sha256(open(f[0], 'rb').read()).hexdigest() if f else None


def auc(y, p):
    n1 = int(y.sum()); n0 = len(y) - n1
    r = rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def rates(y, pred):
    tp = np.sum(pred & (y == 1)); fn = np.sum(~pred & (y == 1))
    tn = np.sum(~pred & (y == 0)); fp = np.sum(pred & (y == 0))
    return tp / (tp + fn), tn / (tn + fp), 2 * tp / (2 * tp + fp + fn), int(fn), int(fp)


def mets(y, p, pred):
    s, sp, f1, fn, fp = rates(y, pred)
    return dict(AUC=auc(y, p), Sens=s, Spec=sp, F1=f1, FN=fn, FP=fp)


# ── Grupos por paciente ──────────────────────────────────────────────────────
db = pd.read_csv(need(PTBXL_DB), usecols=['ecg_id', 'patient_id'])
pmap = dict(zip(db.ecg_id.astype(int), db.patient_id.astype(float).astype(int)))


def group(fn):
    m = PTB_RE.match(fn)
    return f'PTB_{pmap[int(m.group(1))]}' if m else f'REC_{fn}'


def clustered_boot(groups, y):
    codes, uniq = pd.factorize(groups)
    members = [np.flatnonzero(codes == g) for g in range(len(uniq))]
    rng = np.random.RandomState(SEED); out = []
    while len(out) < N_BOOT:
        idx = np.concatenate([members[g] for g in rng.randint(0, len(uniq), len(uniq))])
        if 0 < y[idx].sum() < len(idx):
            out.append(idx)
    return out


def ci(v):
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def load_int8(folder):
    d = pd.read_csv(need(os.path.join(folder, 'holdout_probs_int8.csv')))
    return d.sort_values('filename').reset_index(drop=True)


def load_meta(folder):
    return json.load(open(need(os.path.join(folder, 'metrics_final.json'))))


# ── A. Reprodutibilidade ─────────────────────────────────────────────────────
print('=== A. Reprodutibilidade da referência v4 ===')
dep, ref = load_int8(DEPLOYED), load_int8(REF4)
same_files = (dep.filename.values == ref.filename.values).all()
same_p = same_files and np.array_equal(dep.int8_out.values, ref.int8_out.values)
sha_ref = sha_tflite(REF4)
print(f'SHA-256 .tflite v4_reference_seed42: {sha_ref}')
print(f'  igual ao modelo implantado ({EXPECTED_SHA[:16]}...)? {sha_ref == EXPECTED_SHA}')
print(f'  saídas INT8 idênticas às do modelo implantado? {same_p}')
if not same_p:
    print('  AVISO: a referência v4 não reproduziu o modelo implantado; a comparação float32 × INT8 '
          'abaixo vale para o modelo v4, não necessariamente para o implantado.')

y = ref.label_int.values.astype(int)
grp = np.array([group(f) for f in ref.filename])
boots = clustered_boot(grp, y)
print(f'Teste: {len(y)} registros, {len(set(grp))} pacientes; bootstrap {N_BOOT} por paciente')

# ── B. Float32 × INT8 ────────────────────────────────────────────────────────
print('\n=== B. Float32 × INT8 (modelo de referência, teste n = 4.025) ===')
fl = pd.read_csv(need(os.path.join(REF4, 'holdout_probs_float32.csv'))).sort_values('filename').reset_index(drop=True)
assert (fl.filename.values == ref.filename.values).all() and (fl.label_int.values == y).all()
diag_ref = json.load(open(need(os.path.join(REF4, 'scaler_fold_diag.json'))))
meta_ref = load_meta(REF4)
tau_f = float(diag_ref['tau_float32']); tau_q = float(meta_ref['thresholds']['tau_int8_firmware'])
pf, pq = fl.p_float32.values, ref.p_int8.values
dec_f, dec_q = pf >= tau_f, pq >= tau_q
rows = []
for name, p, dec, tau in (('float32', pf, dec_f, tau_f), ('INT8', pq, dec_q, tau_q)):
    m = mets(y, p, dec)
    bs = [mets(y[i], p[i], dec[i]) for i in boots]
    r = dict(model=name, tau=tau, **m)
    for k in ('AUC', 'Sens', 'Spec', 'F1'):
        r[f'{k}_lo'], r[f'{k}_hi'] = ci([b[k] for b in bs])
    rows.append(r)
    print(f"{name:<8} τ={tau:.6f}  AUC {m['AUC']:.4f} [{r['AUC_lo']:.4f}, {r['AUC_hi']:.4f}]  "
          f"Sens {m['Sens']:.4f} [{r['Sens_lo']:.4f}, {r['Sens_hi']:.4f}]  "
          f"Spec {m['Spec']:.4f} [{r['Spec_lo']:.4f}, {r['Spec_hi']:.4f}]  "
          f"F1 {m['F1']:.4f}  FN {m['FN']} FP {m['FP']}")
d_auc = auc(y, pq) - auc(y, pf)
lo, hi = ci([auc(y[i], pq[i]) - auc(y[i], pf[i]) for i in boots])
agree = np.mean(dec_f == dec_q)
b = int(np.sum((dec_q == y) & (dec_f != y))); c_ = int(np.sum((dec_q != y) & (dec_f == y)))
pmc = binomtest(b, b + c_, 0.5).pvalue if b + c_ else 1.0
print(f'ΔAUC (INT8 − float32) = {d_auc:+.4f} [{lo:+.4f}, {hi:+.4f}]')
print(f'Concordância das decisões: {agree * 100:.2f}% ({int(np.sum(dec_f != dec_q))} registros discordantes; '
      f'só INT8 certo {b}, só float32 certo {c_}; McNemar p = {pmc:.3g})')
pd.DataFrame(rows).assign(dAUC=d_auc, dAUC_lo=lo, dAUC_hi=hi, agreement=agree, mcnemar_p=pmc) \
  .to_csv(f'{OAT}/v4_float32_vs_int8.csv', index=False)

print('\nFloat32 × INT8 (pontuais, metrics_final.json) — configurações da Tabela 5:')
print(f"{'config':<6} {'AUC f32':>8} {'AUC int8':>9} {'Sens f32':>9} {'Sens int8':>10} "
      f"{'Spec f32':>9} {'Spec int8':>10} {'F1 f32':>7} {'F1 int8':>8}")
t5 = []
for k, d in TABLE5.items():
    try:
        m = load_meta(d)
    except SystemExit:
        print(f'{k:<6} (sem metrics_final.json)'); continue
    f, q = m['float32_with_tau_float'], m['int8_with_tau_firmware']
    t5.append(dict(config=k, auc_f32=f['auc'], auc_int8=q['AUC'], sens_f32=f['sens'], sens_int8=q['Sensitivity'],
                   spec_f32=f['spec'], spec_int8=q['Specificity'], f1_f32=f['f1'], f1_int8=q['F1-Score'],
                   tflite_bytes=m.get('tflite_bytes')))
    print(f"{k:<6} {f['auc']:8.4f} {q['AUC']:9.4f} {f['sens']:9.4f} {q['Sensitivity']:10.4f} "
          f"{f['spec']:9.4f} {q['Specificity']:10.4f} {f['f1']:7.4f} {q['F1-Score']:8.4f}")
pd.DataFrame(t5).to_csv(f'{OAT}/v4_float32_vs_int8_table5.csv', index=False)

# ── C. Escalonador por fold ─────────────────────────────────────────────────
print('\n=== C. Escalonador da ACF por fold ===')
dg = pd.DataFrame(diag_ref['diag'])
print('Diferença entre escalonador do fold e global (ACF, 5.000 lags):')
print(dg[['fold', 'n_fit', 'max_abs_dmean_in_sd', 'max_rel_dstd', 'val_mean_abs_dz',
          'val_p99_abs_dz', 'val_max_abs_dz']].to_string(index=False, float_format=lambda v: f'{v:.5f}'))
print(f"Pior caso entre folds: |Δμ| ≤ {dg.max_abs_dmean_in_sd.max():.4f} σ; |Δσ|/σ ≤ {dg.max_rel_dstd.max():.4f}; "
      f"|Δz| médio na validação ≤ {dg.val_mean_abs_dz.max():.5f}")

fs = load_int8(FS4); meta_fs = load_meta(FS4)
diag_fs = json.load(open(need(os.path.join(FS4, 'scaler_fold_diag.json'))))
assert diag_fs['fold_scaler'] is True, 'v4_fold_scaler_seed42 não rodou com OAT_FOLD_SCALER=1'
assert (fs.filename.values == ref.filename.values).all()
pa, pb = fs.p_int8.values, ref.p_int8.values
da, db_ = fs.pred_int8.values.astype(bool), ref.pred_int8.values.astype(bool)
ma, mb = mets(y, pa, da), mets(y, pb, db_)
print(f"\n{'':<24} {'best fold':>9} {'τ float32':>10} {'τ INT8':>9} {'AUC':>7} {'Sens':>7} {'Spec':>7} {'FN':>4} {'FP':>4}")
for lab, meta, dgx, m in (('global (implantado)', meta_ref, diag_ref, mb), ('por fold', meta_fs, diag_fs, ma)):
    print(f"{lab:<24} {dgx['best_fold']:>9} {dgx['tau_float32']:>10.6f} {meta['thresholds']['tau_int8_firmware']:>9.6f} "
          f"{m['AUC']:7.4f} {m['Sens']:7.4f} {m['Spec']:7.4f} {m['FN']:4d} {m['FP']:4d}")
out = {}
for k in ('AUC', 'Sens', 'Spec'):
    v = [mets(y[i], pa[i], da[i])[k] - mets(y[i], pb[i], db_[i])[k] for i in boots]
    out[k] = (ma[k] - mb[k], *ci(v))
b = int(np.sum((da == y) & (db_ != y))); c_ = int(np.sum((da != y) & (db_ == y)))
pmc2 = binomtest(b, b + c_, 0.5).pvalue if b + c_ else 1.0
print('Por fold − global (IC 95% por paciente):  ' + '  '.join(
    f'Δ{k} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]' for k, v in out.items()) + f'  McNemar p = {pmc2:.3g}')
print('Referência de escala: entre sementes da configuração implantada, |ΔAUC| até 0,0009 e |ΔSens| até 0,019.')
pd.DataFrame([dict(comparison='fold_scaler - global', **{f'd{k}': v[0] for k, v in out.items()},
                   **{f'd{k}_lo': v[1] for k, v in out.items()}, **{f'd{k}_hi': v[2] for k, v in out.items()},
                   mcnemar_p=pmc2)]).to_csv(f'{OAT}/v4_fold_scaler_comparison.csv', index=False)
dg.to_csv(f'{OAT}/v4_scaler_fold_diag.csv', index=False)
print(f'\nSalvo em {OAT}/: v4_float32_vs_int8.csv | v4_float32_vs_int8_table5.csv | '
      f'v4_fold_scaler_comparison.csv | v4_scaler_fold_diag.csv')
