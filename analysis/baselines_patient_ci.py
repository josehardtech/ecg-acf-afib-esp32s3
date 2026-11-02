"""
ΔAUC por PACIENTE (cluster bootstrap) dos baselines da Tabela 7 em relação à referência (seed 42).
Usa as mesmas funções e a mesma reamostragem de paired_stats_patient.py.

Uso (na pasta onde está paired_stats_patient.py):
  cd /home/jose_alberto/Modelos_IEEE_Artigo_Revisao/
  conda activate pytorch_env
  python baselines_patient_ci.py
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paired_stats_patient as ps

B = f'{ps.ROOT}/Baselines_E5_E7'
RUNS = {
    'reference_seed42': f'{ps.OAT}/reference_seed42',
    'MobileNet-1D':     f'{B}/arch_mobilenet',
    'CNN-LSTM':         f'{B}/base_lstm',
    'Random forest':    f'{B}/base_rf_rr',
    'RBF-SVM':          f'{B}/base_svm_rr',
}
data = {}
for k, d in RUNS.items():
    df = ps.read_run(d)
    if df is None:
        print(f'AVISO: sem holdout_probs_int8.csv em {d}'); continue
    data[k] = df
ref = data['reference_seed42']
for k, df in data.items():
    assert (df.filename.values == ref.filename.values).all() and (df.label_int.values == ref.label_int.values).all(), k

ptb = ps.load_ptbxl_map(ps.PTBXL_DB)
y = ref.label_int.values.astype(int)
grp = np.array([ps.group_of(f, ptb)[1] for f in ref.filename])
rs = ps.Resampler(grp, y)
print(f'Bootstrap: {ps.N_BOOT} reamostragens de {rs.n_pat} pacientes')

R = (ref.p_int8.values, ref.pred_int8.values.astype(int))
rows = []
for k, df in data.items():
    if k == 'reference_seed42':
        continue
    A = (df.p_int8.values, df.pred_int8.values.astype(int))
    r = ps.paired(y, A, R, rs)
    point, cis = ps.metrics_with_ci(y, *A, rs)
    rows.append(dict(model=k, AUC=point['AUC'], F1=point['F1'], FN=point['FN'], FP=point['FP'],
                     dAUC=r['dAUC'], dAUC_lo=r['dAUC_lo'], dAUC_hi=r['dAUC_hi'], mcnemar_p=r['mcnemar_p']))
    print(f"{k:<14} AUC {point['AUC']:.4f}  ΔAUC {r['dAUC']:+.4f} [{r['dAUC_lo']:+.4f}, {r['dAUC_hi']:+.4f}]  "
          f"FN {point['FN']} FP {point['FP']}  McNemar p={r['mcnemar_p']:.3g}")
os.makedirs('paired_stats_patient', exist_ok=True)
pd.DataFrame(rows).to_csv('paired_stats_patient/baselines_patient.csv', index=False)
print('Salvo em paired_stats_patient/baselines_patient.csv')
