"""
Regenera a Fig. 4 (ROC do modelo implantado, No-LOESS/NLM v3, INT8) a partir das predições atuais.

Fonte: Ablation_LOESS_NLM_v3_ok/holdout_probs_int8.csv (patient-exclusive, n = 4.025).
Ponto de operação: τ do firmware lido do metrics_final.json (0,8125).
Gera roc_curve_esp32_loess_nlm.pdf (mesmo nome do arquivo usado no main.tex) e uma .png de conferência.

Uso:
  cd /home/jose_alberto/Modelos_IEEE_Artigo_Revisao/
  python make_fig4.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

FOLDER = '/home/jose_alberto/Modelos_IEEE_Artigo_Revisao/Ablation_LOESS_NLM_v3_ok'
OUT = '/home/jose_alberto/Modelos_IEEE_Artigo_Revisao/Fig4_v3'

os.makedirs(OUT, exist_ok=True)
d = pd.read_csv(os.path.join(FOLDER, 'holdout_probs_int8.csv'))
meta = json.load(open(os.path.join(FOLDER, 'metrics_final.json')))
tau = float(meta['thresholds']['tau_int8_firmware'])
y, p = d.label_int.values.astype(int), d.p_int8.values.astype(float)
pred = p >= tau
tp = int(np.sum(pred & (y == 1))); fn = int(np.sum(~pred & (y == 1)))
tn = int(np.sum(~pred & (y == 0))); fp = int(np.sum(pred & (y == 0)))
sens, spec, auc = tp / (tp + fn), tn / (tn + fp), roc_auc_score(y, p)
print(f'n={len(y)}  τ={tau:.6f}  AUC={auc:.4f}  Sens={sens:.4f}  Spec={spec:.4f}  TP={tp} FN={fn} FP={fp} TN={tn}')
assert len(y) == 4025 and abs(tau - 0.8125) < 1e-9, 'Conjunto ou τ inesperado'

BLUE, INK, MUTED, GRID_C = '#2a78d6', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})
fpr, tpr, _ = roc_curve(y, p)
fig, ax = plt.subplots(figsize=(3.45, 3.0))
ax.plot([0, 1], [0, 1], ls='--', lw=0.8, color=MUTED)
ax.plot(fpr, tpr, lw=2, color=BLUE, label=f'LiteRT INT8, AUC = {auc:.4f}')
ax.plot(1 - spec, sens, 'o', ms=7, color=BLUE, mec='white', mew=1.2, zorder=5,
        label=f'$\\tau$ = {tau:.4f} (Sens. {sens:.4f}, Spec. {spec:.4f})')
# Detalhe do canto superior esquerdo, onde a curva se decide
ins = ax.inset_axes([0.42, 0.22, 0.52, 0.45])
ins.plot(fpr, tpr, lw=1.6, color=BLUE)
ins.plot(1 - spec, sens, 'o', ms=6, color=BLUE, mec='white', mew=1.0, zorder=5)
ins.set_xlim(0, 0.10); ins.set_ylim(0.90, 1.0)
ins.tick_params(labelsize=6.5)
ins.grid(color=GRID_C, lw=0.5)
for sp in ins.spines.values():
    sp.set_edgecolor(MUTED); sp.set_linewidth(0.6)
ax.set_xlim(-0.01, 1.0); ax.set_ylim(0, 1.01)
ax.set_xlabel('1 $-$ Specificity (FPR)'); ax.set_ylabel('Sensitivity (TPR)')
ax.grid(color=GRID_C, lw=0.6); ax.set_axisbelow(True)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), fontsize=7, frameon=False, borderaxespad=0.2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'roc_curve_esp32_loess_nlm.pdf'))
fig.savefig(os.path.join(OUT, 'roc_curve_esp32_loess_nlm.png'), dpi=200)
print(f'Salvo em {OUT}/roc_curve_esp32_loess_nlm.pdf (.png para conferência)')
