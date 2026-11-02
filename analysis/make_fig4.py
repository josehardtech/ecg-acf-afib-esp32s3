"""
Fig. 4 — curva ROC do modelo implantado (No-LOESS/NLM, LiteRT INT8, teste exclusivo por paciente).
Versão sem inset, no tamanho de uma coluna. Gera roc_curve_esp32_loess_nlm.pdf (nome usado no main.tex).

Uso:  python make_fig4.py [holdout_probs_int8.csv] [pasta_saida]
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

CSV = sys.argv[1] if len(sys.argv) > 1 else \
    '/home/jose_alberto/Modelos_IEEE_Artigo_Revisao/Ablation_LOESS_NLM_v3_ok/holdout_probs_int8.csv'
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(CSV))
TAU_INT8 = 80                      # tau = 0.8125  <=>  saída INT8 >= 80

d = pd.read_csv(CSV)
y, s = d.label_int.values, d.int8_out.values
fpr, tpr, _ = roc_curve(y, s)
auc = roc_auc_score(y, s)
pred = s >= TAU_INT8
P, N = (y == 1).sum(), (y == 0).sum()
se, sp = (pred & (y == 1)).sum() / P, (~pred & (y == 0)).sum() / N

plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6})
fig, ax = plt.subplots(figsize=(3.36, 2.75))
ax.plot([0, 1], [0, 1], ls='--', lw=0.8, color='0.45', zorder=1)
ax.plot(fpr, tpr, lw=1.6, color='#2a78d6', label=f'LiteRT INT8, AUC = {auc:.4f}', zorder=2)
ax.plot(1 - sp, se, 'o', ms=5.5, mfc='#2a78d6', mec='white', mew=1.0, zorder=3,
        label=f'$\\tau$ = 0.8125 (Sens. {se:.4f}, Spec. {sp:.4f})')
ax.set_xlim(-0.01, 1.0); ax.set_ylim(0, 1.01)
ax.set_xlabel(r'1 $-$ Specificity (FPR)'); ax.set_ylabel('Sensitivity (TPR)')
ax.grid(color='0.9', lw=0.5); ax.set_axisbelow(True)
for k in ('top', 'right'): ax.spines[k].set_visible(False)
ax.legend(loc='lower right', frameon=True, framealpha=1, edgecolor='0.8', fontsize=7)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, 'roc_curve_esp32_loess_nlm.pdf'))
fig.savefig(os.path.join(OUT, 'roc_curve_esp32_loess_nlm.png'), dpi=200)
print(f'AUC {auc:.4f} | Sens {se:.4f} | Spec {sp:.4f} -> {OUT}/roc_curve_esp32_loess_nlm.pdf')
