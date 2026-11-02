"""
Transferência do limiar para o Georgia — Revisor 2 (threshold transfer) e Revisor 1, item 5.

Usa apenas as predições INT8 já calculadas (georgia_probs_int8.csv, variante A = ganho do
cabeçalho, a mesma da Tabela 9). Nenhum modelo é retreinado.

Parte 1 — Pontos de operação no Georgia (n = 2.069), IC 95% por bootstrap de registros:
  (a) τ implantado (0,8125, escolhido na validação da fonte)
  (b) τ que iguala a especificidade do teste interno (0,9879)      [oráculo, usa rótulos do Georgia]
  (c) τ que iguala a sensibilidade do teste interno (0,9618)      [oráculo]
  (d) τ de Youden no Georgia inteiro                               [oráculo]
  Em (b)-(d) o τ é re-selecionado em cada reamostragem, então o IC inclui a incerteza do τ.

Parte 2 — Recalibração do τ por site com n registros rotulados:
  Para n em N_CAL, sorteia n registros do Georgia (estratificado pela prevalência), escolhe τ
  na grade INT8 de 256 níveis por (i) Youden ou (ii) especificidade >= 0,9879 no subconjunto,
  e avalia nos 2.069 - n registros restantes. Repete R vezes.

Uso:
  cd /home/jose_alberto/Modelos_IEEE_Artigo_Revisao/
  python georgia_threshold_transfer.py
Saídas em Georgia/eval_ablation_loess_nlm_v3/threshold_transfer/:
  operating_points.csv | recalibration_curve.csv | recalibration_curve.pdf/.png
"""
import os
import sys

import numpy as np
import pandas as pd

IN_CSV = '/home/jose_alberto/Modelos_IEEE_Artigo_Revisao/Georgia/eval_ablation_loess_nlm_v3/georgia_probs_int8.csv'
OUT = '/home/jose_alberto/Modelos_IEEE_Artigo_Revisao/Georgia/eval_ablation_loess_nlm_v3/threshold_transfer'

TAU_DEPLOYED = 0.8125
SPEC_INTERNAL = 0.9879        # especificidade do modelo implantado no teste interno
SENS_INTERNAL = 0.9618        # sensibilidade do modelo implantado no teste interno
N_CAL = [25, 50, 100, 200, 400, 800]
R = 1000                      # repetições por n
N_BOOT, SEED = 2000, 42

# Grade de limiares do firmware: os 256 níveis da saída INT8 (scale 1/256, zero point -128)
GRID = (np.arange(-128, 128) + 128) / 256.0


def rates(y, p, tau):
    pred = p >= tau
    tp = np.sum(pred & (y == 1)); fn = np.sum(~pred & (y == 1))
    tn = np.sum(~pred & (y == 0)); fp = np.sum(pred & (y == 0))
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    return sens, spec, ppv, f1, int(fn), int(fp)


def grid_curves(y, p):
    """Sensibilidade e especificidade em cada nível da grade (vetorizado)."""
    P = p[None, :] >= GRID[:, None]
    pos, neg = y == 1, y == 0
    sens = P[:, pos].sum(1) / max(pos.sum(), 1)
    spec = (~P[:, neg]).sum(1) / max(neg.sum(), 1)
    return sens, spec


def tau_youden(y, p):
    s, sp = grid_curves(y, p)
    return GRID[int(np.argmax(s + sp - 1))]          # primeiro máximo na grade crescente


def tau_spec_target(y, p, target):
    """Menor τ da grade com especificidade >= target (maximiza a sensibilidade sob essa restrição)."""
    _, sp = grid_curves(y, p)
    ok = np.flatnonzero(sp >= target)
    return GRID[ok[0]] if ok.size else GRID[-1]


def tau_sens_target(y, p, target):
    """Maior τ da grade com sensibilidade >= target."""
    s, _ = grid_curves(y, p)
    ok = np.flatnonzero(s >= target)
    return GRID[ok[-1]] if ok.size else GRID[0]


def main():
    if not os.path.isfile(IN_CSV):
        sys.exit(f'ERRO: não encontrado {IN_CSV}')
    os.makedirs(OUT, exist_ok=True)
    d = pd.read_csv(IN_CSV)
    y = d['label_int'].values.astype(int)
    p = d['p_int8_A'].values.astype(float)
    assert len(y) == 2069 and y.sum() == 496, 'Georgia inesperado'
    # Confere que p está na grade INT8 e que o τ implantado reproduz a Tabela 9
    assert np.allclose(np.round(p * 256), p * 256), 'p_int8_A fora da grade INT8'
    s0, sp0, *_ = rates(y, p, TAU_DEPLOYED)
    print(f'Georgia: n={len(y)} AFIB={y.sum()} SR={(y == 0).sum()} | τ implantado: '
          f'Sens {s0:.4f} Spec {sp0:.4f} (Tabela 9: 0.7823 / 0.9943)')

    # ── Parte 1 ──────────────────────────────────────────────────────────────
    rules = {
        'a_deployed_tau':          lambda yy, pp: TAU_DEPLOYED,
        'b_spec_matched_internal': lambda yy, pp: tau_spec_target(yy, pp, SPEC_INTERNAL),
        'c_sens_matched_internal': lambda yy, pp: tau_sens_target(yy, pp, SENS_INTERNAL),
        'd_youden_georgia':        lambda yy, pp: tau_youden(yy, pp),
    }
    rng = np.random.RandomState(SEED)
    boots = []
    while len(boots) < N_BOOT:
        i = rng.randint(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            boots.append(i)

    rows = []
    print('\n=== Parte 1: pontos de operação no Georgia (IC 95%, bootstrap de registros) ===')
    print(f"{'regra':<26} {'τ':>8} {'Sens':>22} {'Spec':>22} {'PPV':>7} {'F1':>7} {'FN':>4} {'FP':>4}")
    for name, rule in rules.items():
        tau = rule(y, p)
        sens, spec, ppv, f1, fn, fp = rates(y, p, tau)
        bs = np.array([rates(y[i], p[i], rule(y[i], p[i]))[:2] for i in boots])
        slo, shi = np.nanpercentile(bs[:, 0], [2.5, 97.5])
        plo, phi = np.nanpercentile(bs[:, 1], [2.5, 97.5])
        rows.append(dict(rule=name, tau=tau, sens=sens, sens_lo=slo, sens_hi=shi,
                         spec=spec, spec_lo=plo, spec_hi=phi, ppv=ppv, f1=f1, fn=fn, fp=fp))
        print(f'{name:<26} {tau:8.6f} {sens:.4f} [{slo:.4f}, {shi:.4f}] '
              f'{spec:.4f} [{plo:.4f}, {phi:.4f}] {ppv:7.4f} {f1:7.4f} {fn:4d} {fp:4d}')
    pd.DataFrame(rows).to_csv(f'{OUT}/operating_points.csv', index=False)

    # ── Parte 2 ──────────────────────────────────────────────────────────────
    idx_pos, idx_neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    prev = y.mean()
    rng = np.random.RandomState(SEED)
    rec = []
    for n in N_CAL:
        n_pos = max(1, int(round(n * prev))); n_neg = n - n_pos
        for r in range(R):
            cal = np.r_[rng.choice(idx_pos, n_pos, replace=False), rng.choice(idx_neg, n_neg, replace=False)]
            mask = np.ones(len(y), bool); mask[cal] = False
            for crit, tau in (('youden', tau_youden(y[cal], p[cal])),
                              ('spec_target', tau_spec_target(y[cal], p[cal], SPEC_INTERNAL))):
                sens, spec, ppv, f1, fn, fp = rates(y[mask], p[mask], tau)
                rec.append(dict(n_cal=n, n_cal_afib=n_pos, rep=r, criterion=crit, tau=tau,
                                sens=sens, spec=spec, ppv=ppv, f1=f1))
    rec = pd.DataFrame(rec)

    def q(x, a):
        return np.nanpercentile(x, a)

    summ = (rec.groupby(['criterion', 'n_cal'])
               .agg(n_cal_afib=('n_cal_afib', 'first'),
                    tau_median=('tau', 'median'),
                    sens_median=('sens', 'median'), sens_p2_5=('sens', lambda x: q(x, 2.5)),
                    sens_p97_5=('sens', lambda x: q(x, 97.5)),
                    spec_median=('spec', 'median'), spec_p2_5=('spec', lambda x: q(x, 2.5)),
                    spec_p97_5=('spec', lambda x: q(x, 97.5)),
                    f1_median=('f1', 'median'),
                    frac_sens_ge_internal=('sens', lambda x: float(np.mean(x >= SENS_INTERNAL))))
               .reset_index())
    summ.to_csv(f'{OUT}/recalibration_curve.csv', index=False)
    print(f'\n=== Parte 2: recalibração do τ com n registros rotulados do Georgia ({R} repetições) ===')
    print('(avaliação nos registros restantes; mediana [percentis 2,5–97,5])')
    for crit in ('youden', 'spec_target'):
        print(f'\nCritério: {crit}')
        for _, s in summ[summ.criterion == crit].iterrows():
            print(f"  n={int(s.n_cal):>4} (AFIB {int(s.n_cal_afib):>3})  τ~{s.tau_median:.4f}  "
                  f"Sens {s.sens_median:.4f} [{s.sens_p2_5:.4f}, {s.sens_p97_5:.4f}]  "
                  f"Spec {s.spec_median:.4f} [{s.spec_p2_5:.4f}, {s.spec_p97_5:.4f}]  "
                  f"F1 {s.f1_median:.4f}  P(Sens>={SENS_INTERNAL})={s.frac_sens_ge_internal:.2f}")

    # ── Figura (critério de Youden) ─────────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, MUTED, GRID_C = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df'
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.edgecolor': MUTED,
                         'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})
    sY = summ[summ.criterion == 'youden']
    x = sY.n_cal.values
    fig, ax = plt.subplots(figsize=(3.45, 2.4))
    for col, key, lab, mk in ((BLUE, 'sens', 'Sensitivity', 'o'), (ORANGE, 'spec', 'Specificity', 's')):
        ax.fill_between(x, sY[f'{key}_p2_5'], sY[f'{key}_p97_5'], color=col, alpha=0.15, lw=0)
        ax.plot(x, sY[f'{key}_median'], color=col, lw=2, marker=mk, ms=4.5,
                mec='white', mew=0.8, label=lab)
    ax.axhline(s0, color=BLUE, lw=1, ls='--')
    ax.text(x[-1], s0 + 0.008, f'Sensitivity, deployed $\\tau$ ({s0:.3f})', color=MUTED,
            fontsize=7, ha='right', va='bottom')
    ax.set_ylim(min(s0, sY.spec_p2_5.min(), sY.sens_p2_5.min()) - 0.03, 1.005)
    ax.set_xscale('log')
    ax.set_xticks(x); ax.set_xticklabels([str(v) for v in x])
    ax.minorticks_off()
    ax.set_xlabel('Labelled Georgia recordings used to recalibrate $\\tau$')
    ax.set_ylabel('Metric on remaining recordings')
    ax.grid(axis='y', color=GRID_C, lw=0.6); ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, fontsize=7, frameon=False,
              handlelength=1.6, columnspacing=1.0, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(f'{OUT}/recalibration_curve.pdf'); fig.savefig(f'{OUT}/recalibration_curve.png', dpi=200)
    print(f'\nSalvo em {OUT}/: operating_points.csv | recalibration_curve.csv | recalibration_curve.pdf/.png')


if __name__ == '__main__':
    main()
