"""
Figures. Every function returns a matplotlib Figure, so the same code both
displays the figure inline in a notebook and saves it to disk.

Nothing here calls matplotlib.use(), so the notebook's own backend is
respected and figures render inline.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .entropy import entropy_cells, SEED

C_UNW, C_ENT, C_BACK, C_THIRD = '#4C72B0', '#C44E52', '#55A868', '#8172B2'
FIGDIR = Path(__file__).resolve().parents[1] / 'figures'

STYLE = {
    'font.size': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'figure.dpi': 110,
    'savefig.bbox': 'tight',
}


def use_style():
    plt.rcParams.update(STYLE)


def save(fig, name, formats=('pdf', 'png')):
    """Write the figure to figures/ in each requested format."""
    FIGDIR.mkdir(exist_ok=True)
    for ext in formats:
        fig.savefig(FIGDIR / f'{name}.{ext}', dpi=300 if ext == 'png' else None)
    return fig


# ------------------------------------------------------------------ Figure 1
def fig_coverage(cov):
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))
    K = cov['K']
    ax[0].plot(K, np.log(K), 'k--', lw=1, label='ln K (maximum)')
    ax[0].plot(K, cov.H_background, 'o-', color=C_BACK, label='background')
    ax[0].plot(K, cov.H_presence, 's-', color=C_ENT, label='presence')
    ax[0].set_ylabel('Shannon entropy $H(P)$')
    ax[0].set_title('(a) Coverage entropy', loc='left', fontweight='bold')

    ax[1].axhline(1, color='k', ls='--', lw=1)
    ax[1].plot(K, cov.J_background, 'o-', color=C_BACK, label='background')
    ax[1].plot(K, cov.J_presence, 's-', color=C_ENT, label='presence')
    ax[1].set_ylabel("Pielou's evenness $J = H/\\ln K$")
    ax[1].set_ylim(0.80, 1.02)
    ax[1].set_title('(b) Evenness', loc='left', fontweight='bold')

    ax[2].plot(K, cov.KL, 'D-', color=C_THIRD)
    ax[2].set_ylabel('$D_{KL}(P_{\\mathrm{pres}} \\| P_{\\mathrm{back}})$')
    ax[2].set_title('(c) Divergence', loc='left', fontweight='bold')

    for a in ax:
        a.set_xscale('log')
        a.set_xlabel('number of cells $K$')
    for a in ax[:2]:
        a.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ Figure 2
def fig_weight_function(X_scaled, y, K=100):
    km, p, H, w_cell = entropy_cells(X_scaled[y == 1], K)
    realised = w_cell[km.labels_]

    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.1))
    grid = np.linspace(p.min() * 0.6, p.max() * 1.2, 400)
    ax[0].plot(grid, -np.log(grid) / H, color=C_ENT, lw=1.8)
    ax[0].axhline(1, color='k', ls='--', lw=1)
    ax[0].axvline(1 / len(p), color='k', ls=':', lw=1)
    ax[0].plot(p, w_cell, 'o', ms=3, color='#333333', alpha=0.55,
               label='observed cells')
    ax[0].annotate('$w=1$ at $p_i=1/K$\n(no reweighting under\nuniform coverage)',
                   xy=(1 / len(p), 1), xytext=(2.1 / len(p), 1.32), fontsize=7.5,
                   arrowprops=dict(arrowstyle='->', lw=0.8))
    ax[0].set_xlabel('cell proportion $p_i$')
    ax[0].set_ylabel('weight $w_i = -\\ln p_i / H(P)$')
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title('(a) Weight function', loc='left', fontweight='bold')

    ax[1].hist(realised, bins=34, color=C_ENT, alpha=0.8, edgecolor='w', lw=0.4)
    ax[1].axvline(realised.mean(), color='k', lw=1.4,
                  label=f'mean = {realised.mean():.3f}')
    ax[1].set_xlabel('realised weight')
    ax[1].set_ylabel('presence records')
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title(f'(b) Realised weights ($K={K}$)', loc='left',
                    fontweight='bold')
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ Figure 3
def fig_entropy_cells_pca(X_scaled, y, K=100):
    km, p, H, w_cell = entropy_cells(X_scaled[y == 1], K)
    pca = PCA(n_components=2).fit(X_scaled)
    Zp, Zb = pca.transform(X_scaled[y == 1]), pca.transform(X_scaled[y == 0])
    Zc = pca.transform(km.cluster_centers_)
    ev = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.5))
    ax[0].scatter(Zb[:, 0], Zb[:, 1], s=3, c='#CCCCCC', alpha=0.5, lw=0,
                  label='background')
    ax[0].scatter(Zp[:, 0], Zp[:, 1], s=4, c=C_ENT, alpha=0.45, lw=0,
                  label='presence')
    ax[0].legend(frameon=False, fontsize=8, markerscale=2.5)
    ax[0].set_title('(a) Sample in covariate space', loc='left',
                    fontweight='bold')

    sc = ax[1].scatter(Zc[:, 0], Zc[:, 1], c=w_cell, s=28 + 240 * p,
                       cmap='RdYlBu_r', edgecolor='k', lw=0.35)
    cb = fig.colorbar(sc, ax=ax[1])
    cb.set_label('entropy weight $w_i$', fontsize=8)
    cb.ax.axhline(1.0, color='k', lw=1.2)
    ax[1].set_title(f'(b) Entropy cells ($K={K}$)', loc='left',
                    fontweight='bold')
    ax[1].text(0.02, 0.02,
               'marker size $\\propto p_i$\nblue: over-represented ($w<1$)\n'
               'red: under-represented ($w>1$)',
               transform=ax[1].transAxes, fontsize=7, va='bottom')

    for a in ax:
        a.set_xlabel(f'PC1 ({ev[0]:.1f}% var)')
        a.set_ylabel(f'PC2 ({ev[1]:.1f}% var)')
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ Figure 4
def fig_paired_auc(cv, metric='AUC',
                   order=('Random Forest', 'MaxEnt', 'GAM', 'Bayesian')):
    models = [m for m in order if m in cv.model.unique()]
    fig, axes = plt.subplots(1, len(models), figsize=(2.8 * len(models), 3.2))
    axes = np.atleast_1d(axes)
    for a, m in zip(axes, models):
        u = cv[(cv.model == m) & (cv.weighting == 'unweighted')] \
            .sort_values('fold')[metric].to_numpy()
        e = cv[(cv.model == m) & (cv.weighting == 'entropy')] \
            .sort_values('fold')[metric].to_numpy()
        for i in range(len(u)):
            a.plot([0, 1], [u[i], e[i]], '-', color='#999999', lw=0.9,
                   alpha=0.8, marker='o', ms=3.2)
        a.errorbar([0, 1], [u.mean(), e.mean()], yerr=[u.std(), e.std()],
                   fmt='D', ms=7, capsize=4, lw=2, color=C_ENT, zorder=5)
        a.set_xticks([0, 1])
        a.set_xticklabels(['unweighted', 'entropy'], fontsize=8)
        a.set_xlim(-0.35, 1.35)
        a.set_title(m, fontweight='bold', fontsize=9.5)
        a.text(0.5, 0.04, f'$\\Delta$={e.mean() - u.mean():+.4f}',
               transform=a.transAxes, ha='center', fontsize=8)
    axes[0].set_ylabel(f'held-out {metric}')
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ Figure 5
def fig_k_sensitivity(ks):
    g = ks.groupby(['K', 'weighting'])[['AUC', 'TSS']].agg(['mean', 'std']) \
        .reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.1))
    for metric, a in zip(['AUC', 'TSS'], ax):
        for tag, col in (('unweighted', C_UNW), ('entropy', C_ENT)):
            s = g[g.weighting == tag]
            a.errorbar(s['K'], s[(metric, 'mean')], yerr=s[(metric, 'std')],
                       marker='o', ms=4, capsize=3, lw=1.4, color=col, label=tag)
        a.set_xscale('log')
        a.set_xlabel('number of entropy cells $K$')
        a.set_ylabel(f'held-out {metric}')
        a.legend(frameon=False, fontsize=8)
    ax[0].set_title('(a) Discrimination', loc='left', fontweight='bold')
    ax[1].set_title('(b) True skill statistic', loc='left', fontweight='bold')
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ Figure 6
def fig_blocked_cv(cv_random, cv_blocked,
                   order=('Random Forest', 'MaxEnt', 'GAM', 'Bayesian')):
    models = [m for m in order if m in cv_random.model.unique()]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ticks = []
    for i, m in enumerate(models):
        a = cv_random[(cv_random.model == m) &
                      (cv_random.weighting == 'unweighted')].AUC.to_numpy()
        b = cv_blocked[(cv_blocked.model == m) &
                       (cv_blocked.weighting == 'unweighted')].AUC.to_numpy()
        b1 = ax.boxplot([a], positions=[i * 3], widths=0.85, patch_artist=True,
                        medianprops=dict(color='k'))
        b2 = ax.boxplot([b], positions=[i * 3 + 1], widths=0.85,
                        patch_artist=True, medianprops=dict(color='k'))
        b1['boxes'][0].set_facecolor(C_UNW)
        b1['boxes'][0].set_alpha(0.75)
        b2['boxes'][0].set_facecolor('#DD8452')
        b2['boxes'][0].set_alpha(0.85)
        ticks.append(i * 3 + 0.5)
    ax.set_xticks(ticks)
    ax.set_xticklabels(models)
    ax.set_ylabel('held-out AUC')
    ax.legend(handles=[Patch(facecolor=C_UNW, alpha=0.75,
                             label='random cross-validation'),
                       Patch(facecolor='#DD8452', alpha=0.85,
                             label='watershed-blocked cross-validation')],
              frameon=False, fontsize=8, loc='lower left')
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ Figure 7
def fig_calibration(y, oof):
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.2))
    ax[0].plot([0, 1], [0, 1], 'k--', lw=1, label='perfect calibration')
    for tag, col in (('unweighted', C_UNW), ('entropy', C_ENT)):
        frac, mean_pred = calibration_curve(y, oof[tag], n_bins=10,
                                            strategy='quantile')
        ax[0].plot(mean_pred, frac, 'o-', ms=4, color=col, label=tag)
    ax[0].set_xlabel('mean predicted probability')
    ax[0].set_ylabel('observed frequency')
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title('(a) Calibration (out-of-fold)', loc='left',
                    fontweight='bold')

    for tag, col in (('unweighted', C_UNW), ('entropy', C_ENT)):
        ax[1].hist(oof[tag], bins=40, histtype='step', lw=1.5, color=col,
                   label=tag)
    ax[1].set_yscale('log')
    ax[1].set_xlabel('predicted probability')
    ax[1].set_ylabel('records (log scale)')
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title('(b) Prediction distribution', loc='left',
                    fontweight='bold')
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ Figure 8
PRETTY = {
    'PRISM_tmean_30yr_normal_800mM2_annual_asc': 'mean annual temperature',
    'GDD_corngrowing_dds_2005': 'growing degree-days',
    'NPP_mean_00_15': 'net primary productivity',
    'wtrshd_size': 'catchment area',
    'east_west_asp': 'aspect (E-W)',
    'north_south_asp': 'aspect (N-S)',
    'springs_cd': 'distance to springs',
    'streams_cd': 'distance to streams',
    'wetlands_cd': 'distance to wetlands',
    'slope': 'slope',
}


def fig_importance(imp, features):
    iu, ie = imp['unweighted'].mean(0), imp['entropy'].mean(0)
    su, se = imp['unweighted'].std(0), imp['entropy'].std(0)
    order = np.argsort(iu)
    labels = [PRETTY.get(features[i], features[i]) for i in order]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ypos = np.arange(len(order))
    ax.barh(ypos - 0.2, iu[order], height=0.38, xerr=su[order], color=C_UNW,
            alpha=0.85, label='unweighted', error_kw=dict(lw=0.8))
    ax.barh(ypos + 0.2, ie[order], height=0.38, xerr=se[order], color=C_ENT,
            alpha=0.85, label='entropy-weighted', error_kw=dict(lw=0.8))
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('permutation importance (drop in held-out AUC)')
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ Figure 9
def fig_posterior(X_scaled, y, weights_entropy, fit_bayes):
    results = {}
    for tag, w in (('unweighted', np.ones(len(y))), ('entropy', weights_entropy)):
        _, fit = fit_bayes(X_scaled, y, w)
        P = fit.predict_draws(X_scaled)
        results[tag] = dict(var=P.var(axis=1), draws=fit.draws,
                            acceptance=fit.acceptance)

    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.4))
    vp = ax[0].violinplot([results['unweighted']['var'], results['entropy']['var']],
                          positions=[0, 1], showmedians=True, widths=0.8)
    for body, col in zip(vp['bodies'], [C_UNW, C_ENT]):
        body.set_facecolor(col)
        body.set_alpha(0.75)
        body.set_edgecolor('k')
        body.set_linewidth(0.5)
    for key in ('cmedians', 'cbars', 'cmins', 'cmaxes'):
        if key in vp:
            vp[key].set_color('k')
            vp[key].set_linewidth(1)
    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels(['unweighted', 'entropy-weighted'])
    ax[0].set_ylabel('posterior predictive variance')
    ax[0].set_yscale('log')
    ax[0].set_title('(a) Predictive uncertainty', loc='left', fontweight='bold')

    du, de = results['unweighted']['draws'], results['entropy']['draws']
    names = ['intercept'] + [f'$\\beta_{{{i + 1}}}$' for i in range(du.shape[1] - 1)]
    ypos = np.arange(len(names))
    ax[1].errorbar(du.mean(0), ypos - 0.16, xerr=1.96 * du.std(0), fmt='o',
                   ms=3.6, lw=1.1, color=C_UNW, label='unweighted')
    ax[1].errorbar(de.mean(0), ypos + 0.16, xerr=1.96 * de.std(0), fmt='s',
                   ms=3.6, lw=1.1, color=C_ENT, label='entropy-weighted')
    ax[1].axvline(0, color='k', lw=0.9, ls='--')
    ax[1].set_yticks(ypos)
    ax[1].set_yticklabels(names, fontsize=8)
    ax[1].set_xlabel('posterior mean (95% credible interval)')
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title('(b) Posterior coefficients', loc='left', fontweight='bold')
    fig.tight_layout()
    return fig, results
