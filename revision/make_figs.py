"""Regenerate every manuscript figure from the archived v1.1 results.
Place in revision/ and run from there with the pinned environment."""
import sys, json, warnings
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from src.entropy import load, coverage_profile, presence_weights, FEATURES
from src.models import fit_bayes
from src.evaluation import out_of_fold
from src import figures as F
F.use_style()
OUT = str(HERE.parent / 'figures') + '/'; R = str(HERE / 'results') + '/'
def save(fig, name):
    for ext in ('pdf', 'png'): fig.savefig(OUT + name + '.' + ext, dpi=300 if ext == 'png' else None)
df, X, y, g = load(); Xs = MinMaxScaler().fit_transform(X)
res = {}
cov = coverage_profile(Xs, y, [25, 50, 100, 200, 400]); save(F.fig_coverage(cov), 'fig1_coverage_entropy')
save(F.fig_weight_function(Xs, y, 100), 'fig2_weight_function')
e1 = pd.read_csv(R + 'E1.csv')
rnd, blk = e1[e1.scheme == 'stratified'], e1[e1.scheme == 'grouped']
save(F.fig_paired_auc(rnd), 'fig4_paired_auc')
save(F.fig_blocked_cv(rnd, blk), 'fig6_blocked_cv')
# Figure 5: resolution sensitivity, ten-fold, 300 trees, entropy and density-ratio weights
e7 = pd.read_csv(R + 'E7.csv')
gk = e7.groupby(['K', 'weighting'])[['AUC', 'TSS']].agg(['mean', 'std']).reset_index()
fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.1))
for metric, a in zip(['AUC', 'TSS'], ax):
    for tag, col, lab in (('unweighted', F.C_UNW, 'unweighted'), ('entropy', F.C_ENT, 'entropy'), ('density_ratio', F.C_THIRD, 'density ratio')):
        s = gk[gk.weighting == tag]
        a.errorbar(s['K'] * {'unweighted': 0.97, 'entropy': 1.0, 'density_ratio': 1.03}[tag], s[(metric, 'mean')], yerr=s[(metric, 'std')],
                   marker='o', ms=4, capsize=3, lw=1.4, color=col, label=lab)
    a.set_xscale('log'); a.set_xlabel('number of entropy cells $K$'); a.set_ylabel(f'held-out {metric}')
    a.legend(frameon=False, fontsize=8)
ax[0].set_title('(a) Discrimination', loc='left', fontweight='bold')
ax[1].set_title('(b) True skill statistic', loc='left', fontweight='bold')
fig.tight_layout(); save(fig, 'fig5_k_sensitivity')
pass
# Figures 7 and 8: out-of-fold Random Forest predictions and permutation importance
oof, imp = out_of_fold(X, y, K=100, n_splits=5, verbose=False)
save(F.fig_calibration(y, oof), 'fig7_calibration')
save(F.fig_importance(imp, FEATURES), 'fig8_feature_importance')
for t in ('unweighted', 'entropy'):
    frac, mp = calibration_curve(y, oof[t], n_bins=10, strategy='quantile')
    res['calib_' + t] = dict(brier=float(brier_score_loss(y, oof[t])), auc=float(roc_auc_score(y, oof[t])), maxdev=float(np.max(np.abs(frac - mp))))
    res['imp_' + t] = dict(zip(FEATURES, imp[t].mean(0).round(4).tolist()))
res['imp_sd_unweighted'] = dict(zip(FEATURES, imp['unweighted'].std(0).round(4).tolist()))
# Figure 9: Bayesian posterior
w, _, _, _ = presence_weights(Xs, y, 100)
fig, post = F.fig_posterior(Xs, y, w, fit_bayes); save(fig, 'fig9_bayesian_posterior')
res['post'] = {t: dict(acc=post[t]['acceptance'], medvar=float(np.median(post[t]['var']))) for t in post}
json.dump(res, open(OUT + 'fig_numbers.json', 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k != 'ksens'}, indent=1, default=float))
