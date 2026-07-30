"""
Cross-validation, resolution sensitivity, and out-of-fold prediction.

Everything reported by these functions is computed on held-out folds. The
scaler, the entropy partition and the weights are all estimated inside the
training fold and then applied to the held-out fold, so nothing fitted on
training data reaches the evaluation.
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import wilcoxon

from .entropy import apply_weights, presence_weights, SEED
from .models import MODELS, N_TREES, tss_kappa


def make_folds(X, y, groups, scheme, n_splits):
    if scheme == 'stratified':
        return list(StratifiedKFold(n_splits, shuffle=True,
                                    random_state=SEED).split(X, y))
    if scheme == 'grouped':
        return list(GroupKFold(n_splits=n_splits).split(X, y, groups=groups))
    raise ValueError("scheme must be 'stratified' or 'grouped'")


def run_cv(X, y, groups, scheme='stratified', n_splits=10, K=100,
           models=None, verbose=True):
    """Fit every framework, weighted and unweighted, on every fold."""
    models = models or list(MODELS)
    rows = []
    for fold, (tr, te) in enumerate(make_folds(X, y, groups, scheme, n_splits)):
        sc = MinMaxScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        ytr, yte = y[tr], y[te]
        if len(np.unique(yte)) < 2:
            continue

        w, _, _, H = presence_weights(Xtr, ytr, K)

        for name in models:
            fitter = MODELS[name]
            for tag, weights in (('unweighted', np.ones(len(ytr))),
                                 ('entropy', w)):
                t0 = time.time()
                predict, _ = fitter(Xtr, ytr, weights)
                prob = predict(Xte)
                tss, kappa, _ = tss_kappa(yte, prob)
                rows.append(dict(scheme=scheme, fold=fold, model=name,
                                 weighting=tag,
                                 AUC=float(roc_auc_score(yte, prob)),
                                 TSS=tss, Kappa=kappa,
                                 Brier=float(brier_score_loss(yte, prob)),
                                 H=H, seconds=round(time.time() - t0, 1)))
                if verbose:
                    print(f'  fold {fold:>2}  {name:<14} {tag:<11} '
                          f'AUC={rows[-1]["AUC"]:.4f}  '
                          f'({rows[-1]["seconds"]:.0f}s)', flush=True)
    return pd.DataFrame(rows)


def k_sensitivity(X, y, K_grid, n_splits=5, n_estimators=200, verbose=True):
    """Random Forest AUC and TSS as a function of the partition resolution."""
    rows = []
    for K in K_grid:
        for fold, (tr, te) in enumerate(StratifiedKFold(
                n_splits, shuffle=True, random_state=SEED).split(X, y)):
            sc = MinMaxScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            ytr, yte = y[tr], y[te]
            w, _, w_cell, H = presence_weights(Xtr, ytr, K)
            for tag, weights in (('unweighted', np.ones(len(ytr))),
                                 ('entropy', w)):
                m = RandomForestClassifier(n_estimators=n_estimators,
                                           min_samples_leaf=2, n_jobs=-1,
                                           random_state=SEED)
                m.fit(Xtr, ytr, sample_weight=weights)
                prob = m.predict_proba(Xte)[:, 1]
                tss, _, _ = tss_kappa(yte, prob)
                rows.append(dict(K=K, fold=fold, weighting=tag, H=H,
                                 AUC=float(roc_auc_score(yte, prob)), TSS=tss,
                                 w_min=float(w_cell.min()),
                                 w_max=float(w_cell.max())))
        if verbose:
            print(f'  K={K} done', flush=True)
    return pd.DataFrame(rows)


def out_of_fold(X, y, K=100, n_splits=5, n_repeats=5, verbose=True):
    """Out-of-fold Random Forest predictions and held-out permutation importance."""
    oof = {'unweighted': np.zeros(len(y)), 'entropy': np.zeros(len(y))}
    imp = {'unweighted': [], 'entropy': []}

    for fold, (tr, te) in enumerate(StratifiedKFold(
            n_splits, shuffle=True, random_state=SEED).split(X, y)):
        sc = MinMaxScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        ytr, yte = y[tr], y[te]
        w, _, _, _ = presence_weights(Xtr, ytr, K)
        for tag, weights in (('unweighted', np.ones(len(ytr))), ('entropy', w)):
            m = RandomForestClassifier(n_estimators=N_TREES, min_samples_leaf=2,
                                       n_jobs=-1, random_state=SEED)
            m.fit(Xtr, ytr, sample_weight=weights)
            oof[tag][te] = m.predict_proba(Xte)[:, 1]
            r = permutation_importance(m, Xte, yte, scoring='roc_auc',
                                       n_repeats=n_repeats, random_state=SEED,
                                       n_jobs=-1)
            imp[tag].append(r.importances_mean)
        if verbose:
            print(f'  fold {fold} done', flush=True)

    return oof, {k: np.asarray(v) for k, v in imp.items()}


def paired_summary(cv, metric='AUC'):
    """Paired difference, 95% CI and Wilcoxon p per model."""
    rows = []
    for model in cv.model.unique():
        a = cv[(cv.model == model) & (cv.weighting == 'unweighted')] \
            .sort_values('fold')[metric].to_numpy()
        b = cv[(cv.model == model) & (cv.weighting == 'entropy')] \
            .sort_values('fold')[metric].to_numpy()
        d = b - a
        ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        rows.append(dict(model=model, metric=metric,
                         unweighted=a.mean(), entropy=b.mean(),
                         difference=d.mean(),
                         ci_low=d.mean() - ci, ci_high=d.mean() + ci,
                         p_wilcoxon=float(wilcoxon(a, b)[1])))
    return pd.DataFrame(rows)


def summary_table(cv):
    """Mean +/- sd of every metric, by model and weighting."""
    g = cv.groupby(['model', 'weighting'])[['AUC', 'TSS', 'Kappa', 'Brier']]
    out = g.agg(['mean', 'std'])
    out.columns = [f'{a}_{b}' for a, b in out.columns]
    return out.round(4)
