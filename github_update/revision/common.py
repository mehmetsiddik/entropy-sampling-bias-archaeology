"""Shared helpers for the revision analyses. Imports the authors' own modules
from the public repository so that models, folds, seed and partition are
identical to the manuscript pipeline."""
import sys, warnings
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.entropy import load, FEATURES, SEED
from src.models import fit_gam, fit_maxent, fit_bayes, N_TREES

def fit_rf_oob(X, y, w):
    m = RandomForestClassifier(n_estimators=N_TREES, min_samples_leaf=2, n_jobs=-1,
                               random_state=SEED, oob_score=True)
    m.fit(X, y, sample_weight=w)
    oob = m.oob_decision_function_[:, 1]
    oob = np.nan_to_num(oob, nan=float(np.mean(y)))
    return (lambda Z: m.predict_proba(Z)[:, 1]), oob

def fit_generic(f):
    def g(X, y, w):
        pred, _ = f(X, y, w)
        return pred, pred(X)          # training-fold fitted probabilities
    return g

MODELS = {'Random Forest': fit_rf_oob, 'MaxEnt': fit_generic(fit_maxent),
          'GAM': fit_generic(fit_gam), 'Bayesian': fit_generic(fit_bayes)}

def youden_cut(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return thr[int(np.argmax(tpr - fpr))]

def tss_kappa_at(y, p, cut):
    pred = (p >= cut).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    tss = tp / (tp + fn) + tn / (tn + fp) - 1
    return float(tss), float(cohen_kappa_score(y, pred))

# ------------------------------------------------------------------ weights
def _counts(labels, K):
    c = np.bincount(labels, minlength=K).astype(float)
    return np.where(c == 0, 0.5, c)          # same smoothing constant as src/entropy.py

def make_weights(Xtr, ytr, K=100, kinds=('entropy',), seed=SEED):
    """Return dict kind -> weight vector for the training fold (background = 1).
    entropy        : -ln p_i / H(P)                (manuscript, presence-only partition)
    ipw_uniform    : (1/K) / p_i                   (inverse frequency, uniform target)
    density_ratio  : q_i / p_i                     (background share / presence share,
                                                    same presence-only partition)
    density_ratio_bgpart : q_i / p_i on a partition fitted to the background only
    All presence-side weights are normalised to unit mean under P."""
    out = {}
    pres = Xtr[ytr == 1]; back = Xtr[ytr == 0]
    K_eff = min(K, max(2, len(pres) // 8))
    km = KMeans(n_clusters=K_eff, random_state=seed, n_init=10).fit(pres)
    lab_p = km.labels_; lab_b = km.predict(back)
    p = _counts(lab_p, K_eff); p /= p.sum()
    q = _counts(lab_b, K_eff); q /= q.sum()
    cell_w = {'entropy': -np.log(p) / float(np.sum(-p * np.log(p))),
              'ipw_uniform': (1.0 / K_eff) / p,
              'density_ratio': q / p}
    for k in kinds:
        if k == 'unweighted':
            out[k] = np.ones(len(ytr)); continue
        if k == 'density_ratio_bgpart':
            kmb = KMeans(n_clusters=K, random_state=seed, n_init=10).fit(back)
            lp = kmb.predict(pres); lb = kmb.labels_
            pp = _counts(lp, K); pp /= pp.sum(); qq = _counts(lb, K); qq /= qq.sum()
            cw, lab, pv = qq / pp, lp, pp
        else:
            cw, lab, pv = cell_w[k], lab_p, p
        w = np.ones(len(ytr)); w[ytr == 1] = cw[lab] / cw[lab].mean()   # unit mean under P
        out[k] = w
    return out

def kish(w):
    return float(w.sum() ** 2 / (w ** 2).sum())

# ------------------------------------------------------------------ folds
def folds(X, y, groups, scheme, n_splits, rep=0):
    if scheme == 'stratified':
        return list(StratifiedKFold(n_splits, shuffle=True, random_state=SEED + rep).split(X, y))
    if scheme == 'grouped':
        if rep == 0:      # identical to the manuscript (sklearn GroupKFold, deterministic)
            return list(GroupKFold(n_splits=n_splits).split(X, y, groups=groups))
        rng = np.random.default_rng(SEED + rep)       # random assignment of whole catchments
        ug = np.unique(groups); rng.shuffle(ug)
        fold_of = {g: i % n_splits for i, g in enumerate(ug)}
        f = np.array([fold_of[g] for g in groups])
        return [(np.where(f != k)[0], np.where(f == k)[0]) for k in range(n_splits)]
    raise ValueError(scheme)

def run(X, y, groups, scheme, n_splits, kinds, models, scaler='minmax', K=100,
        rep=0, tag='', verbose=True):
    rows = []
    for fold, (tr, te) in enumerate(folds(X, y, groups, scheme, n_splits, rep)):
        sc = (MinMaxScaler() if scaler == 'minmax' else StandardScaler()).fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te]); ytr, yte = y[tr], y[te]
        if len(np.unique(yte)) < 2: continue
        W = make_weights(Xtr, ytr, K, kinds)
        for m in models:
            for k in kinds:
                pred, ptr = MODELS[m](Xtr, ytr, W[k])
                p = pred(Xte)
                tss_te, kap_te = tss_kappa_at(yte, p, youden_cut(yte, p))   # original (leaky)
                tss_tr, kap_tr = tss_kappa_at(yte, p, youden_cut(ytr, ptr)) # corrected
                rows.append(dict(tag=tag, scheme=scheme, n_splits=n_splits, rep=rep, fold=fold,
                                 model=m, weighting=k, scaler=scaler, K=K,
                                 AUC=roc_auc_score(yte, p), Brier=brier_score_loss(yte, p),
                                 TSS_testcut=tss_te, Kappa_testcut=kap_te,
                                 TSS=tss_tr, Kappa=kap_tr,
                                 ESS_pres=kish(W[k][ytr == 1]), n_pres=int((ytr == 1).sum()),
                                 w_min=float(W[k][ytr == 1].min()), w_max=float(W[k][ytr == 1].max()),
                                 w_sd=float(W[k][ytr == 1].std())))
        if verbose:
            print(tag, scheme, n_splits, rep, 'fold', fold, 'done', flush=True)
    return pd.DataFrame(rows)
