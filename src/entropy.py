"""
Entropy diagnostic and observation weights over covariate space.

The weight applied to a record in cell i is

    p_i  = n_i / sum_k n_k          proportion of recorded sites in cell i
    H(P) = -sum_i p_i ln p_i        Shannon entropy of that distribution
    w_i  = -ln(p_i) / H(P)          normalised surprisal

Properties, all verifiable on the data:
  * E_P[w] = 1 exactly, so the total presence weight and class balance are unchanged
    (the Kish effective sample size is reduced slightly, to about 0.985 n at K = 100)
  * p_i = 1/K for all i  =>  w_i = 1 for all i, i.e. no correction when coverage
    is already uniform
  * w_i > 0 always, so no record is ever dropped from training
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

SEED = 42

FEATURES = [
    'east_west_asp',
    'GDD_corngrowing_dds_2005',
    'north_south_asp',
    'NPP_mean_00_15',
    'PRISM_tmean_30yr_normal_800mM2_annual_asc',
    'slope',
    'springs_cd',
    'streams_cd',
    'wetlands_cd',
    'wtrshd_size',
]

DATA = Path(__file__).resolve().parents[1] / 'data' / 'Yaworsky_etal_2020_sdmdata.csv'


def load(path=None):
    """Return (dataframe, X, y, groups). Raises if the file is not the expected one."""
    df = pd.read_csv(path or DATA, index_col=0)
    missing = [c for c in FEATURES + ['pa'] if c not in df.columns]
    if missing:
        raise ValueError(f'columns missing from the dataset: {missing}')
    X = df[FEATURES].to_numpy(dtype=float)
    y = df['pa'].to_numpy(dtype=int)
    groups = df['wtrshd_size'].to_numpy()
    if len(y) != 11814 or int((y == 1).sum()) != 1619:
        raise ValueError('dataset does not match the published record '
                         f'(got {len(y)} rows, {(y == 1).sum()} presences)')
    return df, X, y, groups


def scale(X_train, X_apply=None):
    """Min-max scaler fitted on X_train only."""
    sc = MinMaxScaler().fit(X_train)
    return sc.transform(X_train) if X_apply is None else (sc.transform(X_train),
                                                          sc.transform(X_apply))


def entropy_cells(X_cells, K, seed=SEED):
    """Partition X_cells into K cells and return the entropy weights.

    Returns
    -------
    km       : fitted KMeans, so the partition can be applied to held-out data
    p        : cell proportions
    H        : Shannon entropy of p
    w_cell   : weight for each cell, -ln(p) / H
    """
    K_eff = min(K, max(2, len(X_cells) // 8))
    km = KMeans(n_clusters=K_eff, random_state=seed, n_init=10).fit(X_cells)
    counts = np.bincount(km.labels_, minlength=K_eff).astype(float)
    counts = np.where(counts == 0, 0.5, counts)
    p = counts / counts.sum()
    surprisal = -np.log(p)
    H = float(np.sum(p * surprisal))
    return km, p, H, surprisal / H


def presence_weights(X_scaled, y, K, seed=SEED):
    """Weights for a training fold: presences reweighted, background left at 1."""
    km, p, H, w_cell = entropy_cells(X_scaled[y == 1], K, seed)
    w = np.ones(len(y))
    w[y == 1] = w_cell[km.labels_]
    return w, km, w_cell, H


def apply_weights(km, w_cell, X_scaled, y):
    """Apply a partition learned on training presences to another set."""
    w = np.ones(len(y))
    if (y == 1).any():
        w[y == 1] = w_cell[km.predict(X_scaled[y == 1])]
    return w


def coverage_profile(X_scaled, y, K_grid, seed=SEED):
    """Entropy, evenness and divergence of presences vs background, per resolution."""
    rows = []
    for K in K_grid:
        lab = KMeans(n_clusters=K, random_state=seed, n_init=10).fit_predict(X_scaled)
        c1 = np.bincount(lab[y == 1], minlength=K).astype(float)
        c0 = np.bincount(lab[y == 0], minlength=K).astype(float)
        p1 = (c1 + 1e-9) / (c1 + 1e-9).sum()
        p0 = (c0 + 1e-9) / (c0 + 1e-9).sum()
        h1 = float(-(p1 * np.log(p1)).sum())
        h0 = float(-(p0 * np.log(p0)).sum())
        rows.append(dict(K=K,
                         H_presence=h1,
                         H_background=h0,
                         J_presence=h1 / np.log(K),
                         J_background=h0 / np.log(K),
                         KL=float((p1 * np.log(p1 / p0)).sum())))
    return pd.DataFrame(rows)
