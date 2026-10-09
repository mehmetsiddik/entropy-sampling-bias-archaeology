"""Partition-evenness diagnostics (R4.2, R3.7), weight comparison and ESS (R4.1, R3.3),
Figure 3 variance/loadings check and replacement projection (R4.4)."""
from common import *
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
import json
df, X, y, groups = load()
Xs = MinMaxScaler().fit_transform(X)
res = {}
def J(lab, K):
    c = np.bincount(lab, minlength=K).astype(float); p = (c + 1e-9) / (c + 1e-9).sum()
    return float(-(p * np.log(p)).sum() / np.log(K))
def KL(l1, l0, K):
    c1 = np.bincount(l1, minlength=K) + 1e-9; c0 = np.bincount(l0, minlength=K) + 1e-9
    p1, p0 = c1 / c1.sum(), c0 / c0.sum(); return float((p1 * np.log(p1 / p0)).sum())
rows = []
for K in [25, 50, 100, 200, 400]:
    for name, fitset in [('pooled', Xs), ('presence_only', Xs[y == 1]), ('background_only', Xs[y == 0])]:
        km = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit(fitset)
        lab = km.predict(Xs)
        rows.append(dict(K=K, partition=name, J_presence=J(lab[y == 1], K),
                         J_background=J(lab[y == 0], K), KL=KL(lab[y == 1], lab[y == 0], K)))
    print('K', K, flush=True)
# fixed marginal-quantile grid (non-adaptive): median split on each covariate -> 2^10 cells
med = np.median(Xs[y == 0], axis=0)
code = ((Xs > med) * (2 ** np.arange(10))).sum(1).astype(int)
_, lab = np.unique(code, return_inverse=True); Kq = 1024
rows.append(dict(K=Kq, partition='median_grid_1024', J_presence=J(lab[y == 1], Kq),
                 J_background=J(lab[y == 0], Kq), KL=KL(lab[y == 1], lab[y == 0], Kq)))
# terciles on the 8 non-aspect covariates would be 6561 cells; use quartiles on 5 most variable? keep median grid
pd.DataFrame(rows).to_csv('results/E5_partition_evenness.csv', index=False)

# weights on full data at K=100 (as Reviewer 4 did)
W = make_weights(Xs, y, 100, ['entropy', 'ipw_uniform', 'density_ratio', 'density_ratio_bgpart'])
wr = []
km = KMeans(n_clusters=100, random_state=SEED, n_init=10).fit(Xs[y == 1])
for k, w in W.items():
    wp = w[y == 1]
    wr.append(dict(weighting=k, min=wp.min(), max=wp.max(), sd=wp.std(), mean=wp.mean(),
                   ESS=kish(wp), n=len(wp), ESS_ratio=kish(wp) / len(wp)))
pd.DataFrame(wr).to_csv('results/E5_weights.csv', index=False)
ce = np.array([W['entropy'][y == 1][km.labels_ == i][0] for i in range(100)])
cd = np.array([W['density_ratio'][y == 1][km.labels_ == i][0] for i in range(100)])
res['corr_entropy_dr_cells'] = float(np.corrcoef(ce, cd)[0, 1])
from scipy.stats import spearmanr
res['spearman_entropy_dr_cells'] = float(spearmanr(ce, cd)[0])

# Figure 3 check
pca = PCA(2).fit(Xs)
res['var_share'] = dict(zip(FEATURES, (Xs.var(0) / Xs.var(0).sum()).round(3).tolist()))
res['pca_evr_all'] = pca.explained_variance_ratio_.round(3).tolist()
res['pca_loadings_all'] = {f: pca.components_[:, i].round(3).tolist() for i, f in enumerate(FEATURES)}
non = [i for i, f in enumerate(FEATURES) if 'asp' not in f]
pca8 = PCA(2).fit(Xs[:, non]); res['pca_evr_noaspect'] = pca8.explained_variance_ratio_.round(3).tolist()
res['pca_loadings_noaspect'] = {FEATURES[i]: pca8.components_[:, j].round(3).tolist() for j, i in enumerate(non)}
asp = [i for i, f in enumerate(FEATURES) if 'asp' in f]
lab_full = KMeans(100, random_state=SEED, n_init=10).fit_predict(Xs[y == 1])
lab_asp = KMeans(100, random_state=SEED, n_init=10).fit_predict(Xs[y == 1][:, asp])
res['ARI_full_vs_aspect_only'] = float(adjusted_rand_score(lab_full, lab_asp))
Z = StandardScaler().fit_transform(X); pz = PCA(2).fit(Z)
res['pca_evr_zscore_all'] = pz.explained_variance_ratio_.round(3).tolist()
json.dump(res, open('results/E5_diag.json', 'w'), indent=1)
np.savez('results/fig3_data.npz', P=pca8.transform(Xs[:, non]), y=y, w=W['entropy'], lab=km.predict(Xs))
print(json.dumps(res, indent=1))
