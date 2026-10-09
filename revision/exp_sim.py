"""Simulation with a known sampling mechanism (Reviewer 3, major 13; Reviewer 4 summary).

Landscape  : the 10,195 background records (environmentally representative),
             split at random into a training half and a test half in every replicate.
Truth      : psi(x) = suitability from a logistic GAM fitted once to the real data
             (realistic, non-linear). Sites are generated in proportion to psi.
Survey     : s(x) = exp(-gamma * a(x)), a(x) = rank-scaled accessibility index built
             from cost-distance to streams and slope (higher = harder to reach).
             gamma = 0 is unbiased; larger gamma = stronger survey bias.
Recorded   : n_p = 800 presences drawn without replacement with prob ~ psi(x) s(x).
Background : (a) 'representative' - uniform from landscape (as in GSENM data)
             (b) 'survey-biased'  - drawn with prob ~ s(x) (inherits survey coverage)
Weights    : unweighted, entropy (manuscript), inverse frequency (uniform target),
             density ratio (background/presence), oracle 1/s(x) (true inclusion prob.)
Evaluation : on the test half - AUC of true (unbiased) presences, drawn ~ psi, against
             random landscape points; Spearman correlation of predictions with psi.
"""
from common import *
from scipy.stats import spearmanr, rankdata
import sys
df, X, y, groups = load()
Xs_all = MinMaxScaler().fit_transform(X)
pred_truth, _ = fit_gam(Xs_all, y, np.ones(len(y)))
L = Xs_all[y == 0]                                   # landscape
psi = pred_truth(L)
f = {n: i for i, n in enumerate(FEATURES)}
acc = rankdata(L[:, f['streams_cd']]) + rankdata(L[:, f['slope']])
acc = (acc - acc.min()) / (acc.max() - acc.min())    # 0 easy .. 1 hard
GAMMAS = [0.0, 1.5, 3.0, 4.5]
NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 20
rows = []
for rep in range(NREP):
    rng = np.random.default_rng(1000 + rep)
    idx = rng.permutation(len(L)); trI, teI = idx[: len(L) // 2], idx[len(L) // 2:]
    # independent truth test set
    te_p = rng.choice(teI, 400, replace=False, p=psi[teI] / psi[teI].sum())
    te_b = rng.choice(teI, 2000, replace=False)
    Xte = np.vstack([L[te_p], L[te_b]]); yte = np.r_[np.ones(400), np.zeros(2000)]
    for gamma in GAMMAS:
        s = np.exp(-gamma * acc)
        pr = psi[trI] * s[trI]
        pres = rng.choice(trI, 800, replace=False, p=pr / pr.sum())
        for bg in ['representative', 'survey-biased']:
            pb = np.ones(len(trI)) if bg == 'representative' else s[trI]
            back = rng.choice(trI, 4000, replace=False, p=pb / pb.sum())
            Xtr = np.vstack([L[pres], L[back]]); ytr = np.r_[np.ones(800), np.zeros(4000)]
            W = make_weights(Xtr, ytr, 100, ['unweighted', 'entropy', 'ipw_uniform', 'density_ratio'])
            wo = np.ones(len(ytr)); so = 1 / s[pres]; wo[:800] = so / so.mean()
            W['oracle_1/s'] = wo
            for learner in ['Random Forest', 'GAM']:
                for k, w in W.items():
                    if learner == 'Random Forest':
                        m = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                                   n_jobs=-1, random_state=SEED).fit(Xtr, ytr, sample_weight=w)
                        p = m.predict_proba(Xte)[:, 1]; pL = m.predict_proba(L[teI])[:, 1]
                    else:
                        pred, _ = fit_gam(Xtr, ytr, w); p = pred(Xte); pL = pred(L[teI])
                    rows.append(dict(rep=rep, gamma=gamma, background=bg, learner=learner,
                                     weighting=k, AUC_truth=roc_auc_score(yte, p),
                                     rho_psi=spearmanr(pL, psi[teI])[0],
                                     ESS_ratio=kish(w[:800]) / 800))
        print('rep', rep, 'gamma', gamma, flush=True)
    pd.DataFrame(rows).to_csv('results/E8_sim.csv', index=False)
print('FINISHED')
