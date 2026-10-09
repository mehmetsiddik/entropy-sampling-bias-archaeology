from common import *
import sys
df, X, y, groups = load()
ALL = ['Random Forest', 'MaxEnt', 'GAM', 'Bayesian']
which = sys.argv[1]
if which == 'E1':   # four weightings x four models x both designs (manuscript setting)
    kinds = ['unweighted', 'entropy', 'ipw_uniform', 'density_ratio', 'density_ratio_bgpart']
    r = pd.concat([run(X, y, groups, 'stratified', 10, kinds, ALL, tag='E1'),
                   run(X, y, groups, 'grouped', 5, kinds, ALL, tag='E1')])
elif which == 'E2':  # wtrshd_size removed from the predictors (still used for blocking)
    keep = [i for i, f in enumerate(FEATURES) if f != 'wtrshd_size']
    kinds = ['unweighted', 'entropy', 'density_ratio']
    r = pd.concat([run(X[:, keep], y, groups, 'stratified', 10, kinds, ALL, tag='E2_noWS'),
                   run(X[:, keep], y, groups, 'grouped', 5, kinds, ALL, tag='E2_noWS')])
elif which == 'E3':  # z-scoring instead of min-max
    kinds = ['unweighted', 'entropy', 'density_ratio']
    r = pd.concat([run(X, y, groups, 'stratified', 10, kinds, ALL, scaler='z', tag='E3_z'),
                   run(X, y, groups, 'grouped', 5, kinds, ALL, scaler='z', tag='E3_z')])
elif which == 'E4':  # fold number and repeated partitions (Random Forest)
    kinds = ['unweighted', 'entropy', 'density_ratio']
    parts = [run(X, y, groups, 'grouped', 10, kinds, ['Random Forest'], tag='E4_g10')]
    for rep in range(1, 6):
        parts.append(run(X, y, groups, 'grouped', 5, kinds, ['Random Forest'], rep=rep, tag='E4_grep'))
        parts.append(run(X, y, groups, 'stratified', 10, kinds, ['Random Forest'], rep=rep, tag='E4_srep'))
    r = pd.concat(parts)
elif which == 'E7':  # resolution sensitivity for entropy and density-ratio weights
    kinds = ['unweighted', 'entropy', 'density_ratio']
    r = pd.concat([run(X, y, groups, 'stratified', 10, kinds, ['Random Forest'], K=K, tag='E7')
                   for K in [25, 50, 100, 200, 400]])
r.to_csv(f'results/{which}.csv', index=False)
print('FINISHED', which)
