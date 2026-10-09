import pandas as pd, numpy as np, sys
from scipy.stats import wilcoxon
r = pd.read_csv(sys.argv[1])
keys = ['tag','scheme','n_splits','model']
out=[]
for key, g in r.groupby(keys):
    base = g[g.weighting=='unweighted'].sort_values(['rep','fold'])
    for k, h in g.groupby('weighting'):
        h = h.sort_values(['rep','fold'])
        d = h.AUC.values - base.AUC.values
        ci = 1.96*d.std(ddof=1)/np.sqrt(len(d)) if len(d)>1 else np.nan
        p = wilcoxon(h.AUC.values, base.AUC.values)[1] if k!='unweighted' and np.any(d!=0) else np.nan
        out.append(dict(zip(keys,key), weighting=k, n=len(h), AUC=h.AUC.mean(), AUC_sd=h.AUC.std(),
             dAUC=d.mean(), ci_lo=d.mean()-ci, ci_hi=d.mean()+ci, p=p,
             TSS=h.TSS.mean(), TSS_sd=h.TSS.std(), Kappa=h.Kappa.mean(), Kappa_sd=h.Kappa.std(),
             TSS_testcut=h.TSS_testcut.mean(), Kappa_testcut=h.Kappa_testcut.mean(),
             Brier=h.Brier.mean(), ESS=h.ESS_pres.mean()/h.n_pres.mean(), wmin=h.w_min.min(), wmax=h.w_max.max()))
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
print(pd.DataFrame(out).round(4).to_string())
