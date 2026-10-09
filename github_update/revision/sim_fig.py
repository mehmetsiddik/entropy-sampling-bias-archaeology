import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
r = pd.read_csv('results/E8_sim.csv')
s = r.groupby(['learner','background','gamma','weighting']).agg(AUC=('AUC_truth','mean'), AUCsd=('AUC_truth','std'),
     rho=('rho_psi','mean'), rhosd=('rho_psi','std'), ESS=('ESS_ratio','mean')).reset_index()
s.round(4).to_csv('results/E8_sim_summary.csv', index=False)
lab = {'unweighted':'Unweighted','entropy':'Entropy (Eq. 3)','ipw_uniform':'Inverse frequency','density_ratio':'Density ratio','oracle_1/s':'Oracle 1/s(x)'}
col = {'unweighted':'#333333','entropy':'#c0392b','ipw_uniform':'#e67e22','density_ratio':'#2980b9','oracle_1/s':'#27ae60'}
fig, ax = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
for i, L in enumerate(['Random Forest','GAM']):
    for j, bg in enumerate(['representative','survey-biased']):
        a = ax[i, j]
        for k in lab:
            q = s[(s.learner==L)&(s.background==bg)&(s.weighting==k)]
            a.errorbar(q.gamma + 0.05*list(lab).index(k), q.rho, yerr=q.rhosd, marker='o', ms=4, capsize=2, color=col[k], label=lab[k])
        a.set_title(f'({"abcd"[2*i+j]}) {L}, {bg} background', loc='left', fontsize=10)
        a.set_ylabel(r'Spearman $\rho$ with true suitability'); a.grid(alpha=.3)
        if i == 1: a.set_xlabel(r'Survey-bias strength $\gamma$')
ax[0, 0].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig('results/figS_simulation.pdf'); plt.savefig('results/figS_simulation.png', dpi=150)
print(s[s.gamma.isin([0,4.5])].round(3).to_string())
