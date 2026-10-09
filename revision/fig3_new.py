import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, json
d = np.load('results/fig3_data.npz'); P, y, w, lab = d['P'], d['y'], d['w'], d['lab']
ev = json.load(open('results/E5_diag.json'))['pca_evr_noaspect']
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
ax[0].scatter(P[y == 0, 0], P[y == 0, 1], s=3, c='0.7', alpha=.5, label='Background', rasterized=True)
ax[0].scatter(P[y == 1, 0], P[y == 1, 1], s=4, c='#c0392b', alpha=.6, label='Recorded sites', rasterized=True)
ax[0].legend(frameon=False, markerscale=3); ax[0].set_title('(a)', loc='left', fontweight='bold')
K = lab.max() + 1
cx = np.array([P[lab == k].mean(0) for k in range(K)])
share = np.array([(lab[y == 1] == k).mean() for k in range(K)])
wk = np.array([w[(y == 1) & (lab == k)].mean() if ((y == 1) & (lab == k)).any() else np.nan for k in range(K)])
sc = ax[1].scatter(cx[:, 0], cx[:, 1], s=4000 * share + 5, c=wk, cmap='coolwarm', norm=matplotlib.colors.TwoSlopeNorm(1.0, 0.8, 1.55), edgecolor='k', lw=.3)
plt.colorbar(sc, ax=ax[1], label='Entropy weight $w_i$'); ax[1].set_title('(b)', loc='left', fontweight='bold')
for a in ax:
    a.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)'); a.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)')
plt.tight_layout(); plt.savefig('results/fig3_entropy_cells_pca_noaspect.pdf'); plt.savefig('results/fig3_entropy_cells_pca_noaspect.png', dpi=150)
