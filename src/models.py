"""
The four predictive frameworks, all genuinely fitted.

No simulated or placeholder values appear anywhere in this file. Every
function returns a callable that maps a design matrix to predicted
probabilities from a model that has been fitted to the data passed in.
"""
import numpy as np
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer

SEED = 42
N_TREES = 300


# --------------------------------------------------------------- Random Forest
def fit_rf(X, y, w, n_estimators=N_TREES):
    m = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=2,
                               n_jobs=-1, random_state=SEED)
    m.fit(X, y, sample_weight=w)
    return lambda Xte: m.predict_proba(Xte)[:, 1], m


# --------------------------------------------------------------- GAM
def fit_gam(X, y, w):
    """Additive penalised B-spline logistic model: one smooth per covariate,
    no interaction terms."""
    m = Pipeline([
        ('spl', SplineTransformer(n_knots=8, degree=3, include_bias=False)),
        ('lr', LogisticRegression(penalty='l2', C=1.0, max_iter=4000,
                                  solver='lbfgs')),
    ])
    m.fit(X, y, lr__sample_weight=w)
    return lambda Xte: m.predict_proba(Xte)[:, 1], m


# --------------------------------------------------------------- MaxEnt
def _maxent_features(X, knots):
    parts = [X, X ** 2]
    for j in range(X.shape[1]):
        xj = X[:, [j]]
        parts.append(np.clip(xj - knots[j][None, :], 0, None))
        parts.append(np.clip(knots[j][None, :] - xj, 0, None))
    return np.hstack(parts)


def fit_maxent(X, y, w):
    """MaxEnt in its penalised-logistic form (Renner & Warton 2013;
    Fithian & Hastie 2013; Phillips et al. 2017): presence against background
    over linear, quadratic and hinge features with an L1 penalty."""
    knots = [np.quantile(X[:, j], np.linspace(0.1, 0.9, 6))
             for j in range(X.shape[1])]
    F = _maxent_features(X, knots)
    mu, sd = F.mean(0), F.std(0) + 1e-9
    m = LogisticRegression(penalty='l1', C=0.5, solver='liblinear',
                           max_iter=4000, random_state=SEED)
    m.fit((F - mu) / sd, y, sample_weight=w)

    def predict(Xte):
        return m.predict_proba((_maxent_features(Xte, knots) - mu) / sd)[:, 1]

    return predict, m


# --------------------------------------------------------------- Bayesian
def fit_bayes(X, y, w, n_iter=12000, burn=4000, thin=8, tau2=1e2, seed=SEED):
    """Bayesian logistic regression with Gaussian priors, sampled by
    adaptive random-walk Metropolis.

    No spatial random effect: the published data carry no coordinates, so no
    neighbourhood structure can be constructed.
    """
    rng = np.random.default_rng(seed)
    Z = np.hstack([np.ones((X.shape[0], 1)), X])
    d = Z.shape[1]
    ww = w / w.mean()

    def logpost(b):
        eta = Z @ b
        return float(np.sum(ww * (y * eta - np.logaddexp(0.0, eta)))
                     - 0.5 * np.sum(b ** 2) / tau2)

    b = np.zeros(d)
    lp = logpost(b)
    step = 0.05 * np.ones(d)
    draws, accepted = [], 0

    for it in range(n_iter):
        prop = b + rng.normal(0, step)
        lp_new = logpost(prop)
        if np.log(rng.random()) < lp_new - lp:
            b, lp, accepted = prop, lp_new, accepted + 1
        if it < burn and (it + 1) % 500 == 0:
            step *= np.exp((accepted / (it + 1) - 0.234) * 2.0)
        if it >= burn and (it - burn) % thin == 0:
            draws.append(b.copy())

    draws = np.asarray(draws)
    b_mean = draws.mean(0)

    class Fit:
        """Exposes the posterior so the uncertainty figure can use it."""
        def __init__(self):
            self.draws = draws
            self.acceptance = accepted / n_iter

        def predict_draws(self, Xte):
            Ze = np.hstack([np.ones((Xte.shape[0], 1)), Xte])
            return expit(Ze @ draws.T)          # n_obs x n_draws

    def predict(Xte):
        Ze = np.hstack([np.ones((Xte.shape[0], 1)), Xte])
        return expit(Ze @ b_mean)

    return predict, Fit()


# --------------------------------------------------------------- registry
MODELS = {
    'Random Forest': fit_rf,
    'GAM': fit_gam,
    'MaxEnt': fit_maxent,
    'Bayesian': fit_bayes,
}


def tss_kappa(y_true, prob):
    """True skill statistic at the Youden-optimal threshold, and Cohen's kappa
    at the same threshold."""
    fpr, tpr, thr = roc_curve(y_true, prob)
    j = tpr - fpr
    cut = thr[int(np.argmax(j))]
    pred = (prob >= cut).astype(int)
    return float(np.max(j)), float(cohen_kappa_score(y_true, pred)), float(cut)
