# Entropy-Based Diagnosis of Sampling Bias in Archaeological Predictive Modeling

Code and derived data for the manuscript of the same name.

Quantifying sampling bias and correcting it are separate questions, and this
repository answers them separately. It computes the Shannon entropy of the
distribution of recorded archaeological sites across a partition of covariate
space, derives observation weights from that entropy in closed form, and tests
whether applying them changes what four standard predictive frameworks predict.

## Quick start

```bash
git clone https://github.com/mehmetsiddik/entropy-sampling-bias-archaeology.git
cd entropy-sampling-bias-archaeology
pip install -r requirements.txt
jupyter notebook Entropy_Sampling_Bias.ipynb
```

Run the notebook cells in order. Figures appear inline and are written to
`figures/` as PDF and PNG; tables and fold-level results are written to
`results/`. The full run takes roughly 25 minutes; the cross-validation and
sensitivity sections are the slow ones and print progress as they go.

## Layout

```
Entropy_Sampling_Bias.ipynb   main notebook: runs everything, displays every figure
requirements.txt
data/                         input dataset (see data/README.md)
src/
  entropy.py                  entropy diagnostic and the weighting function
  models.py                   the four predictive frameworks
  evaluation.py               cross-validation, sensitivity, out-of-fold prediction
  figures.py                  all nine figures
figures/                      generated, PDF and PNG
results/                      generated, CSV and NPZ
```

## Method

The covariate space is partitioned into `K` cells by k-means. Writing `p_i` for
the proportion of recorded sites falling in cell `i`, the Shannon entropy is

```
H(P) = -sum_i p_i * ln(p_i)
```

and each record in cell `i` receives the weight

```
w_i = -ln(p_i) / H(P)
```

The numerator is the surprisal of cell `i` and the denominator is its
expectation under `P`, which gives the weights three properties the notebook
verifies on the data:

- **unit mean**, so effective sample size and class balance are unchanged;
- **identity under uniform coverage** — if `p_i = 1/K` for all `i` then every
  weight is exactly 1, so no correction is applied where there is no imbalance;
- **strict positivity**, so no record is dropped from training.

Weights are applied to presence records; background records receive unit
weight. The partition and the weights are estimated inside each training fold
and then applied to the held-out fold, so nothing fitted on training data
reaches the evaluation.

Four frameworks are compared on identical folds: Random Forest, a
penalised-spline logistic GAM, MaxEnt in its penalised-logistic form, and a
non-spatial Bayesian logistic regression sampled by adaptive random-walk
Metropolis. Two cross-validation designs are reported: random ten-fold
stratified, and five-fold blocked by catchment.

## A note on what the data allow

The published dataset contains **no coordinates**. Archaeological site
locations in the United States are withheld under statutory protections against
looting, so the supplementary file released with the source study carries only
the presence indicator and the covariate values.

Two consequences run through the whole analysis:

1. No geographic grid, neighbourhood matrix, or distance-based spatial random
   effect can be constructed. All entropy quantities are defined over
   **covariate** space, and no spatial random effect is fitted. The catchment
   grouping used for blocked cross-validation is derived from the watershed-area
   field and is an approximation to a catchment identifier.
2. No covariate records survey effort, coverage, or accessibility. The
   diagnostic measures coverage imbalance but cannot attribute it: concentration
   of recorded sites in particular environments reflects both uneven fieldwork
   and genuine past preference, and these data cannot separate the two. The
   weighting is therefore presented as a diagnostic and a robustness device,
   not as a correction for survey effort.

## Reproducibility

A single seed (42) governs all partitioning, clustering and model fitting, so
results are reproducible exactly rather than only in distribution. The notebook
prints the package versions it ran under, together with a formatted sentence for
the manuscript, and ends with a table of every headline number so that the paper
and the code can be checked against each other.

## Licence

MIT.

## Citing

Add the article citation here once available. The dataset should be cited
separately as Yaworsky et al. (2020); see `data/README.md`.
