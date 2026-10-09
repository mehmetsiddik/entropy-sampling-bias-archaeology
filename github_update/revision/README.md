# Revision analyses

Scripts for the analyses added in the revision. Place this folder in the repository root
(next to `src/`). `common.py` imports the original modules from `../src`.

Environment (needed to reproduce the archived grouped folds exactly):

    python3.11 -m venv venv && . venv/bin/activate
    pip install numpy==1.24.3 scipy==1.11.1 scikit-learn==1.3.0 pandas==2.0.3 matplotlib==3.7.2

Run from inside this folder (`cd revision`); about 3 h on 2 cores:

    mkdir -p results
    python exp_main.py E1   # four weightings x four models x both designs
    python exp_main.py E2   # wtrshd_size removed from predictors
    python exp_main.py E3   # z-scoring instead of min-max
    python exp_main.py E4   # 10 blocked folds; repeated random and blocked partitions (RF)
    python exp_main.py E7   # resolution sensitivity (RF)
    python exp_diag.py      # partition evenness, weight ranges / Kish ESS, PCA check
    python exp_sim.py 10    # simulation with a known survey mechanism
    python summ.py results/E1.csv ; python sim_fig.py ; python fig3_new.py

`results/` holds the CSV/JSON outputs used in the revised manuscript.
