"""Hybrid model experiments: v4 base + selected individual features from v2."""
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from pathlib import Path

optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_optuna_accuracy(X, y, n_trials=100, n_splits=5):
    """Optuna TPE optimization targeting accuracy."""
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.05, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "subsample": trial.suggest_float("subsample", 0.4, 0.8),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.8),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 30),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 5.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
            "verbosity": 0,
        }
        tscv = TimeSeriesSplit(n_splits=n_splits)
        accs = []
        for train_idx, val_idx in tscv.split(X):
            model = xgb.XGBClassifier(**params)
            model.fit(X[train_idx], y[train_idx],
                      eval_set=[(X[val_idx], y[val_idx])], verbose=False)
            preds = (model.predict_proba(X[val_idx])[:, 1] > 0.5).astype(int)
            accs.append(accuracy_score(y[val_idx], preds))
        return np.mean(accs)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study

def main():
    exp_name = sys.argv[1]  # v4.1, v4.2, v4.3
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    DATA = Path(__file__).resolve().parent.parent / "data" / "raw"
    v2 = pd.read_parquet(DATA / "training_data_v2.parquet")
    v4 = pd.read_parquet(DATA / "training_data_v4.parquet")

    v4_features = [c for c in v4.columns if c != "label"]
    y = v4["label"].values

    if exp_name == "v4.1":
        # v4 37 + b1~b3 wRC+ (6 cols)
        extra = [f"{side}_b{i}_wrcplus" for side in ["home","away"] for i in range(1,4)]
        desc = "v4 + b1-b3 wRC+"
    elif exp_name == "v4.2":
        # v4 37 + b1~b5 wRC+ (10 cols)
        extra = [f"{side}_b{i}_wrcplus" for side in ["home","away"] for i in range(1,6)]
        desc = "v4 + b1-b5 wRC+"
    elif exp_name == "v4.3":
        # v4 37 + all 9 batter wRC+ (18 cols)
        extra = [f"{side}_b{i}_wrcplus" for side in ["home","away"] for i in range(1,10)]
        desc = "v4 + all 9 batter wRC+"
    else:
        print(f"Unknown experiment: {exp_name}")
        return

    # Build hybrid dataset
    hybrid_df = pd.concat([v4[v4_features], v2[extra]], axis=1)
    X = hybrid_df.values

    print(f"[{exp_name}] {desc} — {X.shape[1]} features, {n_trials} trials")
    study = run_optuna_accuracy(X, y, n_trials=n_trials)

    best = study.best_trial
    print(f"\nBest accuracy: {best.value:.4f}")
    print(f"Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # Top 5
    trials_sorted = sorted(study.trials, key=lambda t: t.value, reverse=True)[:5]
    print(f"\nTop 5 trials:")
    for t in trials_sorted:
        print(f"  Trial {t.number}: acc={t.value:.4f}  depth={t.params['max_depth']}  lr={t.params['learning_rate']:.4f}  n={t.params['n_estimators']}")

if __name__ == "__main__":
    main()
