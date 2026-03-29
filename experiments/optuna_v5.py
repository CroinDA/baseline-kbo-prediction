"""v5 174-피처 Optuna 하이퍼파라미터 탐색.

Usage:
    python experiments/optuna_v5.py --trials 200
    python experiments/optuna_v5.py --trials 200 --dataset training_data_v5
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "models" / "saved"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_optuna(X, y, n_trials=200, n_splits=5):
    """Optuna TPE 최적화 — accuracy 극대화."""

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "subsample": trial.suggest_float("subsample", 0.4, 0.85),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.7),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 30),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 5.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
            "verbosity": 0,
        }

        tscv = TimeSeriesSplit(n_splits=n_splits)
        accs = []
        briers = []
        for train_idx, val_idx in tscv.split(X):
            model = xgb.XGBClassifier(**params)
            model.fit(
                X[train_idx], y[train_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                verbose=False,
            )
            proba = model.predict_proba(X[val_idx])[:, 1]
            preds = (proba > 0.5).astype(int)
            accs.append(accuracy_score(y[val_idx], preds))
            briers.append(brier_score_loss(y[val_idx], proba))

        trial.set_user_attr("mean_brier", float(np.mean(briers)))
        return np.mean(accs)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


def main():
    parser = argparse.ArgumentParser(description="v5 Optuna 하이퍼파라미터 탐색")
    parser.add_argument("--trials", type=int, default=200, help="탐색 횟수")
    parser.add_argument("--dataset", type=str, default="training_data_v5",
                        help="학습 데이터셋 이름")
    parser.add_argument("--splits", type=int, default=5, help="CV 분할 수")
    args = parser.parse_args()

    # 데이터 로드
    parquet_path = DATA_DIR / f"{args.dataset}.parquet"
    csv_path = DATA_DIR / f"{args.dataset}.csv"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        print(f"데이터 없음: {args.dataset} (.parquet/.csv)")
        sys.exit(1)

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y = df["label"].values

    print(f"[v5 Optuna] {X.shape[0]}행 × {X.shape[1]}피처, {args.trials} trials")

    study = run_optuna(X, y, n_trials=args.trials, n_splits=args.splits)

    best = study.best_trial
    print(f"\nBest accuracy: {best.value:.4f}")
    print(f"Best brier:    {best.user_attrs.get('mean_brier', 'N/A')}")
    print(f"\nBest params (models/train.py XGB_PARAMS에 복사):")
    print("XGB_PARAMS = {")
    for k, v in best.params.items():
        if isinstance(v, int):
            print(f'    "{k}": {v},')
        elif isinstance(v, float):
            print(f'    "{k}": {v:.4f},')
        else:
            print(f'    "{k}": {v!r},')
    print('    "objective": "binary:logistic",')
    print('    "eval_metric": "logloss",')
    print('    "random_state": 42,')
    print('    "verbosity": 0,')
    print("}")

    # Top 5 출력
    trials_sorted = sorted(study.trials, key=lambda t: t.value, reverse=True)[:5]
    print(f"\nTop 5 trials:")
    for t in trials_sorted:
        brier = t.user_attrs.get("mean_brier", 0)
        print(
            f"  Trial {t.number}: acc={t.value:.4f}  brier={brier:.4f}  "
            f"depth={t.params['max_depth']}  lr={t.params['learning_rate']:.4f}  "
            f"n={t.params['n_estimators']}"
        )

    # 결과 JSON 저장
    result = {
        "best_accuracy": round(best.value, 4),
        "best_brier": round(best.user_attrs.get("mean_brier", 0), 4),
        "best_params": best.params,
        "dataset": args.dataset,
        "n_features": X.shape[1],
        "n_samples": X.shape[0],
        "n_trials": args.trials,
        "top5": [
            {
                "trial": t.number,
                "accuracy": round(t.value, 4),
                "brier": round(t.user_attrs.get("mean_brier", 0), 4),
                "params": t.params,
            }
            for t in trials_sorted
        ],
    }
    out_path = RESULTS_DIR / "optuna_v5_results.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
