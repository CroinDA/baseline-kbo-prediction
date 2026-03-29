"""
v7 ablation study — v65 vs v7 변형 비교.

실험 항목:
1. v65 (219feat, control) vs v7 (47feat) vs v7+mirror (47feat×2)
2. Elo 피처 제거 ablation
3. Depth 4 vs 5 vs 6
4. Threshold 0.48~0.52
5. CatBoost vs XGBoost
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss

from models.train_v7 import XGB_PARAMS_V7, _find_best_threshold

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _cv_evaluate(X, y, params, n_splits=3, label=""):
    """TimeSeriesSplit CV로 accuracy/logloss 평가."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
        threshold, tuned_acc = _find_best_threshold(y_val, y_prob)

        results.append({
            "fold": fold + 1,
            "accuracy": accuracy_score(y_val, y_pred),
            "accuracy_tuned": tuned_acc,
            "threshold": threshold,
            "logloss": log_loss(y_val, y_prob),
            "val_size": len(val_idx),
        })

    mean_acc = np.mean([r["accuracy"] for r in results])
    mean_tuned = np.mean([r["accuracy_tuned"] for r in results])
    mean_ll = np.mean([r["logloss"] for r in results])
    avg_threshold = np.mean([r["threshold"] for r in results])

    logger.info(
        "[%s] acc=%.4f, tuned=%.4f (t=%.3f), logloss=%.4f",
        label, mean_acc, mean_tuned, avg_threshold, mean_ll,
    )

    return {
        "label": label,
        "mean_accuracy": round(mean_acc, 4),
        "mean_accuracy_tuned": round(mean_tuned, 4),
        "mean_threshold": round(avg_threshold, 3),
        "mean_logloss": round(mean_ll, 4),
        "folds": results,
    }


def _load_df(name: str) -> pd.DataFrame | None:
    parquet = DATA_DIR / f"{name}.parquet"
    csv = DATA_DIR / f"{name}.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    logger.warning("데이터 없음: %s", name)
    return None


def run_ablation():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    all_results = []

    # ── 1. v65 control ──
    df_v65 = _load_df("training_data_v65_live")
    if df_v65 is None:
        df_v65 = _load_df("training_data_v65")
    if df_v65 is not None:
        feature_cols = [c for c in df_v65.columns if c != "label"]
        X_v65 = df_v65[feature_cols].values
        y_v65 = df_v65["label"].values
        # v65 기존 하이퍼파라미터
        params_v65 = {
            "max_depth": 9, "learning_rate": 0.0093, "n_estimators": 588,
            "subsample": 0.54, "colsample_bytree": 0.33, "min_child_weight": 11,
            "reg_lambda": 1.26, "reg_alpha": 0.28, "gamma": 0.77,
            "objective": "binary:logistic", "eval_metric": "logloss",
            "random_state": 42, "verbosity": 0,
        }
        result = _cv_evaluate(X_v65, y_v65, params_v65, label="v65-control")
        result["rows"] = len(df_v65)
        result["features"] = X_v65.shape[1]
        all_results.append(result)
    else:
        logger.warning("v65 데이터셋 없음 — control 스킵")

    # ── 2. v7 no mirror ──
    df_v7_no = _load_df("training_data_v7_nomirror")
    if df_v7_no is None:
        # Build without mirror
        try:
            from pipeline.build_v7_aggregated import build_dataset_v7
            df_v7_no = build_dataset_v7(
                years=[2023, 2024, 2025],
                output_name="training_data_v7_nomirror",
                augment_mirror=False,
            )
        except Exception as e:
            logger.warning("v7 no-mirror 빌드 실패: %s", e)

    if df_v7_no is not None:
        feature_cols = [c for c in df_v7_no.columns if c != "label"]
        X_v7n = df_v7_no[feature_cols].values
        y_v7n = df_v7_no["label"].values
        result = _cv_evaluate(X_v7n, y_v7n, XGB_PARAMS_V7, label="v7-no-mirror")
        result["rows"] = len(df_v7_no)
        result["features"] = X_v7n.shape[1]
        all_results.append(result)

    # ── 3. v7 with mirror ──
    df_v7 = _load_df("training_data_v7")
    if df_v7 is None:
        try:
            from pipeline.build_v7_aggregated import build_dataset_v7
            df_v7 = build_dataset_v7(
                years=[2023, 2024, 2025],
                output_name="training_data_v7",
                augment_mirror=True,
            )
        except Exception as e:
            logger.warning("v7 mirror 빌드 실패: %s", e)

    if df_v7 is not None:
        feature_cols = [c for c in df_v7.columns if c != "label"]
        X_v7 = df_v7[feature_cols].values
        y_v7 = df_v7["label"].values
        result = _cv_evaluate(X_v7, y_v7, XGB_PARAMS_V7, label="v7-mirror")
        result["rows"] = len(df_v7)
        result["features"] = X_v7.shape[1]
        all_results.append(result)

        # ── 4. Elo 제거 ablation ──
        elo_cols = ["elo_diff", "elo_win_prob", "elo_sp_diff"]
        non_elo_cols = [c for c in feature_cols if c not in elo_cols]
        if len(non_elo_cols) < len(feature_cols):
            X_no_elo = df_v7[non_elo_cols].values
            result = _cv_evaluate(X_no_elo, y_v7, XGB_PARAMS_V7, label="v7-mirror-no-elo")
            result["rows"] = len(df_v7)
            result["features"] = X_no_elo.shape[1]
            all_results.append(result)

        # ── 5. Depth ablation ──
        for depth in [4, 6]:
            params_d = {**XGB_PARAMS_V7, "max_depth": depth}
            result = _cv_evaluate(X_v7, y_v7, params_d, label=f"v7-mirror-depth{depth}")
            result["rows"] = len(df_v7)
            result["features"] = X_v7.shape[1]
            all_results.append(result)

        # ── 6. CatBoost ──
        try:
            from catboost import CatBoostClassifier, Pool
            tscv = TimeSeriesSplit(n_splits=3)
            cb_folds = []
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_v7)):
                cb = CatBoostClassifier(
                    depth=5, learning_rate=0.03, iterations=800,
                    l2_leaf_reg=3.0, random_seed=42, verbose=0,
                )
                cb.fit(
                    Pool(X_v7[train_idx], y_v7[train_idx], feature_names=feature_cols),
                    eval_set=Pool(X_v7[val_idx], y_v7[val_idx], feature_names=feature_cols),
                    early_stopping_rounds=50,
                )
                y_prob = cb.predict_proba(X_v7[val_idx])[:, 1]
                acc = accuracy_score(y_v7[val_idx], (y_prob > 0.5).astype(int))
                t, tuned_acc = _find_best_threshold(y_v7[val_idx], y_prob)
                cb_folds.append({"fold": fold + 1, "accuracy": acc, "accuracy_tuned": tuned_acc, "threshold": t})
            cb_result = {
                "label": "v7-mirror-catboost",
                "mean_accuracy": round(np.mean([f["accuracy"] for f in cb_folds]), 4),
                "mean_accuracy_tuned": round(np.mean([f["accuracy_tuned"] for f in cb_folds]), 4),
                "mean_threshold": round(np.mean([f["threshold"] for f in cb_folds]), 3),
                "rows": len(df_v7),
                "features": X_v7.shape[1],
                "folds": cb_folds,
            }
            all_results.append(cb_result)
            logger.info("[catboost] acc=%.4f, tuned=%.4f", cb_result["mean_accuracy"], cb_result["mean_accuracy_tuned"])
        except ImportError:
            logger.info("CatBoost 미설치 — 스킵")

    # ── 결과 저장 ──
    summary = []
    for r in all_results:
        summary.append({
            "label": r["label"],
            "rows": r.get("rows"),
            "features": r.get("features"),
            "accuracy": r["mean_accuracy"],
            "accuracy_tuned": r.get("mean_accuracy_tuned"),
            "threshold": r.get("mean_threshold"),
            "logloss": r.get("mean_logloss"),
        })

    out_path = RESULTS_DIR / "v7_ablation_results.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info("ablation 결과 저장: %s", out_path)

    # 표 출력
    print("\n" + "=" * 80)
    print(f"{'Variant':<25} {'Rows':>6} {'Feat':>5} {'Acc':>7} {'Tuned':>7} {'Thr':>6} {'LL':>7}")
    print("-" * 80)
    for s in summary:
        print(
            f"{s['label']:<25} {s.get('rows', '?'):>6} {s.get('features', '?'):>5} "
            f"{s['accuracy']:>7.4f} {s.get('accuracy_tuned', 0):>7.4f} "
            f"{s.get('threshold', '-'):>6} {s.get('logloss', '-'):>7}"
        )
    print("=" * 80)

    return summary


if __name__ == "__main__":
    run_ablation()
