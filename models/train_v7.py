"""
v7 모델 학습 — accuracy 직접 최적화 + threshold 튜닝 + CatBoost 옵션.

v65 대비:
- depth 9→5, min_child_weight 15, reg_lambda 2.0
- eval_metric: logloss→error
- GroupKFold (시즌별) CV 옵션
- Threshold 튜닝: 0.45~0.55 최적 탐색
- CatBoost 병행 실험 (설치 시)
"""
import json
import logging
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GroupKFold
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "saved"
MODEL_DIR.mkdir(exist_ok=True)

XGB_PARAMS_V7 = {
    "max_depth": 5,
    "learning_rate": 0.02,
    "n_estimators": 800,
    "subsample": 0.7,
    "colsample_bytree": 0.8,
    "min_child_weight": 15,
    "reg_lambda": 2.0,
    "reg_alpha": 0.5,
    "gamma": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "error",
    "random_state": 42,
    "verbosity": 0,
}


def _find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """0.45~0.55 범위에서 accuracy 최대화 threshold 탐색."""
    best_threshold = 0.50
    best_acc = 0.0
    for t in np.arange(0.45, 0.551, 0.005):
        preds = (y_prob >= t).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_threshold = t
    return round(best_threshold, 3), round(best_acc, 4)


def train_model_v7(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    n_splits: int = 3,
    save_name: str = "xgb_v7",
) -> tuple[xgb.XGBClassifier, dict]:
    """v7 XGBoost 학습 + CV.

    Args:
        X: 피처 배열
        y: 레이블
        groups: 시즌별 그룹 (GroupKFold용). None이면 TimeSeriesSplit.
        n_splits: CV fold 수
        save_name: 모델 저장명

    Returns:
        (학습된 모델, CV 결과)
    """
    if groups is not None:
        cv = GroupKFold(n_splits=n_splits)
        splits = list(cv.split(X, y, groups))
    else:
        cv = TimeSeriesSplit(n_splits=n_splits)
        splits = list(cv.split(X))

    fold_results = []
    thresholds = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(**XGB_PARAMS_V7)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_prob = model.predict_proba(X_val)[:, 1]
        threshold, best_acc = _find_best_threshold(y_val, y_prob)
        thresholds.append(threshold)

        y_pred_default = (y_prob > 0.5).astype(int)
        acc_default = accuracy_score(y_val, y_pred_default)
        brier = brier_score_loss(y_val, y_prob)
        ll = log_loss(y_val, y_prob)

        fold_results.append({
            "fold": fold + 1,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "accuracy_default": round(acc_default, 4),
            "accuracy_tuned": round(best_acc, 4),
            "threshold": threshold,
            "brier_score": round(brier, 4),
            "log_loss": round(ll, 4),
            "best_iteration": model.best_iteration if hasattr(model, 'best_iteration') else XGB_PARAMS_V7["n_estimators"],
        })
        logger.info(
            "Fold %d: acc=%.4f (t=0.50), acc_tuned=%.4f (t=%.3f), logloss=%.4f",
            fold + 1, acc_default, best_acc, threshold, ll,
        )

    # 최적 threshold = CV fold 평균
    optimal_threshold = round(float(np.mean(thresholds)), 3)

    # 전체 데이터로 최종 모델
    avg_iterations = int(np.mean([r["best_iteration"] for r in fold_results]))
    final_params = {**XGB_PARAMS_V7, "n_estimators": max(avg_iterations, 100)}

    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(X, y, verbose=False)

    model_path = MODEL_DIR / f"{save_name}.json"
    final_model.save_model(str(model_path))
    logger.info("모델 저장: %s", model_path)

    cv_results = {
        "folds": fold_results,
        "mean_accuracy_default": round(np.mean([r["accuracy_default"] for r in fold_results]), 4),
        "mean_accuracy_tuned": round(np.mean([r["accuracy_tuned"] for r in fold_results]), 4),
        "optimal_threshold": optimal_threshold,
        "mean_brier": round(np.mean([r["brier_score"] for r in fold_results]), 4),
        "mean_logloss": round(np.mean([r["log_loss"] for r in fold_results]), 4),
        "final_n_estimators": final_params["n_estimators"],
        "model_path": str(model_path),
    }

    results_path = MODEL_DIR / f"{save_name}_cv_results.json"
    results_path.write_text(json.dumps(cv_results, indent=2, ensure_ascii=False))

    # Threshold 저장
    threshold_path = MODEL_DIR / f"{save_name}_threshold.json"
    threshold_path.write_text(json.dumps({"threshold": optimal_threshold}, indent=2))

    logger.info(
        "v7 CV: acc=%.4f (default), acc=%.4f (tuned, t=%.3f), logloss=%.4f",
        cv_results["mean_accuracy_default"],
        cv_results["mean_accuracy_tuned"],
        optimal_threshold,
        cv_results["mean_logloss"],
    )

    return final_model, cv_results


def train_catboost_v7(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    groups: Optional[np.ndarray] = None,
    n_splits: int = 3,
    save_name: str = "catboost_v7",
) -> tuple[object, dict]:
    """CatBoost 학습 (optional, catboost 설치 시)."""
    try:
        from catboost import CatBoostClassifier, Pool
    except ImportError:
        logger.warning("catboost 미설치 — 스킵")
        return None, {"error": "catboost not installed"}

    if groups is not None:
        cv = GroupKFold(n_splits=n_splits)
        splits = list(cv.split(X, y, groups))
    else:
        cv = TimeSeriesSplit(n_splits=n_splits)
        splits = list(cv.split(X))

    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = CatBoostClassifier(
            depth=5,
            learning_rate=0.03,
            iterations=800,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=0,
            eval_metric="Accuracy",
        )
        model.fit(
            Pool(X_train, y_train, feature_names=feature_names),
            eval_set=Pool(X_val, y_val, feature_names=feature_names),
            early_stopping_rounds=50,
        )

        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
        acc = accuracy_score(y_val, y_pred)
        threshold, best_acc = _find_best_threshold(y_val, y_prob)

        fold_results.append({
            "fold": fold + 1,
            "accuracy_default": round(acc, 4),
            "accuracy_tuned": round(best_acc, 4),
            "threshold": threshold,
        })
        logger.info("CatBoost Fold %d: acc=%.4f, tuned=%.4f (t=%.3f)", fold + 1, acc, best_acc, threshold)

    # Final model
    final_model = CatBoostClassifier(
        depth=5, learning_rate=0.03, iterations=800,
        l2_leaf_reg=3.0, random_seed=42, verbose=0,
    )
    final_model.fit(Pool(X, y, feature_names=feature_names))

    model_path = MODEL_DIR / f"{save_name}.cbm"
    final_model.save_model(str(model_path))

    cv_results = {
        "folds": fold_results,
        "mean_accuracy_default": round(np.mean([r["accuracy_default"] for r in fold_results]), 4),
        "mean_accuracy_tuned": round(np.mean([r["accuracy_tuned"] for r in fold_results]), 4),
        "model_path": str(model_path),
    }

    results_path = MODEL_DIR / f"{save_name}_cv_results.json"
    results_path.write_text(json.dumps(cv_results, indent=2, ensure_ascii=False))
    return final_model, cv_results


def load_model_v7(save_name: str = "xgb_v7") -> Optional[xgb.XGBClassifier]:
    model_path = MODEL_DIR / f"{save_name}.json"
    if not model_path.exists():
        logger.warning("모델 파일 없음: %s", model_path)
        return None
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    return model


def load_threshold_v7(save_name: str = "xgb_v7") -> float:
    threshold_path = MODEL_DIR / f"{save_name}_threshold.json"
    if threshold_path.exists():
        data = json.loads(threshold_path.read_text())
        return data.get("threshold", 0.5)
    return 0.5


if __name__ == "__main__":
    import argparse
    import pandas as pd

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser(description="v7 XGBoost 학습")
    parser.add_argument("--dataset", type=str, default="training_data_v7")
    parser.add_argument("--model-name", type=str, default="xgb_v7")
    parser.add_argument("--group-cv", action="store_true", help="시즌별 GroupKFold CV")
    parser.add_argument("--catboost", action="store_true", help="CatBoost 병행")
    args = parser.parse_args()

    DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
    parquet_path = DATA_DIR / f"{args.dataset}.parquet"
    csv_path = DATA_DIR / f"{args.dataset}.csv"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        logger.error("학습 데이터 없음: %s", args.dataset)
        raise SystemExit(1)

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y = df["label"].values

    groups = None
    if args.group_cv and "season" in df.columns:
        groups = df["season"].values

    logger.info("v7 학습: %d행 × %d피처", X.shape[0], X.shape[1])
    model, results = train_model_v7(X, y, groups=groups, save_name=args.model_name)
    logger.info("v7 완료: %s", results["model_path"])

    if args.catboost:
        train_catboost_v7(X, y, feature_names=feature_cols, groups=groups)
