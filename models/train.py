"""
XGBoost 모델 학습 — 시계열 분할 교차검증.

보수적 하이퍼파라미터로 과적합 방지.
시간 기반 분할(Time-Series Split)만 사용.
"""
import json
import logging
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "saved"
MODEL_DIR.mkdir(exist_ok=True)

# ── 하이퍼파라미터 (v2 159피처 Optuna 200t accuracy 최적화) ──
XGB_PARAMS = {
    "max_depth": 9,
    "learning_rate": 0.0093,
    "n_estimators": 588,
    "subsample": 0.54,
    "colsample_bytree": 0.33,
    "min_child_weight": 11,
    "reg_lambda": 1.26,
    "reg_alpha": 0.28,
    "gamma": 0.77,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "verbosity": 0,
}


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    early_stopping_rounds: int = 30,
    save_name: str = "xgb_v5",
) -> tuple[xgb.XGBClassifier, dict]:
    """XGBoost 학습 + 시계열 교차검증.

    Args:
        X: 피처 배열 (N, 10)
        y: 타겟 (홈팀 승리=1, 패배=0)
        n_splits: 시계열 분할 수
        early_stopping_rounds: 조기 종료 라운드
        save_name: 모델 저장 파일명

    Returns:
        (학습된 모델, 검증 결과 딕셔너리)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_pred_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_prob > 0.5).astype(int)

        acc = accuracy_score(y_val, y_pred)
        brier = brier_score_loss(y_val, y_pred_prob)
        ll = log_loss(y_val, y_pred_prob)

        fold_results.append({
            "fold": fold + 1,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "accuracy": round(acc, 4),
            "brier_score": round(brier, 4),
            "log_loss": round(ll, 4),
            "best_iteration": model.best_iteration if hasattr(model, 'best_iteration') else XGB_PARAMS["n_estimators"],
        })
        logger.info(
            "Fold %d: acc=%.4f, brier=%.4f, logloss=%.4f (train=%d, val=%d)",
            fold + 1, acc, brier, ll, len(train_idx), len(val_idx),
        )

    # ── 전체 데이터로 최종 모델 학습 ──
    # 평균 best_iteration 사용
    avg_iterations = int(np.mean([r["best_iteration"] for r in fold_results]))
    final_params = {**XGB_PARAMS, "n_estimators": max(avg_iterations, 50)}

    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(X, y, verbose=False)

    # 저장
    model_path = MODEL_DIR / f"{save_name}.json"
    final_model.save_model(str(model_path))
    logger.info("모델 저장: %s", model_path)

    # 검증 결과 요약
    cv_results = {
        "folds": fold_results,
        "mean_accuracy": round(np.mean([r["accuracy"] for r in fold_results]), 4),
        "mean_brier": round(np.mean([r["brier_score"] for r in fold_results]), 4),
        "mean_logloss": round(np.mean([r["log_loss"] for r in fold_results]), 4),
        "final_n_estimators": final_params["n_estimators"],
        "model_path": str(model_path),
    }

    # 결과 저장
    results_path = MODEL_DIR / f"{save_name}_cv_results.json"
    results_path.write_text(json.dumps(cv_results, indent=2, ensure_ascii=False))

    logger.info(
        "CV 평균: acc=%.4f, brier=%.4f, logloss=%.4f",
        cv_results["mean_accuracy"],
        cv_results["mean_brier"],
        cv_results["mean_logloss"],
    )

    return final_model, cv_results


def load_model(save_name: str = "xgb_v5") -> Optional[xgb.XGBClassifier]:
    """저장된 모델 로드."""
    model_path = MODEL_DIR / f"{save_name}.json"
    if not model_path.exists():
        logger.warning("모델 파일 없음: %s", model_path)
        return None
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    return model


if __name__ == "__main__":
    import argparse
    import pandas as pd

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser(description="XGBoost 모델 학습")
    parser.add_argument("--dataset", type=str, default="training_data_v2",
                        help="데이터셋 이름 (기본: training_data_v2)")
    parser.add_argument("--model-name", type=str, default=None,
                        help="모델 저장 이름 (기본: 데이터셋 이름에서 자동)")
    args = parser.parse_args()

    DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
    parquet_path = DATA_DIR / f"{args.dataset}.parquet"
    csv_path = DATA_DIR / f"{args.dataset}.csv"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        logger.error("학습 데이터 없음: %s (.parquet/.csv)", args.dataset)
        raise SystemExit(1)

    # 피처 컬럼 = label 빼고 전부
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y = df["label"].values

    save_name = args.model_name or args.dataset.replace("training_data", "xgb")
    logger.info("학습 데이터: %d행 × %d피처 (dataset=%s)", X.shape[0], X.shape[1], args.dataset)
    model, results = train_model(X, y, save_name=save_name)
    logger.info("학습 완료. 모델 저장: %s", results["model_path"])
