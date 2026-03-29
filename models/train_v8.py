"""
v8 모델 학습 — v65+ 229피처 + 최적화 파라미터/threshold/sample_weight/deadzone.

v8_config.json에서 최적 파라미터를 읽어 학습.
config 없으면 v65+ 기본 파라미터로 fallback.
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
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# v65+ 기본 파라미터 (fallback)
DEFAULT_PARAMS = {
    "max_depth": 9, "learning_rate": 0.0093, "n_estimators": 588,
    "subsample": 0.54, "colsample_bytree": 0.30, "min_child_weight": 11,
    "reg_lambda": 1.26, "reg_alpha": 0.28, "gamma": 0.77,
    "objective": "binary:logistic", "eval_metric": "logloss",
    "random_state": 42, "verbosity": 0,
}


def load_v8_config() -> dict:
    """v8_config.json 로드. 없으면 기본값 반환."""
    config_path = MODEL_DIR / "v8_config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
            logger.info("v8 config 로드: %s", config_path)
            return config
        except Exception:
            logger.warning("v8_config.json 파싱 실패 — 기본값 사용")
    return {}


def _infer_year_labels(n_rows: int) -> np.ndarray:
    """스케줄 파일 기반 연도 레이블 추정."""
    year_counts = []
    for yr in [2023, 2024, 2025, 2026]:
        sched_file = DATA_DIR / f"schedules_{yr}.json"
        if not sched_file.exists():
            continue
        rows = json.loads(sched_file.read_text())
        completed = [
            g for g in rows
            if g.get("leagueType") == 10100
            and g.get("homeScore") is not None
            and g.get("awayScore") is not None
        ]
        if completed:
            year_counts.append((yr, len(completed)))

    total = sum(c for _, c in year_counts)
    labels = []
    for i, (yr, count) in enumerate(year_counts):
        if i < len(year_counts) - 1:
            n = round(n_rows * count / total)
        else:
            n = n_rows - len(labels)
        labels.extend([yr] * max(n, 0))

    while len(labels) < n_rows:
        labels.append(labels[-1] if labels else 2025)
    return np.array(labels[:n_rows])


def _build_sample_weights(n_rows: int, weight_map: Optional[dict]) -> Optional[np.ndarray]:
    """연도별 sample weight 배열 생성."""
    if weight_map is None:
        return None
    year_labels = _infer_year_labels(n_rows)
    return np.array([weight_map.get(str(int(yr)), weight_map.get(int(yr), 1.0))
                     for yr in year_labels], dtype=np.float64)


def train_v8_model(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    save_name: str = "xgb_v8",
) -> tuple[xgb.XGBClassifier, dict]:
    """v8 모델 학습 + CV 평가.

    v8_config.json에서 params, sample_weights, threshold, deadzone_push 로드.
    """
    config = load_v8_config()
    params = config.get("params", DEFAULT_PARAMS)
    weight_map = config.get("sample_weights")
    threshold = config.get("threshold", 0.5)
    deadzone_push = config.get("deadzone_push", 0.0)
    elo_diff_idx = config.get("elo_diff_feature_idx")

    # XGBoost params에서 non-xgb 키 제거
    xgb_params = {k: v for k, v in params.items()
                  if k in {"max_depth", "learning_rate", "n_estimators", "subsample",
                           "colsample_bytree", "min_child_weight", "reg_lambda", "reg_alpha",
                           "gamma", "objective", "eval_metric", "random_state", "verbosity"}}
    # 기본 설정 보장
    xgb_params.setdefault("objective", "binary:logistic")
    xgb_params.setdefault("eval_metric", "logloss")
    xgb_params.setdefault("random_state", 42)
    xgb_params.setdefault("verbosity", 0)

    sample_weights = _build_sample_weights(X.shape[0], weight_map)

    logger.info("v8 파라미터: %s", {k: v for k, v in xgb_params.items()
                                    if k not in ("objective", "eval_metric", "random_state", "verbosity")})
    logger.info("threshold=%.3f  deadzone_push=%.3f  sample_weights=%s",
                threshold, deadzone_push, "enabled" if sample_weights is not None else "disabled")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        fit_kwargs = {"eval_set": [(X_val, y_val)], "verbose": False}
        if sample_weights is not None:
            fit_kwargs["sample_weight"] = sample_weights[train_idx]

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_train, y_train, **fit_kwargs)

        y_pred_prob = model.predict_proba(X_val)[:, 1]

        # Dead zone push
        if deadzone_push > 0 and elo_diff_idx is not None and elo_diff_idx < X_val.shape[1]:
            elo_diff = X_val[:, elo_diff_idx]
            mask = (y_pred_prob >= 0.48) & (y_pred_prob <= 0.52)
            y_pred_prob[mask] += deadzone_push * np.sign(elo_diff[mask])
            y_pred_prob = np.clip(y_pred_prob, 0.001, 0.999)

        y_pred = (y_pred_prob > 0.5).astype(int)
        acc = accuracy_score(y_val, y_pred)
        brier = brier_score_loss(y_val, y_pred_prob)
        ll = log_loss(y_val, y_pred_prob)

        # Threshold tuned accuracy
        best_t, best_acc = threshold, accuracy_score(y_val, (y_pred_prob >= threshold).astype(int))
        for t in np.arange(0.45, 0.551, 0.005):
            a = accuracy_score(y_val, (y_pred_prob >= t).astype(int))
            if a > best_acc:
                best_acc = a
                best_t = round(float(t), 3)

        fold_results.append({
            "fold": fold + 1,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "accuracy": round(acc, 4),
            "accuracy_tuned": round(best_acc, 4),
            "threshold": best_t,
            "brier_score": round(brier, 4),
            "log_loss": round(ll, 4),
            "best_iteration": model.best_iteration if hasattr(model, 'best_iteration') else xgb_params.get("n_estimators", 588),
        })
        logger.info(
            "Fold %d: acc=%.4f (tuned=%.4f, t=%.3f), brier=%.4f, logloss=%.4f",
            fold + 1, acc, best_acc, best_t, brier, ll,
        )

    # ── 전체 데이터로 최종 모델 학습 ──
    avg_iterations = int(np.mean([r["best_iteration"] for r in fold_results]))
    final_params = {**xgb_params, "n_estimators": max(avg_iterations, 50)}

    final_model = xgb.XGBClassifier(**final_params)
    fit_kwargs = {"verbose": False}
    if sample_weights is not None:
        fit_kwargs["sample_weight"] = sample_weights
    final_model.fit(X, y, **fit_kwargs)

    model_path = MODEL_DIR / f"{save_name}.json"
    final_model.save_model(str(model_path))
    logger.info("모델 저장: %s", model_path)

    cv_results = {
        "version": "v8",
        "folds": fold_results,
        "mean_accuracy": round(float(np.mean([r["accuracy"] for r in fold_results])), 4),
        "mean_accuracy_tuned": round(float(np.mean([r["accuracy_tuned"] for r in fold_results])), 4),
        "mean_threshold": round(float(np.mean([r["threshold"] for r in fold_results])), 3),
        "mean_brier": round(float(np.mean([r["brier_score"] for r in fold_results])), 4),
        "mean_logloss": round(float(np.mean([r["log_loss"] for r in fold_results])), 4),
        "final_n_estimators": final_params["n_estimators"],
        "model_path": str(model_path),
        "config": {
            "params": xgb_params,
            "threshold": threshold,
            "deadzone_push": deadzone_push,
            "sample_weights": weight_map,
        },
    }

    results_path = MODEL_DIR / f"{save_name}_cv_results.json"
    results_path.write_text(json.dumps(cv_results, indent=2, ensure_ascii=False))

    logger.info(
        "v8 CV: acc=%.4f, tuned=%.4f (t=%.3f), brier=%.4f, logloss=%.4f",
        cv_results["mean_accuracy"], cv_results["mean_accuracy_tuned"],
        cv_results["mean_threshold"], cv_results["mean_brier"], cv_results["mean_logloss"],
    )

    return final_model, cv_results


if __name__ == "__main__":
    import argparse
    import pandas as pd

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser(description="v8 모델 학습")
    parser.add_argument("--dataset", type=str, default="training_data_v65plus_live",
                        help="데이터셋 이름 (기본: training_data_v65plus_live)")
    parser.add_argument("--model-name", type=str, default="xgb_v8",
                        help="모델 저장 이름")
    args = parser.parse_args()

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

    logger.info("학습 데이터: %d행 × %d피처", X.shape[0], X.shape[1])
    model, results = train_v8_model(X, y, save_name=args.model_name)
    logger.info("학습 완료. 모델: %s", results["model_path"])
