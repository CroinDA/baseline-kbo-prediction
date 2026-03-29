"""
v2 / v4 / v5 정확도 비교.

동일 조건:
  - XGBoost default params (공정 비교)
  - TimeSeriesSplit 5-fold
  - 메트릭: accuracy, brier score, log loss

사용법:
  python experiments/compare_versions.py
  python experiments/compare_versions.py --skip-build   # 이미 빌드된 데이터 사용
"""
import sys
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "saved"
MODEL_DIR.mkdir(exist_ok=True)

# 공정 비교용 XGBoost 기본 파라미터 (Optuna 튜닝 없이)
XGB_DEFAULT_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "verbosity": 0,
}


def build_all(years: list[int] = None):
    """3개 버전 데이터셋 빌드."""
    if years is None:
        years = [2023, 2024, 2025]

    logger.info("=" * 60)
    logger.info("v2 (159 피처) 데이터셋 빌드")
    logger.info("=" * 60)
    from pipeline.build_v2_individual import build_dataset_v2
    build_dataset_v2(years=years, output_name="training_data_v2_compare")

    logger.info("=" * 60)
    logger.info("v4 (37 피처) 데이터셋 빌드")
    logger.info("=" * 60)
    from pipeline.build_v4_aggregated import build_dataset_v2 as build_dataset_v4
    build_dataset_v4(years=years, output_name="training_data_v4_compare")

    logger.info("=" * 60)
    logger.info("v5 (221 피처) 데이터셋 빌드")
    logger.info("=" * 60)
    from pipeline.build_v5_expanded import build_dataset_v5
    build_dataset_v5(years=years, output_name="training_data_v5_compare")


def load_dataset(name: str) -> pd.DataFrame:
    """데이터셋 로드 (parquet 우선, csv 폴백)."""
    parquet_path = DATA_DIR / f"{name}.parquet"
    csv_path = DATA_DIR / f"{name}.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    elif csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"데이터셋 없음: {name} (.parquet/.csv)")


def evaluate_version(
    df: pd.DataFrame,
    version_name: str,
    n_splits: int = 5,
    save_model: bool = True,
) -> dict:
    """단일 버전 평가: TSCV 5-fold + 전체 학습."""
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y = df["label"].values

    logger.info("  데이터: %d행 × %d피처", X.shape[0], X.shape[1])

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(**XGB_DEFAULT_PARAMS)
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
        })

    mean_acc = np.mean([r["accuracy"] for r in fold_results])
    mean_brier = np.mean([r["brier_score"] for r in fold_results])
    mean_ll = np.mean([r["log_loss"] for r in fold_results])

    # 전체 데이터로 최종 모델 학습
    if save_model:
        final_model = xgb.XGBClassifier(**XGB_DEFAULT_PARAMS)
        final_model.fit(X, y, verbose=False)
        model_path = MODEL_DIR / f"xgb_{version_name}.json"
        final_model.save_model(str(model_path))
        logger.info("  모델 저장: %s", model_path)

    result = {
        "version": version_name,
        "n_features": X.shape[1],
        "n_samples": X.shape[0],
        "mean_accuracy": round(mean_acc, 4),
        "mean_brier": round(mean_brier, 4),
        "mean_logloss": round(mean_ll, 4),
        "folds": fold_results,
    }

    return result


def print_comparison_table(results: list[dict]):
    """비교 테이블 출력."""
    print()
    print("=" * 75)
    print("  v2 / v4 / v5 정확도 비교  (XGB default + TSCV 5-fold)")
    print("=" * 75)
    print(f"{'Version':<10} {'Features':>8} {'Samples':>8} {'Accuracy':>10} {'Brier':>10} {'LogLoss':>10}")
    print("-" * 75)

    for r in results:
        print(
            f"{r['version']:<10} {r['n_features']:>8} {r['n_samples']:>8} "
            f"{r['mean_accuracy']:>10.4f} {r['mean_brier']:>10.4f} {r['mean_logloss']:>10.4f}"
        )

    print("-" * 75)

    # 최고 성능 표시
    best_acc = max(results, key=lambda r: r["mean_accuracy"])
    best_brier = min(results, key=lambda r: r["mean_brier"])
    best_ll = min(results, key=lambda r: r["mean_logloss"])

    print(f"  Best Accuracy : {best_acc['version']} ({best_acc['mean_accuracy']:.4f})")
    print(f"  Best Brier    : {best_brier['version']} ({best_brier['mean_brier']:.4f})")
    print(f"  Best LogLoss  : {best_ll['version']} ({best_ll['mean_logloss']:.4f})")
    print("=" * 75)

    # Fold별 상세
    for r in results:
        print(f"\n  {r['version']} — Fold별 상세:")
        for f in r["folds"]:
            print(
                f"    Fold {f['fold']}: acc={f['accuracy']:.4f}, "
                f"brier={f['brier_score']:.4f}, ll={f['log_loss']:.4f} "
                f"(train={f['train_size']}, val={f['val_size']})"
            )
    print()


def main():
    parser = argparse.ArgumentParser(description="v2/v4/v5 정확도 비교")
    parser.add_argument("--skip-build", action="store_true",
                        help="데이터셋 빌드 스킵 (이미 빌드된 파일 사용)")
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # 1. 데이터셋 빌드
    if not args.skip_build:
        build_all(years=args.years)
    else:
        logger.info("데이터셋 빌드 스킵 (--skip-build)")

    # 2. 데이터셋 로드
    versions = [
        ("v2", "training_data_v2_compare"),
        ("v4", "training_data_v4_compare"),
        ("v5", "training_data_v5_compare"),
    ]

    # 빌드 스킵 시 기존 파일도 시도
    if args.skip_build:
        fallbacks = {
            "v2": ["training_data_v2_compare", "training_data_v2"],
            "v4": ["training_data_v4_compare", "training_data_v4"],
            "v5": ["training_data_v5_compare", "training_data_v5"],
        }
        resolved = []
        for vname, _ in versions:
            for fname in fallbacks[vname]:
                parquet = DATA_DIR / f"{fname}.parquet"
                csv = DATA_DIR / f"{fname}.csv"
                if parquet.exists() or csv.exists():
                    resolved.append((vname, fname))
                    break
            else:
                logger.warning("  %s 데이터셋 없음 — 스킵", vname)
        versions = resolved

    # 3. 평가
    results = []
    for version_name, dataset_name in versions:
        logger.info("")
        logger.info("━" * 40)
        logger.info("  평가: %s (%s)", version_name, dataset_name)
        logger.info("━" * 40)

        try:
            df = load_dataset(dataset_name)
        except FileNotFoundError as e:
            logger.warning("  %s — 스킵", e)
            continue

        result = evaluate_version(df, version_name)
        results.append(result)

    # 4. 결과 출력
    if results:
        print_comparison_table(results)

        # JSON 저장
        out_path = Path(__file__).resolve().parent / "compare_results.json"
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        logger.info("결과 저장: %s", out_path)


if __name__ == "__main__":
    main()
