"""
월간 재학습 스크립트.

매월 1일 실행: 누적 데이터로 v1/v2/v4/v5 모델 재학습 + SHAP 분석.
기존 모델을 백업하고 새 모델로 교체.
"""
import sys
import shutil
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.train import train_model
from models.train_v8 import train_v8_model
from pipeline.build_v65_plus import build_dataset_v65_plus

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "saved"

# 버전별 빌드 함수 매핑
VERSIONS = [
    {
        "name": "v65",
        "module_path": "pipeline.build_v6_timeaware",
        "func_name": "build_dataset_v6",
        "dataset_name": "training_data_v65",
        "model_name": "xgb_v65",
        "kwargs": {
            "include_bench": False,
            "include_sp_war": True,
            "stabilize_rates": True,
            "tune_bullpen": True,
            "use_spring": False,
        },
    },
    {
        "name": "v64",
        "module_path": "pipeline.build_v6_timeaware",
        "func_name": "build_dataset_v6",
        "dataset_name": "training_data_v64",
        "model_name": "xgb_v64",
        "kwargs": {
            "include_bench": False,
            "include_sp_war": True,
            "stabilize_rates": True,
            "tune_bullpen": True,
            "use_spring": False,
        },
    },
]


def monthly_retrain(include_current_year: bool = True):
    """월간 재학습 실행.

    1. 기존 모델 백업
    2. 버전별 학습 데이터 재구축 (현재까지 누적 데이터)
    3. XGBoost 재학습
    4. SHAP 분석
    """
    import importlib

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    now = datetime.now()
    logger.info("=" * 60)
    logger.info("월간 재학습 시작: %s", now.strftime("%Y-%m-%d"))

    # ── 1. 기존 모델 백업 ──
    for version in VERSIONS:
        model_name = version["model_name"]
        model_path = MODEL_DIR / f"{model_name}.json"
        if model_path.exists():
            backup_name = f"{model_name}_backup_{now:%Y%m%d}.json"
            backup_path = MODEL_DIR / backup_name
            shutil.copy2(model_path, backup_path)
            logger.info("백업: %s → %s", model_name, backup_name)

    # ── 2. 학습 데이터 연도 결정 ──
    years = [2023, 2024, 2025]
    if include_current_year and now.year >= 2026:
        years.append(now.year)
    logger.info("학습 데이터 연도: %s", years)

    # ── 3. 버전별 재학습 ──
    for version in VERSIONS:
        ver_name = version["name"]
        logger.info("\n━━ %s 재학습 ━━", ver_name)

        mod = importlib.import_module(version["module_path"])
        build_func = getattr(mod, version["func_name"])
        df = build_func(years=years, output_name=version["dataset_name"], **version.get("kwargs", {}))

        if df.empty:
            logger.error("  %s 데이터셋 비어있음 — 스킵", ver_name)
            continue

        feature_cols = [c for c in df.columns if c != "label"]
        X = df[feature_cols].values.astype(np.float32)
        y = df["label"].values.astype(np.int32)

        logger.info("  %s: %d행 × %d피처", ver_name, X.shape[0], X.shape[1])
        model, cv_results = train_model(X, y, n_splits=5, save_name=version["model_name"])
        logger.info("  %s CV 정확도: %.2f%%", ver_name, cv_results["mean_accuracy"] * 100)

    # ── 3.5. v8 재학습 (v65+ 229피처 + 최적화 파라미터) ──
    logger.info("\n━━ v8 재학습 ━━")
    try:
        df_v8 = build_dataset_v65_plus(years=years, output_name="training_data_v65plus")
        if not df_v8.empty:
            feat_cols_v8 = [c for c in df_v8.columns if c != "label"]
            X_v8 = df_v8[feat_cols_v8].values.astype(np.float32)
            y_v8 = df_v8["label"].values.astype(np.int32)
            logger.info("  v8: %d행 × %d피처", X_v8.shape[0], X_v8.shape[1])
            _, cv_v8 = train_v8_model(X_v8, y_v8, save_name="xgb_v8")
            logger.info("  v8 CV: acc=%.4f, tuned=%.4f", cv_v8["mean_accuracy"], cv_v8["mean_accuracy_tuned"])
        else:
            logger.error("  v8 데이터셋 비어있음 — 스킵")
    except Exception as e:
        logger.exception("  v8 재학습 실패: %s", e)

    # ── 4. SHAP 분석 (v5 기준) ──
    try:
        from analysis.shap_analysis import run_shap_analysis
        run_shap_analysis(save_plot=True)
    except Exception as e:
        logger.warning("SHAP 분석 실패: %s", e)

    logger.info("=" * 60)
    logger.info("월간 재학습 완료!")


if __name__ == "__main__":
    monthly_retrain()
