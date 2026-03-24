"""
월간 재학습 스크립트.

매월 1일 실행: 누적 데이터로 모델 재학습 + SHAP 분석.
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

from pipeline.build_dataset import build_dataset
from models.train import train_model
from features.builder import GameFeatures

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "saved"


def monthly_retrain(include_current_year: bool = True):
    """월간 재학습 실행.

    1. 기존 모델 백업
    2. 학습 데이터 재구축 (현재까지 누적 데이터)
    3. XGBoost 재학습
    4. SHAP 분석
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    now = datetime.now()
    logger.info("=" * 60)
    logger.info("월간 재학습 시작: %s", now.strftime("%Y-%m-%d"))

    # ── 1. 기존 모델 백업 ──
    model_path = MODEL_DIR / "xgb_model.json"
    if model_path.exists():
        backup_name = f"xgb_model_backup_{now:%Y%m%d}.json"
        backup_path = MODEL_DIR / backup_name
        shutil.copy2(model_path, backup_path)
        logger.info("기존 모델 백업: %s", backup_path)

    # ── 2. 학습 데이터 재구축 ──
    years = [2023, 2024, 2025]
    if include_current_year and now.year >= 2026:
        years.append(now.year)

    logger.info("학습 데이터 연도: %s", years)
    df = build_dataset(years=years, output_name="training_data")

    if df.empty:
        logger.error("데이터셋이 비어있음. 재학습 중단.")
        return

    # ── 3. XGBoost 재학습 ──
    feature_cols = GameFeatures.feature_names()
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)

    logger.info("학습 데이터: %d행 × %d피처", X.shape[0], X.shape[1])
    model, cv_results = train_model(X, y, n_splits=5, save_name="xgb_model")

    logger.info("CV 평균 정확도: %.2f%%", cv_results["mean_accuracy"] * 100)
    logger.info("CV 평균 Brier: %.4f", cv_results["mean_brier"])

    # ── 4. SHAP 분석 ──
    try:
        from analysis.shap_analysis import run_shap_analysis
        run_shap_analysis(save_plot=True)
    except Exception as e:
        logger.warning("SHAP 분석 실패: %s", e)

    logger.info("=" * 60)
    logger.info("월간 재학습 완료!")


if __name__ == "__main__":
    monthly_retrain()
