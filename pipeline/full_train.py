"""
전체 학습 오케스트레이터.

1단계: 과거 데이터 수집 (backfill) — API 키 필요
2단계: 학습 데이터셋 구축 (build_dataset) — Elo 시뮬레이션 + 피처 생성
3단계: XGBoost 학습 (train) — 시계열 CV + 모델 저장
4단계: 2026 시즌 Elo 초기화 — 2025 시즌 말 Elo에서 1/3 회귀

API 키 수령 후 1회 실행하면 모델 준비 완료.
"""
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.backfill import run_backfill
from pipeline.build_dataset import build_dataset
from models.train import train_model
from elo.engine import EloEngine
from features.builder import GameFeatures

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def full_train(
    years: list[int] = None,
    skip_backfill: bool = False,
):
    """전체 학습 파이프라인 실행.

    Args:
        years: 학습 데이터 연도 (기본: 2023~2025)
        skip_backfill: True면 데이터 수집 스킵 (이미 수집된 경우)
    """
    if years is None:
        years = [2023, 2024, 2025]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    logger.info("=" * 60)
    logger.info("전체 학습 파이프라인 시작")
    logger.info("=" * 60)

    # ── 1단계: 과거 데이터 수집 ──
    if not skip_backfill:
        logger.info("\n[1/4] 과거 데이터 수집 (backfill)")
        run_backfill(years)
    else:
        logger.info("\n[1/4] 과거 데이터 수집 — 스킵")

    # ── 2단계: 학습 데이터셋 구축 ──
    logger.info("\n[2/4] 학습 데이터셋 구축")
    df = build_dataset(years=years, output_name="training_data")

    if df.empty:
        logger.error("데이터셋이 비어있음. 백필 데이터를 확인하세요.")
        return

    # ── 3단계: XGBoost 학습 ──
    logger.info("\n[3/4] XGBoost 모델 학습")

    feature_cols = GameFeatures.feature_names()
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)

    logger.info("학습 데이터: %d행 × %d피처", X.shape[0], X.shape[1])
    logger.info("홈팀 승률 (라벨 분포): %.1f%%", y.mean() * 100)

    model, cv_results = train_model(X, y, n_splits=5, save_name="xgb_model")

    logger.info("\n── CV 결과 ──")
    logger.info("평균 정확도: %.2f%%", cv_results["mean_accuracy"] * 100)
    logger.info("평균 Brier Score: %.4f", cv_results["mean_brier"])
    logger.info("평균 Log Loss: %.4f", cv_results["mean_logloss"])

    for fold in cv_results["folds"]:
        logger.info(
            "  Fold %d: acc=%.2f%% (train=%d, val=%d)",
            fold["fold"], fold["accuracy"] * 100,
            fold["train_size"], fold["val_size"],
        )

    # ── 4단계: 2026 시즌 Elo 초기화 ──
    logger.info("\n[4/4] 2026 시즌 Elo 초기화")
    elo = EloEngine()
    if elo.load():
        logger.info("2025 시즌 말 Elo 로드 성공")
        # 각 팀 현재 Elo 출력
        for code, rating in elo.get_all_ratings().items():
            from config.constants import TEAM_CODES
            name = TEAM_CODES.get(code, str(code))
            logger.info("  %s: %.0f", name, rating)

        # 시즌 초기화 (1/3 평균 회귀)
        elo.new_season()
        logger.info("\n2026 시즌 초기 Elo (1/3 회귀 적용):")
        for code, rating in elo.get_all_ratings().items():
            name = TEAM_CODES.get(code, str(code))
            logger.info("  %s: %.0f", name, rating)
    else:
        logger.warning("Elo 데이터 없음 — 모든 팀 1500으로 시작")

    logger.info("\n" + "=" * 60)
    logger.info("전체 학습 파이프라인 완료!")
    logger.info("=" * 60)
    logger.info("")
    logger.info("다음 단계:")
    logger.info("  python pipeline/daily_run.py --date 2026-03-28 --dry-run")
    logger.info("  → 개막일 예측 테스트")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="전체 학습 파이프라인")
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025],
                        help="학습 데이터 연도 (기본: 2023 2024 2025)")
    parser.add_argument("--skip-backfill", action="store_true",
                        help="데이터 수집 스킵 (이미 수집된 경우)")
    args = parser.parse_args()

    full_train(years=args.years, skip_backfill=args.skip_backfill)
