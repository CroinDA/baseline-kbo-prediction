"""
전체 학습 오케스트레이터.

1단계: 과거 데이터 수집 (backfill) — API 키 필요
2단계: 학습 데이터셋 구축 (v1/v2/v4/v5 전버전)
3단계: XGBoost 학습 (시계열 CV + 모델 저장)
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
from models.train import train_model
from models.train_v8 import train_v8_model
from pipeline.build_v65_plus import build_dataset_v65_plus
from elo.engine import EloEngine

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# 버전별 빌드 함수 + 데이터셋/모델 이름 매핑
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


def full_train(
    years: list[int] = None,
    skip_backfill: bool = False,
    versions: list[str] = None,
):
    """전체 학습 파이프라인 실행.

    Args:
        years: 학습 데이터 연도 (기본: 2023~2025)
        skip_backfill: True면 데이터 수집 스킵 (이미 수집된 경우)
        versions: 학습할 버전 리스트 (기본: 전체)
    """
    if years is None:
        years = [2023, 2024, 2025]
    if versions is None:
        versions = ["v65", "v64"]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    logger.info("=" * 60)
    logger.info("전체 학습 파이프라인 시작 (버전: %s)", versions)
    logger.info("=" * 60)

    # ── 1단계: 과거 데이터 수집 ──
    if not skip_backfill:
        logger.info("\n[1/3] 과거 데이터 수집 (backfill)")
        run_backfill(years)
    else:
        logger.info("\n[1/3] 과거 데이터 수집 — 스킵")

    # ── 2단계: 버전별 데이터셋 구축 + 학습 ──
    import importlib

    for version in VERSIONS:
        ver_name = version["name"]
        if ver_name not in versions:
            continue

        logger.info("\n[2/3] %s 데이터셋 구축 + 학습", ver_name)

        # 동적 임포트
        mod = importlib.import_module(version["module_path"])
        build_func = getattr(mod, version["func_name"])
        df = build_func(years=years, output_name=version["dataset_name"], **version.get("kwargs", {}))

        if df.empty:
            logger.error("  %s 데이터셋 비어있음 — 스킵", ver_name)
            continue

        feature_cols = [c for c in df.columns if c != "label"]
        X = df[feature_cols].values.astype(np.float32)
        y = df["label"].values.astype(np.int32)

        logger.info("  %s 학습: %d행 × %d피처", ver_name, X.shape[0], X.shape[1])
        model, cv_results = train_model(X, y, n_splits=5, save_name=version["model_name"])
        logger.info("  %s CV 정확도: %.2f%%", ver_name, cv_results["mean_accuracy"] * 100)

    # ── v8 학습 (v65+ 229피처 + 최적화 파라미터) ──
    if "v8" in versions or "v65" in versions:
        logger.info("\n[2.5/3] v8 학습 (v65+ 229피처 + Optuna 최적 파라미터)")
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

    # ── 3단계: 2026 시즌 Elo 초기화 ──
    logger.info("\n[3/3] 2026 시즌 Elo 초기화")
    elo = EloEngine()
    if elo.load():
        logger.info("2025 시즌 말 Elo 로드 성공")
        elo.new_season()
        for code, rating in elo.get_all_ratings().items():
            from config.constants import TEAM_CODES
            name = TEAM_CODES.get(code, str(code))
            logger.info("  %s: %.0f", name, rating)
    else:
        logger.warning("Elo 데이터 없음 — 모든 팀 1500으로 시작")

    logger.info("\n" + "=" * 60)
    logger.info("전체 학습 파이프라인 완료!")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="전체 학습 파이프라인")
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025],
                        help="학습 데이터 연도 (기본: 2023 2024 2025)")
    parser.add_argument("--skip-backfill", action="store_true",
                        help="데이터 수집 스킵 (이미 수집된 경우)")
    parser.add_argument("--versions", nargs="+", default=["v1", "v2", "v4", "v5"],
                        help="학습할 버전 (기본: v1 v2 v4 v5)")
    args = parser.parse_args()

    full_train(years=args.years, skip_backfill=args.skip_backfill, versions=args.versions)
