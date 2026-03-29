"""
v65+ 데이터셋 구축 — v65 기반 + 비교 피처 10개 추가.

v65 219피처에 팀간 비교/집계 파생 피처 추가:
- lineup_wrc_diff, sp_fip_diff, elo_diff, log5 등
- 모델이 "누가 더 강한가"를 직접 학습할 수 있는 비교 시그널
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from pipeline.build_v6_timeaware import build_dataset_v6

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

COMPARISON_FEATURES = [
    "home_lineup_mean_wrc",
    "away_lineup_mean_wrc",
    "lineup_wrc_diff",
    "home_lineup_top3_wrc",
    "away_lineup_top3_wrc",
    "sp_fip_diff",
    "elo_diff",
    "log5_home",
    "pyth_diff",
    "wpct_diff",
]


def add_comparison_features(df: pd.DataFrame) -> pd.DataFrame:
    """v65 DataFrame에 비교 피처 10개 추가."""
    df = df.copy()

    batter_wrc_home = [f"home_b{i}_wrcplus" for i in range(1, 10)]
    batter_wrc_away = [f"away_b{i}_wrcplus" for i in range(1, 10)]

    df["home_lineup_mean_wrc"] = df[batter_wrc_home].mean(axis=1)
    df["away_lineup_mean_wrc"] = df[batter_wrc_away].mean(axis=1)
    df["lineup_wrc_diff"] = df["home_lineup_mean_wrc"] - df["away_lineup_mean_wrc"]

    home_wrc = df[batter_wrc_home].values
    away_wrc = df[batter_wrc_away].values
    df["home_lineup_top3_wrc"] = np.sort(home_wrc, axis=1)[:, -3:].mean(axis=1)
    df["away_lineup_top3_wrc"] = np.sort(away_wrc, axis=1)[:, -3:].mean(axis=1)

    df["sp_fip_diff"] = df["away_sp_fip"] - df["home_sp_fip"]
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    home_wpct = df["home_wpct"].clip(0.01, 0.99)
    away_wpct = df["away_wpct"].clip(0.01, 0.99)
    df["log5_home"] = (home_wpct * (1 - away_wpct)) / (
        home_wpct * (1 - away_wpct) + away_wpct * (1 - home_wpct)
    )

    df["pyth_diff"] = df["home_pyth_exp"] - df["away_pyth_exp"]
    df["wpct_diff"] = df["home_wpct"] - df["away_wpct"]

    return df


def add_comparison_features_live(v65_features: list[float], v65_names: list[str]) -> list[float]:
    """v65 live 피처 벡터에 비교 피처 10개 추가.

    auto_submit에서 v65 피처 벡터를 받아 10개 파생 피처를 append.
    """
    feat = dict(zip(v65_names, v65_features))

    home_wrc = [feat.get(f"home_b{i}_wrcplus", 100.0) for i in range(1, 10)]
    away_wrc = [feat.get(f"away_b{i}_wrcplus", 100.0) for i in range(1, 10)]

    home_mean_wrc = float(np.mean(home_wrc))
    away_mean_wrc = float(np.mean(away_wrc))
    lineup_wrc_diff = home_mean_wrc - away_mean_wrc
    home_top3 = float(np.mean(sorted(home_wrc, reverse=True)[:3]))
    away_top3 = float(np.mean(sorted(away_wrc, reverse=True)[:3]))

    sp_fip_diff = feat.get("away_sp_fip", 4.5) - feat.get("home_sp_fip", 4.5)
    elo_diff = feat.get("home_elo", 1500.0) - feat.get("away_elo", 1500.0)

    home_wpct = max(0.01, min(0.99, feat.get("home_wpct", 0.5)))
    away_wpct = max(0.01, min(0.99, feat.get("away_wpct", 0.5)))
    log5 = (home_wpct * (1 - away_wpct)) / (
        home_wpct * (1 - away_wpct) + away_wpct * (1 - home_wpct)
    )

    pyth_diff = feat.get("home_pyth_exp", 0.5) - feat.get("away_pyth_exp", 0.5)
    wpct_diff = home_wpct - away_wpct

    return v65_features + [
        home_mean_wrc, away_mean_wrc, lineup_wrc_diff,
        home_top3, away_top3,
        sp_fip_diff, elo_diff, log5,
        pyth_diff, wpct_diff,
    ]


def build_dataset_v65_plus(
    years: list[int] = None,
    output_name: str = "training_data_v65plus",
) -> pd.DataFrame:
    """v65 데이터셋 생성 후 비교 피처 추가."""
    if years is None:
        years = [2023, 2024, 2025]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    df = build_dataset_v6(
        years=years,
        output_name=f"_tmp_v65_{output_name}",
        include_bench=False,
        include_sp_war=True,
        stabilize_rates=True,
        tune_bullpen=True,
        use_spring=False,
    )

    df = add_comparison_features(df)
    logger.info("비교 피처 추가 완료: %d열 → %d열", len(df.columns) - 10, len(df.columns))

    csv_path = DATA_DIR / f"{output_name}.csv"
    df.to_csv(csv_path, index=False)
    logger.info("저장: %s", csv_path)

    try:
        parquet_path = DATA_DIR / f"{output_name}.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info("저장: %s", parquet_path)
    except Exception:
        logger.info("Parquet 저장 스킵")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="v65+ 데이터셋 구축")
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    parser.add_argument("--output", type=str, default="training_data_v65plus")
    args = parser.parse_args()

    build_dataset_v65_plus(years=args.years, output_name=args.output)
