"""
실시간 regular-season 누적 반영 후 v65 재학습.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.train import train_model
from pipeline.build_v6_timeaware import build_dataset_v6

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "saved"


def retrain_live_v65(years: list[int] | None = None) -> dict:
    if years is None:
        years = [2023, 2024, 2025, 2026]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    logger.info("live v65 데이터셋 재생성 시작: %s", years)
    df = build_dataset_v6(
        years=years,
        output_name="training_data_v65_live",
        include_bench=False,
        include_sp_war=True,
        stabilize_rates=True,
        tune_bullpen=True,
        use_spring=False,
    )

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y = df["label"].values

    logger.info("live v65 재학습 시작: %d행 × %d피처", X.shape[0], X.shape[1])
    _, cv_results = train_model(X, y, save_name="xgb_v65")

    manifest = {
        "years": years,
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "mean_accuracy": cv_results["mean_accuracy"],
        "model_path": cv_results["model_path"],
    }
    manifest_path = MODEL_DIR / "xgb_v65_live_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    logger.info("live v65 재학습 완료: %s", manifest_path)
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="retrain live v65")
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025, 2026])
    args = parser.parse_args()

    result = retrain_live_v65(args.years)
    print(json.dumps(result, ensure_ascii=False, indent=2))
