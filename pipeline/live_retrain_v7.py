"""
실시간 regular-season 누적 반영 후 v7 재학습.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.train_v7 import train_model_v7
from pipeline.build_v7_aggregated import build_dataset_v7

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "saved"


def retrain_live_v7(years: list[int] | None = None) -> dict:
    if years is None:
        years = [2023, 2024, 2025, 2026]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    logger.info("live v7 데이터셋 재생성 시작: %s", years)
    df = build_dataset_v7(
        years=years,
        output_name="training_data_v7_live",
        augment_mirror=True,
    )

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y = df["label"].values

    logger.info("live v7 재학습 시작: %d행 × %d피처", X.shape[0], X.shape[1])
    _, cv_results = train_model_v7(X, y, save_name="xgb_v7")

    manifest = {
        "years": years,
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "mean_accuracy": cv_results["mean_accuracy_tuned"],
        "threshold": cv_results["optimal_threshold"],
        "model_path": cv_results["model_path"],
    }
    manifest_path = MODEL_DIR / "xgb_v7_live_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    logger.info("live v7 재학습 완료: %s", manifest_path)
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="retrain live v7")
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025, 2026])
    args = parser.parse_args()

    result = retrain_live_v7(args.years)
    print(json.dumps(result, ensure_ascii=False, indent=2))
