"""
실시간 v8 재학습 — v65+ 데이터셋 + v8 최적화 파라미터.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.train_v8 import train_v8_model
from pipeline.build_v65_plus import build_dataset_v65_plus

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "saved"


def retrain_live_v8(years: list[int] | None = None) -> dict:
    if years is None:
        years = [2023, 2024, 2025, 2026]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    logger.info("live v8 데이터셋 재생성: %s", years)
    df = build_dataset_v65_plus(years=years, output_name="training_data_v65plus_live")

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y = df["label"].values

    logger.info("live v8 재학습: %d행 × %d피처", X.shape[0], X.shape[1])
    _, cv_results = train_v8_model(X, y, save_name="xgb_v8")

    manifest = {
        "years": years,
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "mean_accuracy": cv_results["mean_accuracy"],
        "mean_accuracy_tuned": cv_results["mean_accuracy_tuned"],
        "mean_threshold": cv_results["mean_threshold"],
        "model_path": cv_results["model_path"],
    }
    manifest_path = MODEL_DIR / "xgb_v8_live_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    logger.info("live v8 완료: %s", manifest_path)
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="retrain live v8")
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025, 2026])
    args = parser.parse_args()

    result = retrain_live_v8(args.years)
    print(json.dumps(result, ensure_ascii=False, indent=2))
