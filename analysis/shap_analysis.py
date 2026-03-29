"""
SHAP 피처 중요도 분석.

학습된 XGBoost 모델의 피처별 기여도를 시각화.
v5 (221피처) 모델 기준, v2/v4/v1 폴백.
"""
import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.train import load_model

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "analysis" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 모델-데이터 매핑 (우선순위 순)
MODEL_DATA_MAP = [
    ("xgb_v5", "training_data_v5"),
    ("xgb_v2", "training_data_v2"),
    ("xgb_v4", "training_data_v4"),
    ("xgb_v1", "training_data_v1"),
]


def run_shap_analysis(data_path: str = None, save_plot: bool = True):
    """SHAP 분석 실행.

    Args:
        data_path: 학습 데이터 경로. None이면 자동 탐색.
        save_plot: True면 PNG 저장.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # 모델 + 데이터 자동 매칭
    model = None
    df = None

    if data_path:
        df = pd.read_csv(data_path) if data_path.endswith(".csv") else pd.read_parquet(data_path)
        # 데이터에 맞는 모델 찾기
        for model_name, _ in MODEL_DATA_MAP:
            model = load_model(model_name)
            if model is not None:
                break
    else:
        for model_name, dataset_name in MODEL_DATA_MAP:
            model = load_model(model_name)
            if model is None:
                continue
            parquet = DATA_DIR / f"{dataset_name}.parquet"
            csv = DATA_DIR / f"{dataset_name}.csv"
            if parquet.exists():
                df = pd.read_parquet(parquet)
                break
            elif csv.exists():
                df = pd.read_csv(csv)
                break
            else:
                model = None  # 데이터 없으면 다음 버전

    if model is None or df is None:
        logger.error("모델 또는 데이터 없음. 먼저 학습을 실행하세요.")
        return

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float32)

    try:
        import shap
    except ImportError:
        logger.info("shap 패키지 없음. XGBoost 내장 feature importance로 대체.")
        _fallback_importance(model, feature_cols)
        return

    # SHAP 분석
    logger.info("SHAP 분석 시작 (데이터 %d행 × %d피처)", X.shape[0], X.shape[1])
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = sorted(
        zip(feature_cols, mean_abs_shap),
        key=lambda x: -x[1],
    )

    logger.info("\n── SHAP 피처 중요도 (평균 |SHAP|) ──")
    results = []
    for name, val in importance:
        logger.info("  %s: %.4f", name, val)
        results.append({"feature": name, "mean_abs_shap": round(float(val), 4)})

    json_path = OUTPUT_DIR / "shap_importance.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info("저장: %s", json_path)

    if save_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            shap.summary_plot(
                shap_values, X,
                feature_names=feature_cols,
                show=False,
            )
            plot_path = OUTPUT_DIR / "shap_summary.png"
            plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("플롯 저장: %s", plot_path)
        except Exception as e:
            logger.warning("플롯 저장 실패: %s", e)


def _fallback_importance(model, feature_cols: list[str]):
    """SHAP 없을 때 XGBoost 내장 feature importance 사용."""
    importances = model.feature_importances_
    importance = sorted(
        zip(feature_cols, importances),
        key=lambda x: -x[1],
    )

    logger.info("\n── XGBoost Feature Importance (gain) ──")
    results = []
    for name, val in importance:
        logger.info("  %s: %.4f", name, val)
        results.append({"feature": name, "importance": round(float(val), 4)})

    json_path = OUTPUT_DIR / "xgb_importance.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info("저장: %s", json_path)


if __name__ == "__main__":
    run_shap_analysis()
