"""
SHAP 피처 중요도 분석.

학습된 XGBoost 모델의 피처별 기여도를 시각화.
"""
import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.builder import GameFeatures
from models.train import load_model

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "analysis" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def run_shap_analysis(data_path: str = None, save_plot: bool = True):
    """SHAP 분석 실행.

    Args:
        data_path: 학습 데이터 CSV 경로. None이면 기본 경로.
        save_plot: True면 PNG 저장.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # 데이터 로드
    if data_path is None:
        data_path = str(DATA_DIR / "training_data.csv")
    df = pd.read_csv(data_path)

    feature_cols = GameFeatures.feature_names()
    X = df[feature_cols].values.astype(np.float32)

    # 모델 로드
    model = load_model()
    if model is None:
        logger.error("XGBoost 모델 없음. 먼저 학습을 실행하세요.")
        return

    try:
        import shap
    except ImportError:
        logger.error("shap 패키지 없음. pip install shap 실행 필요.")
        # SHAP 없이 XGBoost 내장 feature importance 사용
        logger.info("XGBoost 내장 feature importance로 대체합니다.")
        _fallback_importance(model, feature_cols)
        return

    # SHAP 분석
    logger.info("SHAP 분석 시작 (데이터 %d행)", X.shape[0])
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 평균 절대 SHAP 값
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

    # JSON 저장
    json_path = OUTPUT_DIR / "shap_importance.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info("저장: %s", json_path)

    # 플롯 저장
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
