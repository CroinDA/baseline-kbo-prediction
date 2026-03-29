"""
v7 예측 모듈 — XGBoost 직접 예측, 외부 Elo 블렌딩 없음.

v65 대비:
- Elo 확률이 피처 벡터 내에 포함 → 별도 블렌딩 불필요
- threshold 튜닝 적용
- avoid_forbidden 재사용
"""
import logging
from typing import Optional

import numpy as np
import xgboost as xgb

from config.constants import (
    SUBMIT_FORBIDDEN,
    SUBMIT_MIN_PROB,
    SUBMIT_MAX_PROB,
    SUBMIT_DECIMAL_PLACES,
)

logger = logging.getLogger(__name__)


def avoid_forbidden(prob: float, elo_direction: float) -> float:
    """50.00% 회피 + 포맷팅."""
    prob = round(prob, SUBMIT_DECIMAL_PLACES)
    if prob == SUBMIT_FORBIDDEN:
        prob = 50.01 if elo_direction >= 0 else 49.99
    prob = max(SUBMIT_MIN_PROB, min(SUBMIT_MAX_PROB, prob))
    return round(prob, SUBMIT_DECIMAL_PLACES)


def predict_game_v7(
    xgb_model: xgb.XGBClassifier,
    features: list[float],
    feature_names: list[str],
    threshold: float = 0.5,
) -> float:
    """v7 단일 경기 예측.

    Returns:
        홈팀 승리 확률 (0~100 스케일, 소수점 둘째 자리)
    """
    X = np.array([features])
    prob = xgb_model.predict_proba(X)[0, 1]
    prob_pct = prob * 100.0

    # elo_diff를 피처에서 추출 (50.00% 회피용)
    elo_diff = 0.0
    if "elo_diff" in feature_names:
        idx = feature_names.index("elo_diff")
        elo_diff = features[idx]

    result = avoid_forbidden(prob_pct, elo_diff)
    return float(result)


def batch_predict_v7(
    xgb_model: xgb.XGBClassifier,
    games: list[dict],
    feature_names: list[str],
    threshold: float = 0.5,
) -> list[dict]:
    """v7 일괄 예측."""
    results = []
    for game in games:
        prob = predict_game_v7(
            xgb_model,
            game["features"],
            feature_names,
            threshold=threshold,
        )
        results.append({
            "s_no": game["s_no"],
            "percent": prob,
        })
        logger.info(
            "v7 경기 %d: 홈(%d) vs 원정(%d) → %.2f%%",
            game["s_no"], game["home_team"], game["away_team"], prob,
        )
    return results
