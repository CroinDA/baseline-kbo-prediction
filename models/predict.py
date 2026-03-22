"""
예측 모듈 — Elo + XGBoost 블렌딩.

시즌 진행도에 따라 Elo와 XGBoost의 가중치를 동적으로 조절.
50.00% 회피 로직 포함.
"""
import logging
from typing import Optional

import numpy as np
import xgboost as xgb

from config.constants import (
    BLEND_SCHEDULE,
    SUBMIT_FORBIDDEN,
    SUBMIT_MIN_PROB,
    SUBMIT_MAX_PROB,
    SUBMIT_DECIMAL_PLACES,
)
from elo.engine import EloEngine
from features.builder import GameFeatures

logger = logging.getLogger(__name__)


def get_elo_weight(games_played: int) -> float:
    """시즌 진행 경기 수에 따른 Elo 가중치.

    BLEND_SCHEDULE: [(30, 0.80), (80, 0.60), (120, 0.45), (999, 0.40)]
    """
    for threshold, weight in BLEND_SCHEDULE:
        if games_played <= threshold:
            return weight
    return BLEND_SCHEDULE[-1][1]


def avoid_forbidden(prob: float, elo_direction: float) -> float:
    """50.00% 회피 + 소수점 포맷팅.

    Args:
        prob: 홈팀 승리 확률 (0~100 스케일)
        elo_direction: Elo 기반 방향 (양수면 홈팀 우세)

    Returns:
        제출용 확률 (소수점 둘째 자리)
    """
    prob = round(prob, SUBMIT_DECIMAL_PLACES)

    # 50.00% 회피: Elo 방향으로 밀기
    if prob == SUBMIT_FORBIDDEN:
        if elo_direction >= 0:
            prob = 50.01
        else:
            prob = 49.99

    # 범위 클램핑
    prob = max(SUBMIT_MIN_PROB, min(SUBMIT_MAX_PROB, prob))

    return round(prob, SUBMIT_DECIMAL_PLACES)


def predict_game(
    elo_engine: EloEngine,
    xgb_model: Optional[xgb.XGBClassifier],
    features: GameFeatures,
    home_team: int,
    away_team: int,
    home_sp: Optional[int] = None,
    away_sp: Optional[int] = None,
) -> float:
    """단일 경기 홈팀 승리 확률 예측.

    Elo 확률과 XGBoost 확률을 가중 평균.

    Returns:
        홈팀 승리 확률 (0~100 스케일, 소수점 둘째 자리)
    """
    # ── Layer 1: Elo 예측 ──
    elo_prob = elo_engine.predict(home_team, away_team, home_sp, away_sp)
    elo_prob_pct = elo_prob * 100.0

    # Elo 방향 (50.00% 회피용)
    elo_direction = elo_prob - 0.5

    # ── Layer 2: XGBoost 예측 ──
    if xgb_model is not None:
        X = np.array([features.to_list()])
        xgb_prob = xgb_model.predict_proba(X)[0, 1]
        xgb_prob_pct = xgb_prob * 100.0
    else:
        # XGBoost 모델이 없으면 Elo만 사용
        xgb_prob_pct = elo_prob_pct

    # ── 블렌딩 ──
    elo_weight = get_elo_weight(elo_engine.games_played)
    xgb_weight = 1.0 - elo_weight

    blended = elo_weight * elo_prob_pct + xgb_weight * xgb_prob_pct

    logger.debug(
        "Elo=%.2f%% (w=%.2f), XGB=%.2f%% (w=%.2f) → Blend=%.2f%%",
        elo_prob_pct, elo_weight, xgb_prob_pct, xgb_weight, blended,
    )

    # ── 50.00% 회피 + 포맷팅 ──
    result = avoid_forbidden(blended, elo_direction)

    return result


def batch_predict(
    elo_engine: EloEngine,
    xgb_model: Optional[xgb.XGBClassifier],
    games: list[dict],
) -> list[dict]:
    """여러 경기 일괄 예측.

    Args:
        games: [{"s_no": int, "home_team": int, "away_team": int,
                 "home_sp": int, "away_sp": int, "features": GameFeatures}, ...]

    Returns:
        [{"s_no": int, "percent": float}, ...]
    """
    results = []
    for game in games:
        prob = predict_game(
            elo_engine=elo_engine,
            xgb_model=xgb_model,
            features=game["features"],
            home_team=game["home_team"],
            away_team=game["away_team"],
            home_sp=game.get("home_sp"),
            away_sp=game.get("away_sp"),
        )
        results.append({
            "s_no": game["s_no"],
            "percent": prob,
        })
        logger.info("경기 %d: 홈(%d) vs 원정(%d) → %.2f%%",
                     game["s_no"], game["home_team"], game["away_team"], prob)

    return results
