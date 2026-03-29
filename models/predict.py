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
    PRIOR_SCHEDULE,
    SUBMIT_FORBIDDEN,
    SUBMIT_MIN_PROB,
    SUBMIT_MAX_PROB,
    SUBMIT_DECIMAL_PLACES,
)
from elo.engine import EloEngine

logger = logging.getLogger(__name__)


def get_elo_weight(games_played: int) -> float:
    """시즌 진행 경기 수에 따른 Elo 가중치.

    BLEND_SCHEDULE: [(30, 0.80), (80, 0.60), (120, 0.45), (999, 0.40)]
    """
    for threshold, weight in BLEND_SCHEDULE:
        if games_played <= threshold:
            return weight
    return BLEND_SCHEDULE[-1][1]


def get_prior_weight(games_played: int) -> float:
    """시즌 초반 Prior 비중 (콜드 스타트 대응).

    PRIOR_SCHEDULE: [(20, 0.70), (50, 0.50), (80, 0.30), (999, 0.20)]
    시즌 초반에는 전시즌 Prior에 더 의존하고,
    시즌이 진행될수록 현시즌 데이터 기반 예측에 의존.
    """
    for threshold, weight in PRIOR_SCHEDULE:
        if games_played <= threshold:
            return weight
    return PRIOR_SCHEDULE[-1][1]


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
    features,
    home_team: int,
    away_team: int,
    home_sp: Optional[int] = None,
    away_sp: Optional[int] = None,
    threshold: float = 0.5,
    deadzone_push: float = 0.0,
) -> float:
    """단일 경기 홈팀 승리 확률 예측.

    Elo 확률과 XGBoost 확률을 가중 평균.

    Args:
        features: 피처 벡터 (list[float] 또는 to_list() 가진 객체)
        threshold: v8 threshold correction (0.5 = 보정 없음)
        deadzone_push: v8 dead zone push 강도 (0-1 스케일, 예: 0.02 = 2%p)

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
        feat_list = features.to_list() if hasattr(features, 'to_list') else features
        X = np.array([feat_list])
        xgb_prob = xgb_model.predict_proba(X)[0, 1]
        # v8: threshold correction — shift so model threshold maps to 0.5
        if threshold != 0.5:
            xgb_prob += (0.5 - threshold)
            xgb_prob = float(np.clip(xgb_prob, 0.01, 0.99))
        xgb_prob_pct = xgb_prob * 100.0
    else:
        # XGBoost 모델이 없으면 Elo만 사용
        xgb_prob_pct = elo_prob_pct

    # ── 블렌딩 (Elo + XGBoost) ──
    elo_weight = get_elo_weight(elo_engine.games_played)
    xgb_weight = 1.0 - elo_weight

    blended = elo_weight * elo_prob_pct + xgb_weight * xgb_prob_pct

    # ── 콜드 스타트 Prior-Posterior 블렌딩 ──
    prior_w = get_prior_weight(elo_engine.games_played)
    prior_pct = elo_prob_pct
    posterior_pct = blended

    final_pct = prior_w * prior_pct + (1.0 - prior_w) * posterior_pct

    # ── v8: Dead zone push ──
    if deadzone_push > 0 and 48.0 <= final_pct <= 52.0:
        push_pct = deadzone_push * 100.0
        final_pct += push_pct * (1.0 if elo_direction >= 0 else -1.0)

    logger.debug(
        "Elo=%.2f%% (w=%.2f), XGB=%.2f%% (w=%.2f) → Blend=%.2f%%, "
        "Prior=%.2f (w=%.2f) → Final=%.2f%%",
        elo_prob_pct, elo_weight, xgb_prob_pct, xgb_weight, blended,
        prior_pct, prior_w, final_pct,
    )

    # ── 50.00% 회피 + 포맷팅 ──
    result = avoid_forbidden(final_pct, elo_direction)

    return float(result)


def batch_predict(
    elo_engine: EloEngine,
    xgb_model: Optional[xgb.XGBClassifier],
    games: list[dict],
    threshold: float = 0.5,
    deadzone_push: float = 0.0,
) -> list[dict]:
    """여러 경기 일괄 예측.

    Args:
        games: [{"s_no": int, "home_team": int, "away_team": int,
                 "home_sp": int, "away_sp": int, "features": GameFeatures}, ...]
        threshold: v8 threshold correction
        deadzone_push: v8 dead zone push 강도

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
            threshold=threshold,
            deadzone_push=deadzone_push,
        )
        results.append({
            "s_no": game["s_no"],
            "percent": prob,
        })
        logger.info("경기 %d: 홈(%d) vs 원정(%d) → %.2f%%",
                     game["s_no"], game["home_team"], game["away_team"], prob)

    return results
