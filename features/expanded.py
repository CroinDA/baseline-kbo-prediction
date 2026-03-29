"""
피처 빌더 — v4(37), v2(159), v5(221) 피처 정의 및 추출 함수.

v4: 37개 집약 피처
v2: 159개 개인 타자 피처
v5: 221개 확장 피처 (v2 159 + platoon 2 + weather 3 + wind 2 + batter deepen 36
    + pitcher deepen 4 + situational 15)
"""
import math
import logging
from typing import Optional

import numpy as np

from config.constants import WEATHER_CODES, WIND_DIR_CODES

logger = logging.getLogger(__name__)

# ── 피처 정의 (v4) ──

# 타선 집계 (팀별)
BATTING_AGG_STATS = [
    "top5_wrc",      # 1~5번 타자 평균 wRC+
    "bot4_wrc",      # 6~9번 타자 평균 wRC+
    "team_obp",      # 라인업 전체 평균 OBP
    "team_slg",      # 라인업 전체 평균 SLG
    "bb_k_ratio",    # 라인업 전체 BB/K 비율 (선구안)
    "platoon_adv",   # 상대 선발 대비 플래툰 어드밴티지 비율
]

# 선발투수 stats (ERA 유지 — 실증 중요도 #1)
SP_STATS = ["era", "fip", "k9", "bb9", "war"]

# 팀 컨텍스트
TEAM_STATS = ["elo", "pyth_exp", "recent_wpct", "rest_days"]

# 시퀀스
SEQ_STATS = ["scoring_trend", "lineup_wrc_delta"]

# 환경
ENV_STATS = ["park_factor", "temperature", "is_night"]

# ── v5 신규 피처 (15개, 기존 situational) ──
V5_SITUATIONAL_STATS = [
    # H2H 상대전적 (3)
    "h2h_home_wpct", "h2h_away_wpct", "h2h_n_games",
    # DAGN 피로 (2)
    "dagn_home", "dagn_away",
    # 이동거리 (2)
    "travel_home", "travel_away",
    # 불펜 연투 (2)
    "bp_consec_home", "bp_consec_away",
    # 최근 5경기 폼 (4)
    "recent5_wpct_home", "recent5_wpct_away",
    "recent5_rs_home", "recent5_rs_away",
    # SP 상대팀 전적 (2)
    "sp_vs_opp_home", "sp_vs_opp_away",
]

# ── v5 확장 피처 (47개 신규, 총 221 = 159 + 15 + 47) ──
V5_EXPANDED_STATS = [
    # platoon (2)
    "platoon_adv_home", "platoon_adv_away",
    # weather one-hot (3)
    "weather_clear", "weather_overcast", "weather_rain",
    # wind direction sin/cos (2)
    "wind_dir_sin", "wind_dir_cos",
    # batter wOBA (18)
    *[f"{side}_b{i}_woba" for side in ["home", "away"] for i in range(1, 10)],
    # batter BABIP (18)
    *[f"{side}_b{i}_babip" for side in ["home", "away"] for i in range(1, 10)],
    # pitcher deepen (4)
    "home_sp_obp_against", "home_sp_slg_against",
    "away_sp_obp_against", "away_sp_slg_against",
]

# 리그 평균 폴백값
LEAGUE_AVG_BATTER = {
    "wrcplus": 100.0, "avg": 0.260, "obp": 0.330, "slg": 0.400,
    "hr_rate": 0.025, "bb_rate": 0.085, "k_rate": 0.200,
}

LEAGUE_AVG_BATTER_V5 = {
    **LEAGUE_AVG_BATTER,
    "woba": 0.320, "babip": 0.300,
}

LEAGUE_AVG_SP = {
    "era": 4.50, "fip": 4.50, "k9": 7.0, "bb9": 3.5, "war": 1.0,
}

LEAGUE_AVG_SP_V5 = {
    **LEAGUE_AVG_SP,
    "obp_against": 0.330, "slg_against": 0.400,
}


# ── v4 37개 피처 이름 ──

def feature_names() -> list[str]:
    """37개 피처 이름 리스트."""
    names = []

    # 타선 집계 (12)
    for side in ["home", "away"]:
        for stat in BATTING_AGG_STATS:
            names.append(f"{side}_{stat}")

    # 선발투수 (10)
    for side in ["home", "away"]:
        for stat in SP_STATS:
            names.append(f"{side}_sp_{stat}")

    # 팀 컨텍스트 (8)
    for side in ["home", "away"]:
        for stat in TEAM_STATS:
            names.append(f"{side}_{stat}")

    # 시퀀스 (4)
    for side in ["home", "away"]:
        for stat in SEQ_STATS:
            names.append(f"{side}_{stat}")

    # 환경 (3)
    names.extend(ENV_STATS)

    return names


# ── v2 159개 피처 이름 ──

def feature_names_v2() -> list[str]:
    """159개 v2 피처 이름 리스트 (개인 타자 포함)."""
    names = []
    batter_stats = ["wrcplus", "avg", "obp", "slg", "hr_rate", "bb_rate", "k_rate"]
    sp_stats = ["era", "fip", "whip", "k9", "bb9", "hr9", "war"]
    team_stats = ["elo", "pyth_exp", "wpct", "bp_load", "rest_days"]
    seq_stats = ["scoring_trend", "lineup_wrc_delta"]
    env_stats = ["temperature", "humidity", "wind_speed", "is_night", "park_factor"]

    # 개인 타자 126 (7 × 9 × 2)
    for side in ["home", "away"]:
        for i in range(1, 10):
            for stat in batter_stats:
                names.append(f"{side}_b{i}_{stat}")

    # SP 14 (7 × 2)
    for side in ["home", "away"]:
        for stat in sp_stats:
            names.append(f"{side}_sp_{stat}")

    # 팀 10 (5 × 2)
    for side in ["home", "away"]:
        for stat in team_stats:
            names.append(f"{side}_{stat}")

    # 시퀀스 4 (2 × 2)
    for side in ["home", "away"]:
        for stat in seq_stats:
            names.append(f"{side}_{stat}")

    # 환경 5
    names.extend(env_stats)

    return names


# ── v5 221개 피처 이름 ──

def feature_names_v5() -> list[str]:
    """221개 v5 피처 이름 리스트.

    v2 159 + situational 15 + platoon 2 + weather 3 + wind 2
    + batter wOBA 18 + batter BABIP 18 + pitcher deepen 4 = 221
    """
    return (
        feature_names_v2()
        + list(V5_SITUATIONAL_STATS)
        + list(V5_EXPANDED_STATS)
    )


# ── 추출 함수 ──

def extract_batter_stats(basic_rec: Optional[dict]) -> dict:
    """타자 시즌 기록에서 7개 rate stats 추출 (v2 호환)."""
    if not basic_rec:
        return dict(LEAGUE_AVG_BATTER)

    pa = float(basic_rec.get("PA") or basic_rec.get("ePA") or 1)
    if pa <= 0:
        pa = 1

    return {
        "wrcplus": float(basic_rec.get("wRCplus") or 100.0),
        "avg": float(basic_rec.get("AVG") or 0.260),
        "obp": float(basic_rec.get("OBP") or 0.330),
        "slg": float(basic_rec.get("SLG") or 0.400),
        "hr_rate": float(basic_rec.get("HR") or 0) / pa,
        "bb_rate": float(basic_rec.get("BB") or 0) / pa,
        "k_rate": float(basic_rec.get("SO") or 0) / pa,
    }


def extract_batter_stats_v5(
    basic_rec: Optional[dict],
    deepen_rec: Optional[dict] = None,
) -> dict:
    """타자 시즌 기록에서 9개 stats 추출 (v5 — wOBA, BABIP 추가)."""
    base = extract_batter_stats(basic_rec)
    woba = 0.320
    babip = 0.300

    if deepen_rec:
        woba = float(deepen_rec.get("wOBA") or 0.320)
        babip = float(deepen_rec.get("BABIP") or 0.300)
    elif basic_rec:
        # basic에 있을 수도 있음
        if basic_rec.get("wOBA"):
            woba = float(basic_rec["wOBA"])
        if basic_rec.get("BABIP"):
            babip = float(basic_rec["BABIP"])

    base["woba"] = woba
    base["babip"] = babip
    return base


def extract_sp_stats(
    basic_rec: Optional[dict],
    deepen_rec: Optional[dict],
) -> dict:
    """선발투수 시즌 기록에서 5개 stats 추출 (v4 호환)."""
    if not basic_rec:
        return dict(LEAGUE_AVG_SP)

    k9 = 7.0
    bb9 = 3.5
    if deepen_rec:
        k9 = float(deepen_rec.get("K9") or 7.0)
        bb9 = float(deepen_rec.get("BB9") or 3.5)
    else:
        ip = float(basic_rec.get("IP") or 1)
        if ip > 0:
            k9 = float(basic_rec.get("SO") or 0) / ip * 9
            bb9 = float(basic_rec.get("BB") or 0) / ip * 9

    return {
        "era": float(basic_rec.get("ERA") or 4.50),
        "fip": float(basic_rec.get("FIP") or 4.50),
        "k9": k9,
        "bb9": bb9,
        "war": float(basic_rec.get("WAR") or 0.0),
    }


def extract_sp_stats_v5(
    basic_rec: Optional[dict],
    deepen_rec: Optional[dict],
) -> dict:
    """선발투수 시즌 기록에서 7개 stats 추출 (v5 — OBP/SLG against 추가)."""
    base = extract_sp_stats(basic_rec, deepen_rec)

    obp_against = 0.330
    slg_against = 0.400
    if deepen_rec:
        obp_against = float(deepen_rec.get("OBP") or 0.330)
        slg_against = float(deepen_rec.get("SLG") or 0.400)

    base["obp_against"] = obp_against
    base["slg_against"] = slg_against
    return base


# ── 인코딩 함수 ──

def encode_weather(code) -> list[float]:
    """날씨 코드 → [clear, overcast, rain] one-hot (3개)."""
    label = WEATHER_CODES.get(code, "clear")
    return [
        1.0 if label == "clear" else 0.0,
        1.0 if label == "overcast" else 0.0,
        1.0 if label == "rain" else 0.0,
    ]


def encode_wind_direction(code) -> list[float]:
    """풍향 코드 → (sin, cos) 주기 인코딩 (2개)."""
    degrees = WIND_DIR_CODES.get(code)
    if degrees is None:
        return [0.0, 1.0]  # 기본: 북풍 (0°)
    rad = math.radians(degrees)
    return [math.sin(rad), math.cos(rad)]


# ── 집약 함수 (v4 전용) ──

def aggregate_batting(batters: list[dict], platoon_adv: float = 0.5) -> dict:
    """9명 타자 스탯 → 6개 집약 피처.

    Args:
        batters: 길이 9 리스트, 각 원소 = extract_batter_stats 결과
        platoon_adv: 플래툰 어드밴티지 비율 (0~1)
    """
    # 9명 보장
    while len(batters) < 9:
        batters.append(dict(LEAGUE_AVG_BATTER))

    top5 = batters[:5]
    bot4 = batters[5:9]

    top5_wrc = float(np.mean([b["wrcplus"] for b in top5]))
    bot4_wrc = float(np.mean([b["wrcplus"] for b in bot4]))
    team_obp = float(np.mean([b["obp"] for b in batters]))
    team_slg = float(np.mean([b["slg"] for b in batters]))

    avg_bb = float(np.mean([b["bb_rate"] for b in batters]))
    avg_k = float(np.mean([b["k_rate"] for b in batters]))
    bb_k = avg_bb / avg_k if avg_k > 0 else 0.5

    return {
        "top5_wrc": top5_wrc,
        "bot4_wrc": bot4_wrc,
        "team_obp": team_obp,
        "team_slg": team_slg,
        "bb_k_ratio": bb_k,
        "platoon_adv": platoon_adv,
    }


def compute_platoon_advantage(
    batter_hands: list[int],
    opposing_sp_throw: int,
) -> float:
    """플래툰 어드밴티지 비율 계산.

    Args:
        batter_hands: 타자 9명의 p_bat 값 (1=좌타, 2=우타, 3=스위치)
        opposing_sp_throw: 상대 선발투수 p_throw (1=우투, 3=좌투)

    Returns:
        0~1 사이 비율 (어드밴티지 가진 타자 비율)
    """
    if not batter_hands:
        return 0.5

    advantage_count = 0
    for p_bat in batter_hands:
        if p_bat == 3:  # 스위치 히터 — 항상 어드밴티지
            advantage_count += 1
        elif opposing_sp_throw == 1 and p_bat == 1:  # 우투 vs 좌타
            advantage_count += 1
        elif opposing_sp_throw == 3 and p_bat == 2:  # 좌투 vs 우타
            advantage_count += 1

    return advantage_count / len(batter_hands)


def build_game_features(
    # 타선 집계
    home_batting: dict,       # aggregate_batting 결과
    away_batting: dict,
    # 선발투수
    home_sp: dict,
    away_sp: dict,
    # 팀 컨텍스트
    home_elo: float, away_elo: float,
    home_pyth: float, away_pyth: float,
    home_recent_wpct: float, away_recent_wpct: float,
    home_rest: float, away_rest: float,
    # 시퀀스
    home_scoring_trend: float = 0.0,
    away_scoring_trend: float = 0.0,
    home_lineup_wrc_delta: float = 0.0,
    away_lineup_wrc_delta: float = 0.0,
    # 환경
    park_factor: float = 1.0,
    temperature: float = 15.0,
    is_night: float = 1.0,
) -> list[float]:
    """37개 피처 벡터 생성."""
    row = []

    # 타선 집계 (12)
    for batting in [home_batting, away_batting]:
        for stat in BATTING_AGG_STATS:
            row.append(float(batting.get(stat, 0.0)))

    # 선발투수 (10)
    for sp in [home_sp, away_sp]:
        for stat in SP_STATS:
            row.append(float(sp.get(stat, LEAGUE_AVG_SP.get(stat, 0.0))))

    # 팀 컨텍스트 (8)
    row.extend([
        home_elo, home_pyth, home_recent_wpct, home_rest,
        away_elo, away_pyth, away_recent_wpct, away_rest,
    ])

    # 시퀀스 (4)
    row.extend([
        home_scoring_trend, home_lineup_wrc_delta,
        away_scoring_trend, away_lineup_wrc_delta,
    ])

    # 환경 (3)
    row.extend([park_factor, temperature, is_night])

    return row
