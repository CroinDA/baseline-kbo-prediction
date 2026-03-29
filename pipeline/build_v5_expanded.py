"""
v5 학습 데이터셋 구축 — 221개 피처.

v2 159 피처 (개인 타자 7×9×2 + SP 7×2 + 팀 5×2 + 시퀀스 4 + 환경 5)
+ situational 15 피처 (H2H, DAGN, travel, bp_consec, recent5, sp_vs_opp)
+ expanded 47 피처:
  - platoon_adv 2 (v4에서 복원)
  - weather one-hot 3 (clear, overcast, rain)
  - wind_dir sin/cos 2
  - batter wOBA 18 (9×2)
  - batter BABIP 18 (9×2)
  - pitcher OBP/SLG against 4 (2×2)

핵심 원칙: 각 경기의 피처는 해당 경기 이전 데이터만 사용 (미래 정보 누출 금지).
"""
import sys
import json
import math
import logging
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.constants import (
    TEAM_CODES,
    LEAGUE_REGULAR,
    PYTHAGOREAN_EXPONENT,
    NIGHT_GAME_HOUR,
    STADIUM_COORDS,
    BULLPEN_LOAD_DAYS,
)
from elo.engine import EloEngine
from features.expanded import (
    feature_names_v5,
    extract_batter_stats_v5,
    extract_sp_stats_v5,
    extract_sp_stats,
    compute_platoon_advantage,
    encode_weather,
    encode_wind_direction,
    LEAGUE_AVG_BATTER_V5,
    LEAGUE_AVG_BATTER,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

LEAGUE_AVG_SP_V2 = {
    "era": 4.50, "fip": 4.50, "whip": 1.40, "k9": 7.0,
    "bb9": 3.5, "hr9": 1.0, "war": 1.0,
}


# ── 데이터 로드 ──

def _parse_schedules(years: list[int]) -> list[dict]:
    """완료된 정규시즌 경기만 시간순 정렬하여 반환."""
    all_games = []
    for year in years:
        f = DATA_DIR / f"schedules_{year}.json"
        if not f.exists():
            logger.warning("일정 파일 없음: %s", f)
            continue
        games = json.loads(f.read_text())
        for g in games:
            if g.get("leagueType") != LEAGUE_REGULAR:
                continue
            if g.get("homeScore") is not None and g.get("awayScore") is not None:
                g["_year"] = int(g.get("year", year))
                g["_sort_key"] = g.get("gameDate", 0)
                gd = g.get("gameDate", 0)
                g["_date"] = datetime.fromtimestamp(gd).date() if gd > 0 else None
                all_games.append(g)

    all_games.sort(key=lambda x: (x["_year"], x["_sort_key"], x.get("s_no", 0)))
    logger.info("완료된 정규시즌 경기 총 %d개", len(all_games))
    return all_games


def _find_year_record(records: list[dict], year: int) -> Optional[dict]:
    for rec in records:
        rec_year = rec.get("year")
        if rec_year is not None and str(rec_year) == str(year):
            return rec
    return None


def _load_lineup_db(years: list[int]) -> dict:
    lineup_db = {}
    for year in years:
        f = DATA_DIR / f"lineups_{year}.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        for s_no_str, resp in data.items():
            teams = {}
            for key, val in resp.items():
                if isinstance(val, list) and val:
                    teams[int(key)] = val
            if teams:
                lineup_db[int(s_no_str)] = teams
    logger.info("라인업 데이터 로드: %d경기", len(lineup_db))
    return lineup_db


def _load_batter_db(years: list[int]) -> dict:
    """타자별 시즌 basic + deepen 로드 → {(p_no, year): {"basic": rec, "deepen": rec}}"""
    batter_db = {}
    for year in years:
        f = DATA_DIR / f"batter_seasons_{year}.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        for p_no_str, resp in data.items():
            p_no = int(p_no_str)
            basic_list = resp.get("basic", {}).get("list", [])
            deepen_list = resp.get("deepen", {}).get("list", [])
            basic_rec = _find_year_record(basic_list, year)
            deepen_rec = _find_year_record(deepen_list, year)
            if basic_rec:
                batter_db[(p_no, year)] = {
                    "basic": basic_rec,
                    "deepen": deepen_rec,
                }
    logger.info("타자 데이터 로드 (basic+deepen): %d명-시즌", len(batter_db))
    return batter_db


def _load_pitcher_db(years: list[int]) -> dict:
    pitcher_db = {}
    for year in years:
        f = DATA_DIR / f"pitcher_seasons_{year}.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        for p_no_str, resp in data.items():
            p_no = int(p_no_str)
            basic_list = resp.get("basic", {}).get("list", [])
            deepen_list = resp.get("deepen", {}).get("list", [])
            basic_rec = _find_year_record(basic_list, year)
            deepen_rec = _find_year_record(deepen_list, year)
            if basic_rec:
                pitcher_db[(p_no, year)] = {
                    "basic": basic_rec,
                    "deepen": deepen_rec,
                }
    logger.info("투수 데이터 로드: %d명-시즌", len(pitcher_db))
    return pitcher_db


# ── Park Factor ──

def _compute_park_factors(years: list[int]) -> dict:
    park_runs = defaultdict(list)
    for year in years:
        f = DATA_DIR / f"boxscores_{year}.json"
        if not f.exists():
            continue
        boxes = json.loads(f.read_text())
        for b in boxes:
            gi = b.get("gameInfo", {})
            s_code = gi.get("s_code")
            hs = gi.get("homeScore")
            aws = gi.get("awayScore")
            lt = gi.get("leagueType")
            if s_code and hs is not None and aws is not None and lt == LEAGUE_REGULAR:
                park_runs[s_code].append(hs + aws)

    all_runs = []
    for runs in park_runs.values():
        all_runs.extend(runs)
    league_avg = np.mean(all_runs) if all_runs else 9.5

    park_factors = {}
    for s_code, runs in park_runs.items():
        if len(runs) >= 10:
            park_factors[s_code] = np.mean(runs) / league_avg
    logger.info("Park Factor 계산: %d구장", len(park_factors))
    return park_factors


# ── 유틸리티 ──

def _pythagorean(rs: float, ra: float) -> float:
    if rs <= 0 and ra <= 0:
        return 0.5
    rs_e = rs ** PYTHAGOREAN_EXPONENT
    ra_e = ra ** PYTHAGOREAN_EXPONENT
    denom = rs_e + ra_e
    return rs_e / denom if denom > 0 else 0.5


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _get_stadium_distance(s_code_from: int, s_code_to: int) -> float:
    c1 = STADIUM_COORDS.get(s_code_from)
    c2 = STADIUM_COORDS.get(s_code_to)
    if not c1 or not c2:
        return 0.0
    return _haversine(c1[0], c1[1], c2[0], c2[1])


def _compute_bp_load(team_code, game_date, team_game_dates) -> float:
    dates = team_game_dates.get(team_code, set())
    if not game_date:
        return 0.0
    count = 0
    for d in range(1, BULLPEN_LOAD_DAYS + 1):
        if (game_date - timedelta(days=d)) in dates:
            count += 1
    return float(count * 3)


# ── 개별 타자 스탯 추출 (v5 — 9개 stats) ──

def _get_individual_batters_v5(
    s_no: int,
    team_code: int,
    opposing_team_code: int,
    year: int,
    lineup_db: dict,
    batter_db: dict,
) -> tuple[list[dict], list[int], Optional[int]]:
    """경기 라인업에서 9명 개인 타자 스탯(v5) + p_bat + 상대SP p_throw 반환.

    Returns:
        (batters_v5, batter_hands, opposing_sp_throw)
    """
    default_batters = [dict(LEAGUE_AVG_BATTER_V5) for _ in range(9)]
    default_hands = [2] * 9

    game_lineups = lineup_db.get(s_no)
    if not game_lineups:
        return default_batters, default_hands, None

    players = game_lineups.get(team_code)
    if not players:
        return default_batters, default_hands, None

    # 상대팀 라인업에서 SP의 p_throw 찾기
    opposing_sp_throw = None
    opposing_players = game_lineups.get(opposing_team_code, [])
    for p in opposing_players:
        if str(p.get("battingOrder", "")) == "P":
            opposing_sp_throw = p.get("p_throw")
            break

    order_map = {}
    hand_map = {}
    for p in players:
        p_no = p.get("p_no")
        order_raw = p.get("battingOrder", "0")
        if not p_no:
            continue
        try:
            order = int(order_raw)
        except (ValueError, TypeError):
            continue
        if 1 <= order <= 9:
            batter_data = batter_db.get((int(p_no), year))
            if batter_data:
                order_map[order] = extract_batter_stats_v5(
                    batter_data["basic"], batter_data.get("deepen"),
                )
            else:
                order_map[order] = dict(LEAGUE_AVG_BATTER_V5)
            hand_map[order] = p.get("p_bat", 2)

    batters = [order_map.get(i, dict(LEAGUE_AVG_BATTER_V5)) for i in range(1, 10)]
    hands = [hand_map.get(i, 2) for i in range(1, 10)]
    return batters, hands, opposing_sp_throw


def _get_sp_full_stats_v5(p_no, year: int, pitcher_db: dict) -> dict:
    """선발투수 9개 스탯 (v5 — whip, hr9 + obp_against, slg_against)."""
    if not p_no:
        return {**LEAGUE_AVG_SP_V2, "obp_against": 0.330, "slg_against": 0.400}
    data = pitcher_db.get((p_no, year))
    if not data:
        return {**LEAGUE_AVG_SP_V2, "obp_against": 0.330, "slg_against": 0.400}

    basic_rec = data["basic"]
    deepen_rec = data.get("deepen")

    # v5 base: era, fip, k9, bb9, war + obp_against, slg_against
    stats_v5 = extract_sp_stats_v5(basic_rec, deepen_rec)

    # v2 추가: whip, hr9
    stats_v5["whip"] = float(basic_rec.get("WHIP") or 1.40)
    ip = float(basic_rec.get("IP") or 1)
    stats_v5["hr9"] = float(basic_rec.get("HR") or 0) / ip * 9 if ip > 0 else 1.0
    return stats_v5


# ── 신규 피처 계산 함수 (situational) ──

def _compute_h2h(home, away, h2h_tracker):
    key = frozenset({home, away})
    rec = h2h_tracker.get(key)
    if not rec or rec["total"] == 0:
        return 0.5, 0.5, 0.0
    home_wpct = rec.get(home, 0) / rec["total"]
    away_wpct = rec.get(away, 0) / rec["total"]
    return home_wpct, away_wpct, float(rec["total"])


def _compute_dagn(team_code, game_date, game_hour, team_prev_game):
    if game_hour >= NIGHT_GAME_HOUR:
        return 0.0
    prev = team_prev_game.get(team_code)
    if not prev:
        return 0.0
    prev_date, prev_hour = prev
    if game_date and prev_date:
        delta = (game_date - prev_date).days
        if delta == 1 and prev_hour >= NIGHT_GAME_HOUR:
            return 1.0
    return 0.0


def _compute_bp_consecutive(team_code, game_date, team_game_dates):
    dates = team_game_dates.get(team_code, set())
    if not dates or not game_date:
        return 0.0
    consec = 0
    check_date = game_date - timedelta(days=1)
    for _ in range(5):
        if check_date in dates:
            consec += 1
            check_date -= timedelta(days=1)
        else:
            break
    return float(min(consec, 3))


def _compute_recent5(team_code, team_results, team_rs_history):
    results = team_results.get(team_code, [])
    rs_hist = team_rs_history.get(team_code, [])
    if len(results) < 5:
        wpct = sum(results) / len(results) if results else 0.5
    else:
        wpct = sum(results[-5:]) / 5.0
    if len(rs_hist) < 5:
        rs_avg = np.mean(rs_hist) if rs_hist else 4.5
    else:
        rs_avg = np.mean(rs_hist[-5:])
    return float(wpct), float(rs_avg)


def _compute_sp_vs_opp(sp_no, opp_team, sp_vs_team_tracker):
    if not sp_no:
        return 0.5
    key = (sp_no, opp_team)
    rec = sp_vs_team_tracker.get(key)
    if not rec or rec["total"] == 0:
        return 0.5
    return rec["wins"] / rec["total"]


# ── 메인 빌드 ──

def build_dataset_v5(
    years: list[int] = None,
    output_name: str = "training_data_v5",
) -> pd.DataFrame:
    """v5 학습 데이터셋 구축 (221개 피처).

    Returns:
        학습 데이터프레임 (N rows × 222 cols: 221 features + label)
    """
    if years is None:
        years = [2023, 2024, 2025]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # ── 데이터 로드 ──
    all_games = _parse_schedules(years)
    if not all_games:
        logger.error("경기 데이터 없음.")
        return pd.DataFrame()

    lineup_db = _load_lineup_db(years)
    batter_db = _load_batter_db(years)
    pitcher_db = _load_pitcher_db(years)
    park_factors = _compute_park_factors(years)

    # ── Elo 엔진 ──
    elo = EloEngine()

    # ── 팀별 누적 통계 ──
    team_runs_scored = defaultdict(float)
    team_runs_allowed = defaultdict(float)
    team_games = defaultdict(int)
    team_results = defaultdict(list)
    team_rs_history = defaultdict(list)
    team_lineup_wrc_history = defaultdict(list)
    team_last_game_date = {}
    team_prev_game = {}
    team_game_dates = defaultdict(set)
    team_prev_s_code = {}

    # ── H2H / SP vs Team 추적 ──
    h2h_tracker = defaultdict(lambda: defaultdict(int))
    sp_vs_team_tracker = defaultdict(lambda: {"wins": 0, "total": 0})

    prev_year = None
    rows = []
    feat_names = feature_names_v5()

    for i, game in enumerate(all_games):
        year = game["_year"]
        home = game["homeTeam"]
        away = game["awayTeam"]
        home_score = game["homeScore"]
        away_score = game["awayScore"]
        home_sp_no = game.get("homeSP")
        away_sp_no = game.get("awaySP")
        game_date = game.get("_date")
        s_code = game.get("s_code")

        hm = str(game.get("hm", "18:00:00"))
        try:
            game_hour = int(hm.split(":")[0]) if ":" in hm else int(hm[:2])
        except (ValueError, IndexError):
            game_hour = 18

        # 시즌 전환
        if prev_year is not None and year != prev_year:
            logger.info("시즌 전환: %d → %d", prev_year, year)
            elo.new_season()
            team_runs_scored.clear()
            team_runs_allowed.clear()
            team_games.clear()
            team_results.clear()
            team_rs_history.clear()
            team_lineup_wrc_history.clear()
            team_last_game_date.clear()
            team_prev_game.clear()
            team_game_dates.clear()
            team_prev_s_code.clear()
        prev_year = year

        s_no = game.get("s_no")

        # ━━━ 1. 개인 타자 (9×9×2 = 162 values, v2용 7개 + v5용 2개) ━━━
        home_batters, home_hands, away_sp_throw = _get_individual_batters_v5(
            s_no, home, away, year, lineup_db, batter_db,
        )
        away_batters, away_hands, home_sp_throw = _get_individual_batters_v5(
            s_no, away, home, year, lineup_db, batter_db,
        )

        # ━━━ 2. SP (9개 × 2 = 18 values, v2용 7개 + v5용 2개) ━━━
        home_sp_stats = _get_sp_full_stats_v5(home_sp_no, year, pitcher_db)
        away_sp_stats = _get_sp_full_stats_v5(away_sp_no, year, pitcher_db)

        # SP Elo 보정
        if home_sp_no:
            elo.update_sp_rating(home_sp_no, home, home_sp_stats["fip"])
        if away_sp_no:
            elo.update_sp_rating(away_sp_no, away, away_sp_stats["fip"])
        elo.update_team_sp_avg(home, 4.20)
        elo.update_team_sp_avg(away, 4.20)

        # ━━━ 3. 팀 컨텍스트 (5×2 = 10) ━━━
        home_elo_val = elo.get_rating(home)
        away_elo_val = elo.get_rating(away)

        h_games = max(team_games[home], 1)
        a_games = max(team_games[away], 1)
        home_rs_avg = team_runs_scored[home] / h_games if h_games > 0 else 4.5
        home_ra_avg = team_runs_allowed[home] / h_games if h_games > 0 else 4.5
        away_rs_avg = team_runs_scored[away] / a_games if a_games > 0 else 4.5
        away_ra_avg = team_runs_allowed[away] / a_games if a_games > 0 else 4.5

        home_pyth = _pythagorean(home_rs_avg * 9, home_ra_avg * 9)
        away_pyth = _pythagorean(away_rs_avg * 9, away_ra_avg * 9)

        home_wpct = sum(team_results.get(home, [])) / len(team_results[home]) if team_results.get(home) else 0.5
        away_wpct = sum(team_results.get(away, [])) / len(team_results[away]) if team_results.get(away) else 0.5

        home_bp_load = _compute_bp_load(home, game_date, team_game_dates)
        away_bp_load = _compute_bp_load(away, game_date, team_game_dates)

        if game_date and home in team_last_game_date:
            home_rest = float((game_date - team_last_game_date[home]).days)
        else:
            home_rest = 3.0
        if game_date and away in team_last_game_date:
            away_rest = float((game_date - team_last_game_date[away]).days)
        else:
            away_rest = 3.0

        # ━━━ 4. 시퀀스 (2×2 = 4) ━━━
        home_today_wrc = float(np.mean([b["wrcplus"] for b in home_batters]))
        away_today_wrc = float(np.mean([b["wrcplus"] for b in away_batters]))

        home_rs_hist = team_rs_history.get(home, [])
        home_trend = float(np.mean(home_rs_hist[-5:]) - team_runs_scored[home] / team_games[home]) if len(home_rs_hist) >= 5 and team_games[home] > 0 else 0.0

        away_rs_hist = team_rs_history.get(away, [])
        away_trend = float(np.mean(away_rs_hist[-5:]) - team_runs_scored[away] / team_games[away]) if len(away_rs_hist) >= 5 and team_games[away] > 0 else 0.0

        home_wrc_hist = team_lineup_wrc_history.get(home, [])
        home_wrc_delta = float(home_today_wrc - np.mean(home_wrc_hist)) if home_wrc_hist else 0.0
        away_wrc_hist = team_lineup_wrc_history.get(away, [])
        away_wrc_delta = float(away_today_wrc - np.mean(away_wrc_hist)) if away_wrc_hist else 0.0

        # ━━━ 5. 환경 (5개) ━━━
        temp = game.get("temperature") or 15.0
        if temp == 0 or temp > 45 or temp < -15:
            temp = 15.0
        temp = max(-10.0, min(40.0, float(temp)))

        humidity = float(game.get("humidity") or 50.0)
        humidity = max(0.0, min(100.0, humidity))

        wind_speed = float(game.get("windSpeed") or 0.0)
        wind_speed = max(0.0, min(30.0, wind_speed))

        is_night = 1.0 if game_hour >= NIGHT_GAME_HOUR else 0.0
        pf = park_factors.get(s_code, 1.0)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # v2 159 피처 벡터 조립
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        row = []

        # 개인 타자: 7 stats × 9 × 2 = 126
        batter_stats_order = ["wrcplus", "avg", "obp", "slg", "hr_rate", "bb_rate", "k_rate"]
        for batters in [home_batters, away_batters]:
            for b in batters:
                for stat in batter_stats_order:
                    row.append(float(b.get(stat, LEAGUE_AVG_BATTER.get(stat, 0.0))))

        # SP: 7 × 2 = 14
        sp_stats_order = ["era", "fip", "whip", "k9", "bb9", "hr9", "war"]
        for sp in [home_sp_stats, away_sp_stats]:
            for stat in sp_stats_order:
                row.append(float(sp.get(stat, LEAGUE_AVG_SP_V2.get(stat, 0.0))))

        # 팀: 5 × 2 = 10
        row.extend([
            home_elo_val, home_pyth, home_wpct, home_bp_load, home_rest,
            away_elo_val, away_pyth, away_wpct, away_bp_load, away_rest,
        ])

        # 시퀀스: 4
        row.extend([home_trend, home_wrc_delta, away_trend, away_wrc_delta])

        # 환경: 5
        row.extend([temp, humidity, wind_speed, is_night, pf])

        assert len(row) == 159, f"v2 part: expected 159, got {len(row)}"

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # situational 15 피처
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        # H2H (3)
        h2h_home, h2h_away, h2h_n = _compute_h2h(home, away, h2h_tracker)
        row.extend([h2h_home, h2h_away, h2h_n])

        # DAGN (2)
        dagn_home = _compute_dagn(home, game_date, game_hour, team_prev_game)
        dagn_away = _compute_dagn(away, game_date, game_hour, team_prev_game)
        row.extend([dagn_home, dagn_away])

        # 이동거리 (2)
        travel_home = _get_stadium_distance(team_prev_s_code.get(home, s_code), s_code) if s_code else 0.0
        travel_away = _get_stadium_distance(team_prev_s_code.get(away, s_code), s_code) if s_code else 0.0
        row.extend([travel_home, travel_away])

        # 불펜 연투 (2)
        bp_consec_home = _compute_bp_consecutive(home, game_date, team_game_dates)
        bp_consec_away = _compute_bp_consecutive(away, game_date, team_game_dates)
        row.extend([bp_consec_home, bp_consec_away])

        # 최근 5경기 폼 (4)
        r5_wpct_home, r5_rs_home = _compute_recent5(home, team_results, team_rs_history)
        r5_wpct_away, r5_rs_away = _compute_recent5(away, team_results, team_rs_history)
        row.extend([r5_wpct_home, r5_wpct_away, r5_rs_home, r5_rs_away])

        # SP vs 상대팀 (2)
        sp_vs_home = _compute_sp_vs_opp(home_sp_no, away, sp_vs_team_tracker)
        sp_vs_away = _compute_sp_vs_opp(away_sp_no, home, sp_vs_team_tracker)
        row.extend([sp_vs_home, sp_vs_away])

        assert len(row) == 174, f"situational part: expected 174, got {len(row)}"

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # expanded 47 피처 (v5 신규)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        # platoon_adv (2) — v4에서 복원
        if away_sp_throw is not None:
            platoon_home = compute_platoon_advantage(home_hands, away_sp_throw)
        else:
            platoon_home = 0.5
        if home_sp_throw is not None:
            platoon_away = compute_platoon_advantage(away_hands, home_sp_throw)
        else:
            platoon_away = 0.5
        row.extend([platoon_home, platoon_away])

        # weather one-hot (3)
        weather_code = game.get("weather")
        row.extend(encode_weather(weather_code))

        # wind direction sin/cos (2)
        wind_dir_code = game.get("windDirection")
        row.extend(encode_wind_direction(wind_dir_code))

        # batter wOBA (18) — home 9 + away 9
        for batters in [home_batters, away_batters]:
            for b in batters:
                row.append(float(b.get("woba", 0.320)))

        # batter BABIP (18) — home 9 + away 9
        for batters in [home_batters, away_batters]:
            for b in batters:
                row.append(float(b.get("babip", 0.300)))

        # pitcher OBP/SLG against (4)
        row.append(float(home_sp_stats.get("obp_against", 0.330)))
        row.append(float(home_sp_stats.get("slg_against", 0.400)))
        row.append(float(away_sp_stats.get("obp_against", 0.330)))
        row.append(float(away_sp_stats.get("slg_against", 0.400)))

        assert len(row) == 221, f"v5 total: expected 221, got {len(row)}"

        # 라벨
        if home_score > away_score:
            label = 1
        elif home_score < away_score:
            label = 0
        else:
            _update_state_after_game(
                home, away, home_score, away_score,
                home_sp_no, away_sp_no, game_date, game_hour, s_code,
                home_today_wrc, away_today_wrc,
                elo, team_runs_scored, team_runs_allowed, team_games,
                team_results, team_rs_history, team_lineup_wrc_history,
                team_last_game_date, team_prev_game, team_game_dates,
                team_prev_s_code, h2h_tracker, sp_vs_team_tracker,
            )
            continue

        rows.append(row + [label])

        # ── 경기 결과 반영 ──
        _update_state_after_game(
            home, away, home_score, away_score,
            home_sp_no, away_sp_no, game_date, game_hour, s_code,
            home_today_wrc, away_today_wrc,
            elo, team_runs_scored, team_runs_allowed, team_games,
            team_results, team_rs_history, team_lineup_wrc_history,
            team_last_game_date, team_prev_game, team_game_dates,
            team_prev_s_code, h2h_tracker, sp_vs_team_tracker,
        )

        if (i + 1) % 500 == 0:
            logger.info("  진행: %d/%d 경기 처리", i + 1, len(all_games))

    # ── DataFrame 생성 ──
    columns = feat_names + ["label"]
    df = pd.DataFrame(rows, columns=columns)

    logger.info("데이터셋 생성 완료: %d행 × %d열", len(df), len(df.columns))
    logger.info("피처 수: %d개", len(feat_names))
    logger.info("홈팀 승률: %.1f%%", df["label"].mean() * 100)

    # ── NaN 처리 ──
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        logger.warning("NaN %d개 발견 → 0.0으로 대체", nan_count)
        df = df.fillna(0.0)

    # ── 저장 ──
    csv_path = DATA_DIR / f"{output_name}.csv"
    df.to_csv(csv_path, index=False)
    logger.info("저장: %s", csv_path)

    try:
        parquet_path = DATA_DIR / f"{output_name}.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info("저장: %s", parquet_path)
    except Exception:
        logger.info("Parquet 저장 스킵 (pyarrow 없음)")

    elo.save()
    logger.info("Elo 최종 상태 저장")

    return df


def _update_state_after_game(
    home, away, home_score, away_score,
    home_sp_no, away_sp_no, game_date, game_hour, s_code,
    home_today_wrc, away_today_wrc,
    elo, team_runs_scored, team_runs_allowed, team_games,
    team_results, team_rs_history, team_lineup_wrc_history,
    team_last_game_date, team_prev_game, team_game_dates,
    team_prev_s_code, h2h_tracker, sp_vs_team_tracker,
):
    """경기 결과를 모든 추적 상태에 반영."""
    elo.update(home, away, home_score, away_score, home_sp_no, away_sp_no)

    team_runs_scored[home] += home_score
    team_runs_allowed[home] += away_score
    team_runs_scored[away] += away_score
    team_runs_allowed[away] += home_score
    team_games[home] += 1
    team_games[away] += 1

    home_won = home_score > away_score
    team_results[home].append(home_won)
    team_results[away].append(not home_won)

    team_rs_history[home].append(float(home_score))
    team_rs_history[away].append(float(away_score))

    team_lineup_wrc_history[home].append(float(home_today_wrc))
    team_lineup_wrc_history[away].append(float(away_today_wrc))

    if game_date:
        team_last_game_date[home] = game_date
        team_last_game_date[away] = game_date
        team_prev_game[home] = (game_date, game_hour)
        team_prev_game[away] = (game_date, game_hour)
        team_game_dates[home].add(game_date)
        team_game_dates[away].add(game_date)

    if s_code:
        team_prev_s_code[home] = s_code
        team_prev_s_code[away] = s_code

    # H2H 갱신
    h2h_key = frozenset({home, away})
    if home_score != away_score:
        winner = home if home_won else away
        h2h_tracker[h2h_key][winner] += 1
        h2h_tracker[h2h_key]["total"] += 1

    # SP vs 상대팀 갱신
    if home_sp_no and home_score != away_score:
        sp_key = (home_sp_no, away)
        sp_vs_team_tracker[sp_key]["total"] += 1
        if home_won:
            sp_vs_team_tracker[sp_key]["wins"] += 1
    if away_sp_no and home_score != away_score:
        sp_key = (away_sp_no, home)
        sp_vs_team_tracker[sp_key]["total"] += 1
        if not home_won:
            sp_vs_team_tracker[sp_key]["wins"] += 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="v5 학습 데이터셋 구축 (221 피처)")
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    parser.add_argument("--output", type=str, default="training_data_v5")
    args = parser.parse_args()

    build_dataset_v5(years=args.years, output_name=args.output)
