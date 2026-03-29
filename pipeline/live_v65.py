"""
v65 live 피처 빌더.

v64 기반 시점누적/로스터/불펜 집계 피처의 실전용 경로.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

from config.constants import LEAGUE_REGULAR, NIGHT_GAME_HOUR
from data.collector import get_game_lineup, get_player_day, get_player_season
from elo.engine import EloEngine
from pipeline import daily_run as dr
from pipeline.build_v6_timeaware import (
    LEAGUE_AVG_BATTER,
    LEAGUE_AVG_SP,
    _batter_snapshot,
    _build_availability_features,
    _build_batter_prior_cache,
    _build_bench_features,
    _build_bullpen_features,
    _build_pitcher_prior_cache,
    _compute_bp_load,
    _empty_batter_counts,
    _empty_pitcher_counts,
    _extract_batter_prior_counts,
    _extract_pitcher_prior_counts,
    _load_batter_summary_db,
    _load_pitcher_summary_db,
    _pitcher_snapshot,
    _pythagorean,
    _safe_float,
    feature_names_v6,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
LIVE_VARIANT = "v65"
LIVE_INCLUDE_BENCH = False
LIVE_INCLUDE_SP_WAR = True
LIVE_STABILIZE_RATES = True
LIVE_TUNE_BULLPEN = True

_resource_cache: dict | None = None


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _save_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _find_year_record(records: list[dict], year: int):
    for rec in records:
        rec_year = rec.get("year")
        if rec_year is not None and str(rec_year) == str(year):
            return rec
    return None


def _ensure_resources(target_year: int) -> dict:
    global _resource_cache
    if _resource_cache is not None and _resource_cache.get("target_year") == target_year:
        return _resource_cache

    prior_years = [2023, 2024, 2025]
    batter_summary_db = _load_batter_summary_db(prior_years)
    pitcher_summary_db = _load_pitcher_summary_db(prior_years)
    prior_cache_years = sorted(set(prior_years + [target_year]))
    rosters = _load_json(DATA_DIR / f"rosters_{target_year}.json", {})

    _resource_cache = {
        "target_year": target_year,
        "batter_prior_cache": _build_batter_prior_cache(prior_cache_years, batter_summary_db),
        "pitcher_prior_cache": _build_pitcher_prior_cache(prior_cache_years, pitcher_summary_db),
        "batter_player_ids": {p_no for p_no, _ in batter_summary_db.keys()},
        "pitcher_player_ids": {p_no for p_no, _ in pitcher_summary_db.keys()},
        "rosters": rosters,
    }
    return _resource_cache


def _season_cache_path(mode: str, year: int) -> Path:
    suffix = "full"
    prefix = "batter" if mode == "batting" else "pitcher"
    return DATA_DIR / f"{prefix}_seasons_{suffix}_{year}.json"


def _day_cache_path(mode: str, year: int) -> Path:
    prefix = "batter" if mode == "batting" else "pitcher"
    return DATA_DIR / f"{prefix}_days_{year}.json"


def _filter_year_records(resp: dict, year: int) -> dict:
    filtered = {}
    for section in ["basic", "deepen", "fielding"]:
        section_data = resp.get(section)
        if not section_data:
            continue
        if isinstance(section_data, dict) and "list" in section_data:
            year_records = [r for r in section_data["list"] if str(r.get("year")) == str(year)]
            if year_records:
                filtered[section] = {"list": year_records}
        else:
            filtered[section] = section_data
    filtered["result_cd"] = resp.get("result_cd", 100)
    filtered["result_msg"] = resp.get("result_msg", "Success")
    return filtered


def _ensure_player_season_cached(p_no: int, mode: str, year: int, league_type: int) -> dict:
    path = _season_cache_path(mode, year)
    data = _load_json(path, {})
    p_key = str(int(p_no))
    if p_key in data:
        return data[p_key]

    resp = get_player_season(p_no, m2=mode, year=year, league_type=league_type)
    filtered = _filter_year_records(resp, year)
    data[p_key] = filtered
    _save_json(path, data)
    return filtered


def _ensure_player_day_cached(p_no: int, mode: str, year: int, month: int) -> dict:
    path = _day_cache_path(mode, year)
    data = _load_json(path, {})
    p_key = str(int(p_no))
    month_key = f"{month:02d}"
    player_months = data.get(p_key, {})
    if month_key in player_months:
        return player_months[month_key]

    resp = get_player_day(p_no, m2=mode, year=year, month=month)
    player_months[month_key] = resp
    data[p_key] = player_months
    _save_json(path, data)
    return resp


def _extract_pitcher_appearance_dates(p_no: int, year: int, target_dt: datetime) -> set[date]:
    months = {target_dt.month}
    if target_dt.month > 1:
        months.add(target_dt.month - 1)

    appearance_dates = set()
    for month in sorted(months):
        resp = _ensure_player_day_cached(p_no, "pitching", year, month)
        for key, value in resp.items():
            if not (str(key).isdigit() and isinstance(value, dict)):
                continue
            mmdd = value.get("gameDate")
            if not mmdd:
                continue
            try:
                month_str, day_str = str(mmdd).split("-")
                d = date(year, int(month_str), int(day_str))
            except Exception:
                continue
            if d < target_dt.date() and _safe_float(value.get("GS")) <= 0:
                appearance_dates.add(d)
    return appearance_dates


def _build_current_batter_state(players: list[int], year: int) -> dict:
    state = defaultdict(_empty_batter_counts)
    for p_no in players:
        if not p_no:
            continue
        resp = _ensure_player_season_cached(p_no, "batting", year, LEAGUE_REGULAR)
        basic_rec = _find_year_record(resp.get("basic", {}).get("list", []), year)
        if basic_rec:
            state[p_no].update(_extract_batter_prior_counts(basic_rec))
    return state


def _build_current_pitcher_state(players: list[int], year: int, target_dt: datetime) -> dict:
    state = defaultdict(_empty_pitcher_counts)
    for p_no in players:
        if not p_no:
            continue
        resp = _ensure_player_season_cached(p_no, "pitching", year, LEAGUE_REGULAR)
        basic_rec = _find_year_record(resp.get("basic", {}).get("list", []), year)
        deepen_rec = _find_year_record(resp.get("deepen", {}).get("list", []), year)
        if basic_rec:
            state[p_no].update(_extract_pitcher_prior_counts(basic_rec, deepen_rec or {}))
            state[p_no]["appearance_dates"] = _extract_pitcher_appearance_dates(p_no, year, target_dt)
    return state


def _get_roster_players(date_str: str, team_code: int) -> list[int]:
    dr.collect_daily_roster(date_str)
    roster_file = DATA_DIR / f"rosters_{date_str[:4]}.json"
    rosters = _load_json(roster_file, {})
    players = rosters.get(date_str, [])
    return [int(rec["p_no"]) for rec in players if rec.get("t_code") == team_code and rec.get("p_no")]


def _get_previous_roster_players(resources: dict, game_date: date, team_code: int, lookback_days: int = 7) -> list[int]:
    rosters = resources.get("rosters", {})
    for days_back in range(1, lookback_days + 1):
        date_str = (game_date - timedelta(days=days_back)).isoformat()
        players = rosters.get(date_str, [])
        team_players = [int(rec["p_no"]) for rec in players if rec.get("t_code") == team_code and rec.get("p_no")]
        if team_players:
            return team_players
    return []


def _get_lineup_players(s_no: int, team_code: int) -> list[int]:
    try:
        resp = get_game_lineup(s_no)
    except Exception:
        return []
    players = resp.get(str(team_code), [])
    order_map = {}
    for rec in players:
        p_no = rec.get("p_no")
        order_raw = str(rec.get("battingOrder", ""))
        if not p_no:
            continue
        if order_raw == "P":
            continue
        try:
            order = int(order_raw)
        except (TypeError, ValueError):
            continue
        if 1 <= order <= 9:
            order_map[order] = int(p_no)
    return [order_map[i] for i in range(1, 10) if i in order_map]


def _position_group(pos: int | None) -> str:
    """포지션을 그룹으로 매핑. 멀티포지션 대체 허용."""
    if pos == 2:
        return "C"
    if pos in (3, 4, 5, 6):
        return "IF"
    if pos in (7, 8, 9):
        return "OF"
    if pos == 10:
        return "DH"
    return "UNKNOWN"


def _get_player_position(p_no: int, year: int) -> int | None:
    """batter_seasons_full에서 선수 포지션 조회. 없으면 None."""
    for y in [year, year - 1, year - 2]:
        path = DATA_DIR / f"batter_seasons_full_{y}.json"
        data = _load_json(path, {})
        resp = data.get(str(p_no), {})
        basic_list = resp.get("basic", {}).get("list", [])
        if basic_list:
            return basic_list[0].get("p_position")
    return None


def _get_player_pa(p_no: int, year: int) -> int:
    """올시즌 PA 조회. 없으면 이전 시즌."""
    for y in [year, year - 1]:
        path = DATA_DIR / f"batter_seasons_full_{y}.json"
        data = _load_json(path, {})
        resp = data.get(str(p_no), {})
        basic_list = resp.get("basic", {}).get("list", [])
        if basic_list:
            pa = basic_list[0].get("PA")
            if pa is not None:
                return int(pa)
    return 0


def _get_last_lineup_for_team(team_code: int, year: int, before_s_no: int) -> list[int]:
    """해당 팀의 직전 경기 라인업(1~9번 타순) 반환."""
    lineup_file = DATA_DIR / f"lineups_{year}.json"
    lineups = _load_json(lineup_file, {})

    # s_no 역순으로 탐색 (최신 경기부터)
    candidates = []
    for s_no_str, data in lineups.items():
        s_no = int(s_no_str)
        if s_no >= before_s_no:
            continue
        team_players = data.get(str(team_code), [])
        if isinstance(team_players, list) and team_players:
            candidates.append((s_no, team_players))

    if not candidates:
        return []

    # 가장 최근 경기
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, players = candidates[0]

    order_map = {}
    for rec in players:
        p_no = rec.get("p_no")
        order_raw = str(rec.get("battingOrder", ""))
        if not p_no or order_raw == "P":
            continue
        try:
            order = int(order_raw)
        except (TypeError, ValueError):
            continue
        if 1 <= order <= 9:
            order_map[order] = int(p_no)

    return [order_map[i] for i in range(1, 10) if i in order_map]


def _build_fallback_lineup(
    team_code: int,
    year: int,
    s_no: int,
    roster_players: list[int],
) -> list[int]:
    """직전 라인업 기반 fallback. 말소 선수는 같은 포지션 로스터 대체."""
    prev_lineup = _get_last_lineup_for_team(team_code, year, s_no)
    if not prev_lineup:
        return []

    roster_set = set(roster_players)
    result = []

    for p_no in prev_lineup:
        if p_no in roster_set:
            result.append(p_no)
        else:
            # 말소된 선수 → 같은 포지션 그룹 대체자 찾기
            pos = _get_player_position(p_no, year)
            pos_group = _position_group(pos)
            already_used = set(result) | set(prev_lineup)
            candidates = []
            for rp in roster_players:
                if rp in already_used:
                    continue
                rp_pos = _get_player_position(rp, year)
                if _position_group(rp_pos) == pos_group:
                    candidates.append((rp, rp_pos, _get_player_pa(rp, year)))

            if candidates:
                # SS(6)/2B(4) 대체 시 1B(3)는 후순위
                if pos in (4, 6):
                    preferred = [(rp, pa) for rp, rp_pos, pa in candidates if rp_pos != 3]
                    if preferred:
                        candidates = [(rp, rp_pos, pa) for rp, rp_pos, pa in candidates if rp_pos != 3]
                # PA 가장 많은 선수 = 주로 뛰는 백업
                candidates_sorted = [(rp, pa) for rp, _, pa in candidates]
                candidates_sorted.sort(key=lambda x: x[1], reverse=True)
                candidates = candidates_sorted
                result.append(candidates[0][0])
                logger.info(
                    "fallback 대체: %d(%s, 말소) → %d(PA=%d)",
                    p_no, pos_group, candidates[0][0], candidates[0][1],
                )
            else:
                # 같은 그룹 대체 불가 → 빈 슬롯 (리그 평균으로 채워짐)
                logger.warning("fallback 대체 실패: %d(%s) — 리그 평균 사용", p_no, pos_group)

    return result


def build_game_features_v65_live(
    game: dict,
    elo_engine: EloEngine,
    year: int,
    use_confirmed_lineup: bool = True,
) -> dict:
    resources = _ensure_resources(year)
    st = dr._season_state
    home = game["home_team"]
    away = game["away_team"]
    target_dt = game.get("_datetime", datetime(year, 3, 28, 18, 0))
    game_date = target_dt.date()
    date_str = game_date.isoformat()

    home_lineup = _get_lineup_players(game["s_no"], home) if use_confirmed_lineup else []
    away_lineup = _get_lineup_players(game["s_no"], away) if use_confirmed_lineup else []
    home_roster = _get_roster_players(date_str, home)
    away_roster = _get_roster_players(date_str, away)

    # 라인업 미확정 시 직전 경기 라인업 + 로스터 교차검증 fallback
    if len(home_lineup) < 9:
        fb = _build_fallback_lineup(home, year, game["s_no"], home_roster)
        if fb:
            logger.info("홈(%s) fallback 라인업 적용: %d명", home, len(fb))
            home_lineup = fb
    if len(away_lineup) < 9:
        fb = _build_fallback_lineup(away, year, game["s_no"], away_roster)
        if fb:
            logger.info("원정(%s) fallback 라인업 적용: %d명", away, len(fb))
            away_lineup = fb

    batter_players = sorted(set(home_lineup + away_lineup))
    pitcher_players = sorted(set(home_roster + away_roster + [game.get("home_sp"), game.get("away_sp")]))
    batter_state = _build_current_batter_state(batter_players, year)
    pitcher_state = _build_current_pitcher_state(pitcher_players, year, target_dt)

    row = []

    home_batters = [
        _batter_snapshot(
            p_no,
            year,
            resources["batter_prior_cache"],
            batter_state,
            stabilize_rates=LIVE_STABILIZE_RATES,
        )
        for p_no in home_lineup
    ]
    away_batters = [
        _batter_snapshot(
            p_no,
            year,
            resources["batter_prior_cache"],
            batter_state,
            stabilize_rates=LIVE_STABILIZE_RATES,
        )
        for p_no in away_lineup
    ]
    while len(home_batters) < 9:
        home_batters.append(dict(LEAGUE_AVG_BATTER))
    while len(away_batters) < 9:
        away_batters.append(dict(LEAGUE_AVG_BATTER))

    batter_stats_order = ["wrcplus", "avg", "obp", "slg", "hr_rate", "bb_rate", "k_rate", "bbk_ratio", "pa"]
    for batters in [home_batters, away_batters]:
        for b in batters:
            for stat in batter_stats_order:
                row.append(float(b.get(stat, LEAGUE_AVG_BATTER[stat])))

    home_sp_stats = _pitcher_snapshot(
        game.get("home_sp"),
        year,
        resources["pitcher_prior_cache"],
        pitcher_state,
        stabilize_rates=LIVE_STABILIZE_RATES,
    )
    away_sp_stats = _pitcher_snapshot(
        game.get("away_sp"),
        year,
        resources["pitcher_prior_cache"],
        pitcher_state,
        stabilize_rates=LIVE_STABILIZE_RATES,
    )
    if game.get("home_sp"):
        elo_engine.update_sp_rating(game["home_sp"], home, home_sp_stats["fip"])
    if game.get("away_sp"):
        elo_engine.update_sp_rating(game["away_sp"], away, away_sp_stats["fip"])
    elo_engine.update_team_sp_avg(home, 4.20)
    elo_engine.update_team_sp_avg(away, 4.20)

    for sp in [home_sp_stats, away_sp_stats]:
        for stat in ["era", "fip", "whip", "k9", "bb9", "hr9", "obp_against", "kbb_ratio", "np_per_app", "ip", "war"]:
            row.append(float(sp.get(stat, LEAGUE_AVG_SP.get(stat, 0.0))))

    h_games = max(st["team_games"].get(home, 0), 1)
    a_games = max(st["team_games"].get(away, 0), 1)
    home_rs_avg = st["team_runs_scored"].get(home, 0) / h_games
    home_ra_avg = st["team_runs_allowed"].get(home, 0) / h_games
    away_rs_avg = st["team_runs_scored"].get(away, 0) / a_games
    away_ra_avg = st["team_runs_allowed"].get(away, 0) / a_games
    home_results = st["team_results"].get(home, [])
    away_results = st["team_results"].get(away, [])
    home_wpct = sum(home_results) / len(home_results) if home_results else 0.5
    away_wpct = sum(away_results) / len(away_results) if away_results else 0.5
    home_rest = float((game_date - st["team_prev_game"][home][0]).days) if home in st["team_prev_game"] else 3.0
    away_rest = float((game_date - st["team_prev_game"][away][0]).days) if away in st["team_prev_game"] else 3.0
    row.extend([
        elo_engine.get_rating(home), _pythagorean(home_rs_avg * 9, home_ra_avg * 9), home_wpct,
        _compute_bp_load(home, game_date, st["team_game_dates"]), home_rest,
        elo_engine.get_rating(away), _pythagorean(away_rs_avg * 9, away_ra_avg * 9), away_wpct,
        _compute_bp_load(away, game_date, st["team_game_dates"]), away_rest,
    ])

    home_today_wrc = sum(b["wrcplus"] for b in home_batters) / len(home_batters) if home_batters else 100.0
    away_today_wrc = sum(b["wrcplus"] for b in away_batters) / len(away_batters) if away_batters else 100.0
    home_rs_hist = st["team_runs_history"].get(home, [])
    away_rs_hist = st["team_runs_history"].get(away, [])
    home_trend = (sum(home_rs_hist[-5:]) / 5.0 - st["team_runs_scored"].get(home, 0) / st["team_games"][home]) if len(home_rs_hist) >= 5 and st["team_games"][home] > 0 else 0.0
    away_trend = (sum(away_rs_hist[-5:]) / 5.0 - st["team_runs_scored"].get(away, 0) / st["team_games"][away]) if len(away_rs_hist) >= 5 and st["team_games"][away] > 0 else 0.0
    home_wrc_hist = st["team_lineup_wrc_history"].get(home, [])
    away_wrc_hist = st["team_lineup_wrc_history"].get(away, [])
    home_wrc_delta = home_today_wrc - (sum(home_wrc_hist) / len(home_wrc_hist)) if home_wrc_hist else 0.0
    away_wrc_delta = away_today_wrc - (sum(away_wrc_hist) / len(away_wrc_hist)) if away_wrc_hist else 0.0
    row.extend([float(home_trend), float(home_wrc_delta), float(away_trend), float(away_wrc_delta)])

    temp = max(-10.0, min(40.0, float(game.get("temperature") or 15.0)))
    humidity = max(0.0, min(100.0, float(game.get("humidity") or 50.0)))
    wind_speed = max(0.0, min(30.0, float(game.get("wind_speed") or game.get("windSpeed") or 0.0)))
    is_night = 1.0 if game.get("game_hour", 18) >= NIGHT_GAME_HOUR else 0.0
    row.extend([temp, humidity, wind_speed, is_night, 1.0])

    for team_code, sp_no, roster_players in [(home, game.get("home_sp"), home_roster), (away, game.get("away_sp"), away_roster)]:
        bp = _build_bullpen_features(
            team_code,
            sp_no,
            year,
            roster_players,
            game_date,
            resources["pitcher_prior_cache"],
            pitcher_state,
            stabilize_rates=LIVE_STABILIZE_RATES,
            tuned=LIVE_TUNE_BULLPEN,
        )
        row.extend([
            bp["bp_core_strength"],
            bp["bp_chase_strength"],
            bp["bp_long_strength"],
            bp["bp_fatigue_index"],
            bp["bp_3plus_count"],
        ])

    home_prev_roster = _get_previous_roster_players(resources, game_date, home)
    away_prev_roster = _get_previous_roster_players(resources, game_date, away)
    home_availability = _build_availability_features(
        home_lineup,
        home_roster,
        home_prev_roster,
        year,
        resources["batter_prior_cache"],
        batter_state,
        resources["pitcher_prior_cache"],
        pitcher_state,
        resources["batter_player_ids"],
        resources["pitcher_player_ids"],
        stabilize_rates=LIVE_STABILIZE_RATES,
    )
    away_availability = _build_availability_features(
        away_lineup,
        away_roster,
        away_prev_roster,
        year,
        resources["batter_prior_cache"],
        batter_state,
        resources["pitcher_prior_cache"],
        pitcher_state,
        resources["batter_player_ids"],
        resources["pitcher_player_ids"],
        stabilize_rates=LIVE_STABILIZE_RATES,
    )
    for availability in [home_availability, away_availability]:
        row.extend([
            availability["lineup_gap"],
            availability["roster_loss_hit"],
            availability["roster_loss_pitch"],
        ])

    assert len(row) == len(feature_names_v6(include_bench=LIVE_INCLUDE_BENCH, include_sp_war=LIVE_INCLUDE_SP_WAR))
    return {
        "s_no": game["s_no"],
        "home_team": home,
        "away_team": away,
        "home_sp": game.get("home_sp"),
        "away_sp": game.get("away_sp"),
        "features": row,
    }
