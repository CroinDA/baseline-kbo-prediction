"""
v7 live 피처 빌더.

v65 live 패턴을 미러링하되 47개 집계 피처 출력.
Elo를 ML 피처로 직접 계산.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from config.constants import ELO_HOME_ADVANTAGE, LEAGUE_REGULAR, NIGHT_GAME_HOUR
from data.collector import get_game_lineup, get_player_day, get_player_season
from elo.engine import EloEngine
from pipeline import daily_run as dr
from pipeline.build_v6_timeaware import (
    LEAGUE_AVG_BATTER,
    LEAGUE_AVG_SP,
    _batter_snapshot,
    _build_batter_prior_cache,
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
)
from pipeline.build_v7_aggregated import (
    _aggregate_lineup,
    _aggregate_bullpen,
    _log5,
    feature_names_v7,
)
from pipeline.live_v65 import (
    _build_current_batter_state,
    _build_current_pitcher_state,
    _ensure_player_season_cached,
    _find_year_record,
    _get_lineup_players,
    _get_previous_roster_players as _get_prev_roster_live,
    _get_roster_players,
    _load_json,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

_resource_cache_v7: dict | None = None


def _ensure_resources_v7(target_year: int) -> dict:
    global _resource_cache_v7
    if _resource_cache_v7 is not None and _resource_cache_v7.get("target_year") == target_year:
        return _resource_cache_v7

    prior_years = [2023, 2024, 2025]
    batter_summary_db = _load_batter_summary_db(prior_years)
    pitcher_summary_db = _load_pitcher_summary_db(prior_years)
    prior_cache_years = sorted(set(prior_years + [target_year]))
    rosters = _load_json(DATA_DIR / f"rosters_{target_year}.json", {})

    _resource_cache_v7 = {
        "target_year": target_year,
        "batter_prior_cache": _build_batter_prior_cache(prior_cache_years, batter_summary_db),
        "pitcher_prior_cache": _build_pitcher_prior_cache(prior_cache_years, pitcher_summary_db),
        "rosters": rosters,
    }
    return _resource_cache_v7


def build_game_features_v7_live(
    game: dict,
    elo_engine: EloEngine,
    year: int,
    use_confirmed_lineup: bool = True,
) -> dict:
    """v7 live 피처 빌더 — 47개 집계 피처 출력."""
    resources = _ensure_resources_v7(year)
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

    batter_players = sorted(set(home_lineup + away_lineup))
    pitcher_players = sorted(set(home_roster + away_roster + [game.get("home_sp"), game.get("away_sp")]))
    batter_state = _build_current_batter_state(batter_players, year)
    pitcher_state = _build_current_pitcher_state(pitcher_players, year, target_dt)

    row = []

    # ── A. 라인업 집계 ──
    home_batters = [
        _batter_snapshot(p_no, year, resources["batter_prior_cache"], batter_state, stabilize_rates=True)
        for p_no in home_lineup
    ]
    away_batters = [
        _batter_snapshot(p_no, year, resources["batter_prior_cache"], batter_state, stabilize_rates=True)
        for p_no in away_lineup
    ]
    while len(home_batters) < 9:
        home_batters.append(dict(LEAGUE_AVG_BATTER))
    while len(away_batters) < 9:
        away_batters.append(dict(LEAGUE_AVG_BATTER))

    home_agg = _aggregate_lineup(home_batters)
    away_agg = _aggregate_lineup(away_batters)

    for stat in ["lineup_mean_wrcplus", "lineup_mean_obp", "lineup_mean_slg",
                  "lineup_mean_bbk", "lineup_top3_wrcplus", "lineup_std_wrcplus",
                  "lineup_total_pa"]:
        row.append(home_agg[stat])
    for stat in ["lineup_mean_wrcplus", "lineup_mean_obp", "lineup_mean_slg",
                  "lineup_mean_bbk", "lineup_top3_wrcplus", "lineup_std_wrcplus",
                  "lineup_total_pa"]:
        row.append(away_agg[stat])
    row.append(home_agg["lineup_mean_wrcplus"] - away_agg["lineup_mean_wrcplus"])

    # ── B. 선발투수 ──
    home_sp_stats = _pitcher_snapshot(
        game.get("home_sp"), year, resources["pitcher_prior_cache"], pitcher_state, stabilize_rates=True,
    )
    away_sp_stats = _pitcher_snapshot(
        game.get("away_sp"), year, resources["pitcher_prior_cache"], pitcher_state, stabilize_rates=True,
    )
    if game.get("home_sp"):
        elo_engine.update_sp_rating(game["home_sp"], home, home_sp_stats["fip"])
    if game.get("away_sp"):
        elo_engine.update_sp_rating(game["away_sp"], away, away_sp_stats["fip"])
    elo_engine.update_team_sp_avg(home, 4.20)
    elo_engine.update_team_sp_avg(away, 4.20)

    for sp in [home_sp_stats, away_sp_stats]:
        row.extend([
            float(sp.get("fip", LEAGUE_AVG_SP["fip"])),
            float(sp.get("whip", LEAGUE_AVG_SP["whip"])),
            float(sp.get("k9", LEAGUE_AVG_SP["k9"])),
            float(sp.get("bb9", LEAGUE_AVG_SP["bb9"])),
        ])
    row.append(
        float(away_sp_stats.get("fip", LEAGUE_AVG_SP["fip"]))
        - float(home_sp_stats.get("fip", LEAGUE_AVG_SP["fip"]))
    )

    # ── C. Elo ──
    home_elo = elo_engine.get_rating(home)
    away_elo = elo_engine.get_rating(away)
    elo_diff = home_elo - away_elo + ELO_HOME_ADVANTAGE
    elo_win_prob = elo_engine.predict(home, away, game.get("home_sp"), game.get("away_sp"))
    home_sp_adj = elo_engine.get_sp_adjustment(game["home_sp"], home) if game.get("home_sp") else 0.0
    away_sp_adj = elo_engine.get_sp_adjustment(game["away_sp"], away) if game.get("away_sp") else 0.0
    elo_sp_diff = home_sp_adj - away_sp_adj
    row.extend([float(elo_diff), float(elo_win_prob), float(elo_sp_diff)])

    # ── D. 팀 컨텍스트 ──
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
    home_bp_load = _compute_bp_load(home, game_date, st["team_game_dates"])
    away_bp_load = _compute_bp_load(away, game_date, st["team_game_dates"])
    row.extend([
        _pythagorean(home_rs_avg * 9, home_ra_avg * 9), home_wpct, home_rest, home_bp_load,
        _pythagorean(away_rs_avg * 9, away_ra_avg * 9), away_wpct, away_rest, away_bp_load,
    ])

    # ── E. 불펜 ──
    home_bp_raw = _build_bullpen_features(
        home, game.get("home_sp"), year, home_roster, game_date,
        resources["pitcher_prior_cache"], pitcher_state, stabilize_rates=True, tuned=True,
    )
    away_bp_raw = _build_bullpen_features(
        away, game.get("away_sp"), year, away_roster, game_date,
        resources["pitcher_prior_cache"], pitcher_state, stabilize_rates=True, tuned=True,
    )
    home_bp = _aggregate_bullpen(home_bp_raw)
    away_bp = _aggregate_bullpen(away_bp_raw)
    row.extend([
        home_bp["bp_strength"], home_bp["bp_fatigue"],
        away_bp["bp_strength"], away_bp["bp_fatigue"],
    ])

    # ── F. 모멘텀 ──
    home_today_wrc = home_agg["lineup_mean_wrcplus"]
    away_today_wrc = away_agg["lineup_mean_wrcplus"]
    home_rs_hist = st["team_runs_history"].get(home, [])
    away_rs_hist = st["team_runs_history"].get(away, [])
    home_trend = (sum(home_rs_hist[-5:]) / 5.0 - st["team_runs_scored"].get(home, 0) / st["team_games"][home]) if len(home_rs_hist) >= 5 and st["team_games"][home] > 0 else 0.0
    away_trend = (sum(away_rs_hist[-5:]) / 5.0 - st["team_runs_scored"].get(away, 0) / st["team_games"][away]) if len(away_rs_hist) >= 5 and st["team_games"][away] > 0 else 0.0
    home_wrc_hist = st["team_lineup_wrc_history"].get(home, [])
    away_wrc_hist = st["team_lineup_wrc_history"].get(away, [])
    home_wrc_delta = home_today_wrc - (sum(home_wrc_hist) / len(home_wrc_hist)) if home_wrc_hist else 0.0
    away_wrc_delta = away_today_wrc - (sum(away_wrc_hist) / len(away_wrc_hist)) if away_wrc_hist else 0.0
    row.extend([float(home_trend), float(home_wrc_delta), float(away_trend), float(away_wrc_delta)])

    # ── G. Log5 ──
    row.append(_log5(home_wpct, away_wpct))

    # ── H. 환경 ──
    temp = max(-10.0, min(40.0, float(game.get("temperature") or 15.0)))
    is_night = 1.0 if game.get("game_hour", 18) >= NIGHT_GAME_HOUR else 0.0
    row.extend([temp, is_night, 1.0])

    feat_names = feature_names_v7()
    assert len(row) == len(feat_names), f"v7 live: expected {len(feat_names)}, got {len(row)}"

    return {
        "s_no": game["s_no"],
        "home_team": home,
        "away_team": away,
        "home_sp": game.get("home_sp"),
        "away_sp": game.get("away_sp"),
        "features": row,
    }
