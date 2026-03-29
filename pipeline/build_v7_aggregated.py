"""
v7 학습 데이터셋 구축 — 집계 피처 47개 + mirror augmentation.

v65 대비 핵심 변경:
1. 개별 타자 162개 → 라인업 집계 15개로 압축
2. Elo를 ML 피처로 직접 투입 (외부 블렌딩 제거)
3. Mirror augmentation (홈/원정 스왑) → 데이터 2배
4. Log5 기대 승률 피처 추가
5. 불펜 3분류 → 단일 bp_strength + bp_fatigue 2개로 단순화
"""
import sys
import json
import logging
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.constants import (
    BULLPEN_LOAD_DAYS,
    ELO_HOME_ADVANTAGE,
    FIP_CONSTANT_DEFAULT,
    LEAGUE_REGULAR,
    NIGHT_GAME_HOUR,
    PYTHAGOREAN_EXPONENT,
)
from elo.engine import EloEngine
from pipeline.build_v6_timeaware import (
    LEAGUE_AVG_BATTER,
    LEAGUE_AVG_SP,
    TUNED_BULLPEN_GROUP_WEIGHTS,
    TUNED_CONSEC_MULTIPLIERS,
    ROLE_QUALITY_MULTIPLIERS,
    _batter_snapshot,
    _pitcher_snapshot,
    _build_bullpen_features,
    _build_batter_prior_cache,
    _build_pitcher_prior_cache,
    _load_schedules,
    _load_lineup_db,
    _load_roster_db,
    _load_batter_summary_db,
    _load_pitcher_summary_db,
    _load_day_index,
    _batter_state_default,
    _pitcher_state_default,
    _empty_batter_counts,
    _empty_pitcher_counts,
    _get_lineup_players,
    _update_batter_state,
    _update_pitcher_state,
    _pythagorean,
    _compute_bp_load,
    _safe_float,
    _get_previous_roster_players,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# ── v7 피처 이름 정의 ──

def feature_names_v7() -> list[str]:
    names = []
    # A. 라인업 집계 (7×2 + 1 diff = 15)
    lineup_stats = [
        "lineup_mean_wrcplus", "lineup_mean_obp", "lineup_mean_slg",
        "lineup_mean_bbk", "lineup_top3_wrcplus", "lineup_std_wrcplus",
        "lineup_total_pa",
    ]
    for side in ["home", "away"]:
        for stat in lineup_stats:
            names.append(f"{side}_{stat}")
    names.append("lineup_wrcplus_diff")

    # B. 선발투수 (4×2 + 1 diff = 9)
    sp_stats = ["sp_fip", "sp_whip", "sp_k9", "sp_bb9"]
    for side in ["home", "away"]:
        for stat in sp_stats:
            names.append(f"{side}_{stat}")
    names.append("sp_fip_diff")

    # C. Elo (3)
    names.extend(["elo_diff", "elo_win_prob", "elo_sp_diff"])

    # D. 팀 컨텍스트 (4×2 = 8)
    team_stats = ["pyth_exp", "wpct", "rest_days", "bp_load"]
    for side in ["home", "away"]:
        for stat in team_stats:
            names.append(f"{side}_{stat}")

    # E. 불펜 (2×2 = 4)
    bp_stats = ["bp_strength", "bp_fatigue"]
    for side in ["home", "away"]:
        for stat in bp_stats:
            names.append(f"{side}_{stat}")

    # F. 모멘텀 (2×2 = 4)
    momentum_stats = ["scoring_trend", "lineup_wrc_delta"]
    for side in ["home", "away"]:
        for stat in momentum_stats:
            names.append(f"{side}_{stat}")

    # G. Log5 (1)
    names.append("log5_home_prob")

    # H. 환경 (3)
    names.extend(["temperature", "is_night", "park_factor"])

    return names


# ── 새 집계 함수 ──

def _aggregate_lineup(batter_snapshots: list[dict]) -> dict:
    """9타자 스냅샷 → 7개 집계값."""
    wrc_values = [b.get("wrcplus", LEAGUE_AVG_BATTER["wrcplus"]) for b in batter_snapshots]
    obp_values = [b.get("obp", LEAGUE_AVG_BATTER["obp"]) for b in batter_snapshots]
    slg_values = [b.get("slg", LEAGUE_AVG_BATTER["slg"]) for b in batter_snapshots]
    bbk_values = [b.get("bbk_ratio", LEAGUE_AVG_BATTER["bbk_ratio"]) for b in batter_snapshots]
    pa_values = [b.get("pa", 0.0) for b in batter_snapshots]

    sorted_wrc = sorted(wrc_values, reverse=True)

    return {
        "lineup_mean_wrcplus": float(np.mean(wrc_values)),
        "lineup_mean_obp": float(np.mean(obp_values)),
        "lineup_mean_slg": float(np.mean(slg_values)),
        "lineup_mean_bbk": float(np.mean(bbk_values)),
        "lineup_top3_wrcplus": float(np.mean(sorted_wrc[:3])),
        "lineup_std_wrcplus": float(np.std(wrc_values)),
        "lineup_total_pa": float(sum(pa_values)),
    }


def _log5(home_wpct: float, away_wpct: float) -> float:
    """Bill James Log5: pairwise 기대 홈승률."""
    h = max(0.01, min(0.99, home_wpct))
    a = max(0.01, min(0.99, away_wpct))
    return (h * (1 - a)) / (h * (1 - a) + a * (1 - h))


def _aggregate_bullpen(bp_features: dict) -> dict:
    """v6 불펜 5피처 → v7 2피처로 압축."""
    # bp_strength = core×0.5 + chase×0.3 + long×0.2 가중평균
    strength = (
        bp_features["bp_core_strength"] * 0.5
        + bp_features["bp_chase_strength"] * 0.3
        + bp_features["bp_long_strength"] * 0.2
    )
    return {
        "bp_strength": float(strength),
        "bp_fatigue": float(bp_features["bp_fatigue_index"]),
    }


def _mirror_row(row: list[float], feat_names: list[str]) -> list[float]:
    """홈/원정 스왑 + diff 부호 반전 + 레이블 반전."""
    mirrored = list(row)
    n = len(feat_names)
    label = mirrored[n]  # label is at index n

    # Build index maps
    home_indices = {}
    away_indices = {}
    diff_indices = []

    for i, name in enumerate(feat_names):
        if name.startswith("home_"):
            suffix = name[5:]
            home_indices[suffix] = i
        elif name.startswith("away_"):
            suffix = name[5:]
            away_indices[suffix] = i
        elif "diff" in name or name == "log5_home_prob":
            diff_indices.append(i)

    # Swap home/away
    for suffix in home_indices:
        if suffix in away_indices:
            hi, ai = home_indices[suffix], away_indices[suffix]
            mirrored[hi], mirrored[ai] = row[ai], row[hi]

    # Flip diffs
    for i in diff_indices:
        name = feat_names[i]
        if name == "log5_home_prob":
            mirrored[i] = 1.0 - row[i]
        else:
            mirrored[i] = -row[i]

    # Flip label
    mirrored[n] = 1 - label

    return mirrored


def build_dataset_v7(
    years: list[int] = None,
    output_name: str = "training_data_v7",
    augment_mirror: bool = True,
) -> pd.DataFrame:
    if years is None:
        years = [2023, 2024, 2025]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    all_games = _load_schedules(years)
    lineup_db = _load_lineup_db(years)
    roster_db = _load_roster_db(years)
    batter_summary_db = _load_batter_summary_db(years)
    pitcher_summary_db = _load_pitcher_summary_db(years)
    batter_by_game = _load_day_index(years, "batting")
    pitcher_by_game = _load_day_index(years, "pitching")

    if not batter_by_game or not pitcher_by_game:
        raise RuntimeError(
            "v7은 batter_days_*.json / pitcher_days_*.json 이 필요합니다."
        )

    batter_prior_cache = _build_batter_prior_cache(years, batter_summary_db)
    pitcher_prior_cache = _build_pitcher_prior_cache(years, pitcher_summary_db)

    elo = EloEngine()

    batter_state = defaultdict(_batter_state_default)
    pitcher_state = defaultdict(_pitcher_state_default)

    team_runs_scored = defaultdict(float)
    team_runs_allowed = defaultdict(float)
    team_games = defaultdict(int)
    team_results = defaultdict(list)
    team_rs_history = defaultdict(list)
    team_lineup_wrc_history = defaultdict(list)
    team_last_game_date = {}
    team_game_dates = defaultdict(set)

    prev_year = None
    rows = []
    feat_names = feature_names_v7()

    for i, game in enumerate(all_games):
        year = game["_year"]
        home = game["homeTeam"]
        away = game["awayTeam"]
        s_no = game.get("s_no")
        game_date = game.get("_date")
        date_str = game_date.isoformat() if game_date else None
        home_score = game["homeScore"]
        away_score = game["awayScore"]
        home_sp_no = game.get("homeSP")
        away_sp_no = game.get("awaySP")

        hm = str(game.get("hm", "18:00:00"))
        try:
            game_hour = int(hm.split(":")[0]) if ":" in hm else int(hm[:2])
        except (ValueError, IndexError):
            game_hour = 18

        if prev_year is not None and year != prev_year:
            logger.info("시즌 전환: %d → %d", prev_year, year)
            elo.new_season()
            batter_state.clear()
            pitcher_state.clear()
            team_runs_scored.clear()
            team_runs_allowed.clear()
            team_games.clear()
            team_results.clear()
            team_rs_history.clear()
            team_lineup_wrc_history.clear()
            team_last_game_date.clear()
            team_game_dates.clear()
        prev_year = year

        home_lineup = _get_lineup_players(s_no, home, lineup_db)
        away_lineup = _get_lineup_players(s_no, away, lineup_db)
        home_roster = roster_db.get(date_str, {}).get(home, [])
        away_roster = roster_db.get(date_str, {}).get(away, [])

        # ── A. 라인업 집계 ──
        home_batters = [
            _batter_snapshot(p_no, year, batter_prior_cache, batter_state, stabilize_rates=True)
            for p_no in home_lineup
        ]
        away_batters = [
            _batter_snapshot(p_no, year, batter_prior_cache, batter_state, stabilize_rates=True)
            for p_no in away_lineup
        ]
        while len(home_batters) < 9:
            home_batters.append(dict(LEAGUE_AVG_BATTER))
        while len(away_batters) < 9:
            away_batters.append(dict(LEAGUE_AVG_BATTER))

        home_agg = _aggregate_lineup(home_batters)
        away_agg = _aggregate_lineup(away_batters)

        row = []
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
            home_sp_no, year, pitcher_prior_cache, pitcher_state, stabilize_rates=True
        )
        away_sp_stats = _pitcher_snapshot(
            away_sp_no, year, pitcher_prior_cache, pitcher_state, stabilize_rates=True
        )

        for sp in [home_sp_stats, away_sp_stats]:
            row.extend([
                float(sp.get("fip", LEAGUE_AVG_SP["fip"])),
                float(sp.get("whip", LEAGUE_AVG_SP["whip"])),
                float(sp.get("k9", LEAGUE_AVG_SP["k9"])),
                float(sp.get("bb9", LEAGUE_AVG_SP["bb9"])),
            ])
        # sp_fip_diff: 원정SP FIP - 홈SP FIP (양수=홈 유리)
        row.append(
            float(away_sp_stats.get("fip", LEAGUE_AVG_SP["fip"]))
            - float(home_sp_stats.get("fip", LEAGUE_AVG_SP["fip"]))
        )

        # ── C. Elo ──
        if home_sp_no:
            elo.update_sp_rating(home_sp_no, home, home_sp_stats["fip"])
        if away_sp_no:
            elo.update_sp_rating(away_sp_no, away, away_sp_stats["fip"])
        elo.update_team_sp_avg(home, 4.20)
        elo.update_team_sp_avg(away, 4.20)

        home_elo = elo.get_rating(home)
        away_elo = elo.get_rating(away)
        elo_diff = home_elo - away_elo + ELO_HOME_ADVANTAGE
        elo_win_prob = elo.predict(home, away, home_sp_no, away_sp_no)
        home_sp_adj = elo.get_sp_adjustment(home_sp_no, home) if home_sp_no else 0.0
        away_sp_adj = elo.get_sp_adjustment(away_sp_no, away) if away_sp_no else 0.0
        elo_sp_diff = home_sp_adj - away_sp_adj

        row.extend([float(elo_diff), float(elo_win_prob), float(elo_sp_diff)])

        # ── D. 팀 컨텍스트 ──
        h_games = max(team_games[home], 1)
        a_games = max(team_games[away], 1)
        home_rs_avg = team_runs_scored[home] / h_games
        home_ra_avg = team_runs_allowed[home] / h_games
        away_rs_avg = team_runs_scored[away] / a_games
        away_ra_avg = team_runs_allowed[away] / a_games
        home_pyth = _pythagorean(home_rs_avg * 9, home_ra_avg * 9)
        away_pyth = _pythagorean(away_rs_avg * 9, away_ra_avg * 9)
        home_wpct = sum(team_results.get(home, [])) / len(team_results[home]) if team_results.get(home) else 0.5
        away_wpct = sum(team_results.get(away, [])) / len(team_results[away]) if team_results.get(away) else 0.5
        home_rest = float((game_date - team_last_game_date[home]).days) if game_date and home in team_last_game_date else 3.0
        away_rest = float((game_date - team_last_game_date[away]).days) if game_date and away in team_last_game_date else 3.0
        home_bp_load = _compute_bp_load(home, game_date, team_game_dates)
        away_bp_load = _compute_bp_load(away, game_date, team_game_dates)

        row.extend([
            home_pyth, home_wpct, home_rest, home_bp_load,
            away_pyth, away_wpct, away_rest, away_bp_load,
        ])

        # ── E. 불펜 ──
        home_bp_raw = _build_bullpen_features(
            home, home_sp_no, year, home_roster, game_date,
            pitcher_prior_cache, pitcher_state, stabilize_rates=True, tuned=True,
        )
        away_bp_raw = _build_bullpen_features(
            away, away_sp_no, year, away_roster, game_date,
            pitcher_prior_cache, pitcher_state, stabilize_rates=True, tuned=True,
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
        home_rs_hist = team_rs_history.get(home, [])
        away_rs_hist = team_rs_history.get(away, [])
        home_trend = float(np.mean(home_rs_hist[-5:]) - team_runs_scored[home] / team_games[home]) if len(home_rs_hist) >= 5 and team_games[home] > 0 else 0.0
        away_trend = float(np.mean(away_rs_hist[-5:]) - team_runs_scored[away] / team_games[away]) if len(away_rs_hist) >= 5 and team_games[away] > 0 else 0.0
        home_wrc_hist = team_lineup_wrc_history.get(home, [])
        away_wrc_hist = team_lineup_wrc_history.get(away, [])
        home_wrc_delta = float(home_today_wrc - np.mean(home_wrc_hist)) if home_wrc_hist else 0.0
        away_wrc_delta = float(away_today_wrc - np.mean(away_wrc_hist)) if away_wrc_hist else 0.0
        row.extend([home_trend, home_wrc_delta, away_trend, away_wrc_delta])

        # ── G. Log5 ──
        row.append(_log5(home_wpct, away_wpct))

        # ── H. 환경 ──
        temp = max(-10.0, min(40.0, float(game.get("temperature") or 15.0)))
        is_night = 1.0 if game_hour >= NIGHT_GAME_HOUR else 0.0
        row.extend([temp, is_night, 1.0])

        assert len(row) == len(feat_names), f"expected {len(feat_names)}, got {len(row)}"

        # ── 레이블 ──
        if home_score > away_score:
            label = 1
        elif home_score < away_score:
            label = 0
        else:
            label = None

        if label is not None:
            rows.append(row + [label])

        # ── 상태 갱신 (경기 후) ──
        for rec in batter_by_game.get(s_no, []):
            _update_batter_state(rec, batter_state)
        for rec in pitcher_by_game.get(s_no, []):
            _update_pitcher_state(rec, pitcher_state)

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
            team_game_dates[home].add(game_date)
            team_game_dates[away].add(game_date)

        if (i + 1) % 250 == 0:
            logger.info("  진행: %d/%d 경기 처리", i + 1, len(all_games))

    # ── DataFrame 생성 ──
    columns = feat_names + ["label"]
    df = pd.DataFrame(rows, columns=columns)
    logger.info("v7 원본 데이터: %d행 × %d열", len(df), len(df.columns))

    # ── Mirror augmentation ──
    if augment_mirror and len(df) > 0:
        mirrored_rows = []
        for _, r in df.iterrows():
            original = r.tolist()
            mirrored_rows.append(_mirror_row(original, feat_names))
        df_mirror = pd.DataFrame(mirrored_rows, columns=columns)
        df = pd.concat([df, df_mirror], ignore_index=True)
        logger.info("Mirror augmentation 적용: %d행", len(df))

    logger.info("v7 최종 데이터셋: %d행 × %d열", len(df), len(df.columns))
    logger.info("홈팀 승률: %.1f%%", df["label"].mean() * 100 if len(df) else 0.0)

    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        logger.warning("NaN %d개 발견 → 0.0으로 대체", nan_count)
        df = df.fillna(0.0)

    csv_path = DATA_DIR / f"{output_name}.csv"
    df.to_csv(csv_path, index=False)
    logger.info("저장: %s", csv_path)

    try:
        parquet_path = DATA_DIR / f"{output_name}.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info("저장: %s", parquet_path)
    except Exception:
        logger.info("Parquet 저장 스킵 (pyarrow 없음)")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="v7 데이터셋 구축")
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    parser.add_argument("--output", type=str, default="training_data_v7")
    parser.add_argument("--no-mirror", action="store_true", help="Mirror augmentation 비활성화")
    args = parser.parse_args()

    build_dataset_v7(
        years=args.years,
        output_name=args.output,
        augment_mirror=not args.no_mirror,
    )
