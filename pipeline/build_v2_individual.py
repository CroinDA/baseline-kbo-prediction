"""
v2 학습 데이터셋 구축 — 159개 개인 타자 피처.

피처 구성:
  - 개인 타자 7 × 9 × 2 = 126
  - SP 7 × 2 = 14
  - 팀 컨텍스트 5 × 2 = 10
  - 시퀀스 2 × 2 = 4
  - 환경 5 = 5
  = 총 159개

핵심 원칙: 각 경기의 피처는 해당 경기 이전 데이터만 사용 (미래 정보 누출 금지).
"""
import sys
import json
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
    BULLPEN_LOAD_DAYS,
)
from elo.engine import EloEngine
from features.expanded import (
    feature_names_v2,
    extract_batter_stats,
    extract_sp_stats,
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
    """타자별 시즌 기록 로드 → {(p_no, year): basic_rec}"""
    batter_db = {}
    for year in years:
        f = DATA_DIR / f"batter_seasons_{year}.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        for p_no_str, resp in data.items():
            p_no = int(p_no_str)
            basic_list = resp.get("basic", {}).get("list", [])
            basic_rec = _find_year_record(basic_list, year)
            if basic_rec:
                batter_db[(p_no, year)] = basic_rec
    logger.info("타자 데이터 로드: %d명-시즌", len(batter_db))
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


def _compute_bp_load(team_code, game_date, team_game_dates) -> float:
    """불펜 부하: 최근 3일간 경기수 × 3이닝 추정."""
    dates = team_game_dates.get(team_code, set())
    if not game_date:
        return 0.0
    count = 0
    for d in range(1, BULLPEN_LOAD_DAYS + 1):
        if (game_date - timedelta(days=d)) in dates:
            count += 1
    return float(count * 3)


# ── 개별 타자 스탯 추출 ──

def _get_individual_batters(
    s_no: int,
    team_code: int,
    year: int,
    lineup_db: dict,
    batter_db: dict,
) -> list[dict]:
    """경기 라인업에서 9명 개인 타자 스탯 반환."""
    default = [dict(LEAGUE_AVG_BATTER) for _ in range(9)]
    game_lineups = lineup_db.get(s_no)
    if not game_lineups:
        return default

    players = game_lineups.get(team_code)
    if not players:
        return default

    order_map = {}
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
            basic_rec = batter_db.get((int(p_no), year))
            order_map[order] = extract_batter_stats(basic_rec)

    return [order_map.get(i, dict(LEAGUE_AVG_BATTER)) for i in range(1, 10)]


def _get_sp_full_stats(p_no, year: int, pitcher_db: dict) -> dict:
    """선발투수 7개 스탯 (v2 호환)."""
    if not p_no:
        return dict(LEAGUE_AVG_SP_V2)
    data = pitcher_db.get((p_no, year))
    if not data:
        return dict(LEAGUE_AVG_SP_V2)

    basic_rec = data["basic"]
    deepen_rec = data.get("deepen")
    stats = extract_sp_stats(basic_rec, deepen_rec)

    stats["whip"] = float(basic_rec.get("WHIP") or 1.40)
    ip = float(basic_rec.get("IP") or 1)
    stats["hr9"] = float(basic_rec.get("HR") or 0) / ip * 9 if ip > 0 else 1.0
    return stats


# ── 메인 빌드 ──

def build_dataset_v2(
    years: list[int] = None,
    output_name: str = "training_data_v2",
) -> pd.DataFrame:
    """v2 학습 데이터셋 구축 (159개 피처).

    Returns:
        학습 데이터프레임 (N rows × 160 cols: 159 features + label)
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
    team_game_dates = defaultdict(set)

    prev_year = None
    rows = []
    feat_names = feature_names_v2()

    for i, game in enumerate(all_games):
        year = game["_year"]
        home = game["homeTeam"]
        away = game["awayTeam"]
        home_score = game["homeScore"]
        away_score = game["awayScore"]
        home_sp_no = game.get("homeSP")
        away_sp_no = game.get("awaySP")
        game_date = game.get("_date")

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
            team_game_dates.clear()
        prev_year = year

        s_no = game.get("s_no")

        # 1. 개인 타자 (7×9×2 = 126)
        home_batters = _get_individual_batters(s_no, home, year, lineup_db, batter_db)
        away_batters = _get_individual_batters(s_no, away, year, lineup_db, batter_db)

        # 2. SP (7×2 = 14)
        home_sp_stats = _get_sp_full_stats(home_sp_no, year, pitcher_db)
        away_sp_stats = _get_sp_full_stats(away_sp_no, year, pitcher_db)

        # SP Elo 보정
        if home_sp_no:
            elo.update_sp_rating(home_sp_no, home, home_sp_stats["fip"])
        if away_sp_no:
            elo.update_sp_rating(away_sp_no, away, away_sp_stats["fip"])
        elo.update_team_sp_avg(home, 4.20)
        elo.update_team_sp_avg(away, 4.20)

        # 3. 팀 컨텍스트 (5×2 = 10)
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

        # bp_load
        home_bp_load = _compute_bp_load(home, game_date, team_game_dates)
        away_bp_load = _compute_bp_load(away, game_date, team_game_dates)

        # rest_days
        if game_date and home in team_last_game_date:
            home_rest = float((game_date - team_last_game_date[home]).days)
        else:
            home_rest = 3.0
        if game_date and away in team_last_game_date:
            away_rest = float((game_date - team_last_game_date[away]).days)
        else:
            away_rest = 3.0

        # 4. 시퀀스 (2×2 = 4)
        home_today_wrc = float(np.mean([b["wrcplus"] for b in home_batters]))
        away_today_wrc = float(np.mean([b["wrcplus"] for b in away_batters]))

        home_rs_hist = team_rs_history.get(home, [])
        if len(home_rs_hist) >= 5 and team_games[home] > 0:
            home_trend = float(np.mean(home_rs_hist[-5:]) - team_runs_scored[home] / team_games[home])
        else:
            home_trend = 0.0

        away_rs_hist = team_rs_history.get(away, [])
        if len(away_rs_hist) >= 5 and team_games[away] > 0:
            away_trend = float(np.mean(away_rs_hist[-5:]) - team_runs_scored[away] / team_games[away])
        else:
            away_trend = 0.0

        home_wrc_hist = team_lineup_wrc_history.get(home, [])
        home_wrc_delta = float(home_today_wrc - np.mean(home_wrc_hist)) if home_wrc_hist else 0.0
        away_wrc_hist = team_lineup_wrc_history.get(away, [])
        away_wrc_delta = float(away_today_wrc - np.mean(away_wrc_hist)) if away_wrc_hist else 0.0

        # 5. 환경 (5개)
        temp = game.get("temperature") or 15.0
        if temp == 0 or temp > 45 or temp < -15:
            temp = 15.0
        temp = max(-10.0, min(40.0, float(temp)))

        humidity = float(game.get("humidity") or 50.0)
        humidity = max(0.0, min(100.0, humidity))

        wind_speed = float(game.get("windSpeed") or 0.0)
        wind_speed = max(0.0, min(30.0, wind_speed))

        is_night = 1.0 if game_hour >= NIGHT_GAME_HOUR else 0.0

        s_code = game.get("s_code")
        pf = park_factors.get(s_code, 1.0)

        # ── 159 피처 벡터 조립 ──
        row = []

        # 개인 타자: home b1~b9, away b1~b9 = 126
        batter_stats_order = ["wrcplus", "avg", "obp", "slg", "hr_rate", "bb_rate", "k_rate"]
        for batters in [home_batters, away_batters]:
            for b in batters:
                for stat in batter_stats_order:
                    row.append(float(b.get(stat, LEAGUE_AVG_BATTER.get(stat, 0.0))))

        # SP: home 7, away 7 = 14
        sp_stats_order = ["era", "fip", "whip", "k9", "bb9", "hr9", "war"]
        for sp in [home_sp_stats, away_sp_stats]:
            for stat in sp_stats_order:
                row.append(float(sp.get(stat, LEAGUE_AVG_SP_V2.get(stat, 0.0))))

        # 팀: home 5, away 5 = 10
        row.extend([
            home_elo_val, home_pyth, home_wpct, home_bp_load, home_rest,
            away_elo_val, away_pyth, away_wpct, away_bp_load, away_rest,
        ])

        # 시퀀스: 4
        row.extend([home_trend, home_wrc_delta, away_trend, away_wrc_delta])

        # 환경: 5
        row.extend([temp, humidity, wind_speed, is_night, pf])

        assert len(row) == 159, f"Expected 159, got {len(row)}"

        # 라벨
        if home_score > away_score:
            label = 1
        elif home_score < away_score:
            label = 0
        else:
            # 무승부 — 상태 갱신만
            _update_state(
                home, away, home_score, away_score, home_sp_no, away_sp_no,
                game_date, home_today_wrc, away_today_wrc,
                elo, team_runs_scored, team_runs_allowed, team_games,
                team_results, team_rs_history, team_lineup_wrc_history,
                team_last_game_date, team_game_dates,
            )
            continue

        rows.append(row + [label])

        # ── 경기 결과 반영 ──
        _update_state(
            home, away, home_score, away_score, home_sp_no, away_sp_no,
            game_date, home_today_wrc, away_today_wrc,
            elo, team_runs_scored, team_runs_allowed, team_games,
            team_results, team_rs_history, team_lineup_wrc_history,
            team_last_game_date, team_game_dates,
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


def _update_state(
    home, away, home_score, away_score, home_sp_no, away_sp_no,
    game_date, home_today_wrc, away_today_wrc,
    elo, team_runs_scored, team_runs_allowed, team_games,
    team_results, team_rs_history, team_lineup_wrc_history,
    team_last_game_date, team_game_dates,
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
        team_game_dates[home].add(game_date)
        team_game_dates[away].add(game_date)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="v2 학습 데이터셋 구축 (159 피처)")
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    parser.add_argument("--output", type=str, default="training_data_v2")
    args = parser.parse_args()

    build_dataset_v2(years=args.years, output_name=args.output)
