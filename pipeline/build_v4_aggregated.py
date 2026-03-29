"""
확장 학습 데이터셋 구축 (v4) — 37개 집약 피처.

타선 집계(12) + 선발투수(10) + 팀 컨텍스트(8) + 시퀀스(4) + 환경(3) = 37개
핵심 원칙: 각 경기의 피처는 해당 경기 이전 데이터만 사용 (미래 정보 누출 금지)
"""
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.constants import (
    TEAM_CODES,
    ELO_HOME_ADVANTAGE,
    ROLLING_WINDOW_RECENT,
    LEAGUE_REGULAR,
    PYTHAGOREAN_EXPONENT,
    NIGHT_GAME_HOUR,
)
from elo.engine import EloEngine
from features.expanded import (
    feature_names,
    extract_batter_stats,
    extract_sp_stats,
    aggregate_batting,
    compute_platoon_advantage,
    build_game_features,
    LEAGUE_AVG_BATTER,
    LEAGUE_AVG_SP,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


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
    """리스트에서 해당 연도 레코드 찾기."""
    for rec in records:
        rec_year = rec.get("year")
        if rec_year is not None and str(rec_year) == str(year):
            return rec
    return None


def _load_lineup_db(years: list[int]) -> dict:
    """경기별 라인업 로드 → {s_no: {team_code: [players]}}"""
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
    """투수별 시즌 기록 로드 → {(p_no, year): {"basic": rec, "deepen": rec}}"""
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


# ── 라인업 → 집약 피처 ──

def _get_lineup_aggregated(
    s_no: int,
    team_code: int,
    opposing_team_code: int,
    year: int,
    lineup_db: dict,
    batter_db: dict,
) -> dict:
    """경기 라인업에서 6개 집약 타선 피처 생성.

    Returns:
        aggregate_batting 결과 dict (top5_wrc, bot4_wrc, team_obp, team_slg,
        bb_k_ratio, platoon_adv)
    """
    game_lineups = lineup_db.get(s_no)
    if not game_lineups:
        return aggregate_batting([dict(LEAGUE_AVG_BATTER)] * 9, platoon_adv=0.5)

    players = game_lineups.get(team_code)
    if not players:
        return aggregate_batting([dict(LEAGUE_AVG_BATTER)] * 9, platoon_adv=0.5)

    # 1-9번 타자 스탯 + p_bat 추출
    order_map = {}
    batter_hands = {}
    opposing_sp_throw = None

    # 상대팀 라인업에서 SP의 p_throw 찾기
    opposing_players = game_lineups.get(opposing_team_code, [])
    for p in opposing_players:
        if str(p.get("battingOrder", "")) == "P":
            opposing_sp_throw = p.get("p_throw")
            break

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
            batter_hands[order] = p.get("p_bat", 2)  # 기본 우타

    # 9명 타자 스탯 리스트
    batters = []
    hands = []
    for i in range(1, 10):
        batters.append(order_map.get(i, dict(LEAGUE_AVG_BATTER)))
        hands.append(batter_hands.get(i, 2))

    # 플래툰 어드밴티지 계산
    if opposing_sp_throw is not None:
        platoon_adv = compute_platoon_advantage(hands, opposing_sp_throw)
    else:
        platoon_adv = 0.5

    return aggregate_batting(batters, platoon_adv=platoon_adv)


# ── 유틸리티 ──

def _pythagorean(rs: float, ra: float) -> float:
    if rs <= 0 and ra <= 0:
        return 0.5
    rs_e = rs ** PYTHAGOREAN_EXPONENT
    ra_e = ra ** PYTHAGOREAN_EXPONENT
    denom = rs_e + ra_e
    return rs_e / denom if denom > 0 else 0.5


def _recent_win_pct(results: list[bool], window: int = ROLLING_WINDOW_RECENT) -> float:
    if not results:
        return 0.5
    tail = results[-window:]
    return sum(tail) / len(tail)


def _compute_scoring_trend(
    team_code: int,
    team_runs_history: dict,
    team_runs_scored_total: dict,
    team_games_total: dict,
) -> float:
    """scoring_trend: 최근 5경기 평균 득점 - 시즌 평균."""
    rs_hist = team_runs_history.get(team_code, [])
    total_games = team_games_total.get(team_code, 0)
    if len(rs_hist) >= 5 and total_games > 0:
        rs_5 = np.mean(rs_hist[-5:])
        season_avg_rs = team_runs_scored_total.get(team_code, 0) / total_games
        return float(rs_5 - season_avg_rs)
    return 0.0


# ── 메인 빌드 ──

def _compute_park_factors(years: list[int]) -> dict:
    """박스스코어에서 구장별 Park Factor 계산 → {s_code: float}"""
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


def build_dataset_v2(
    years: list[int] = None,
    output_name: str = "training_data_v4",
) -> pd.DataFrame:
    """집약 학습 데이터셋 구축 (37개 피처).

    Returns:
        학습 데이터프레임 (N rows × 38 cols: 37 features + label)
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
        logger.error("경기 데이터 없음. 먼저 backfill.py 실행 필요.")
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
    team_recent_results = defaultdict(list)
    team_last_game_date = {}

    # ── 시퀀스 추적 ──
    team_runs_history = defaultdict(list)
    team_lineup_wrc_history = defaultdict(list)  # 라인업 평균 wRC+ 히스토리

    prev_year = None
    rows = []
    feat_names = feature_names()

    for i, game in enumerate(all_games):
        year = game["_year"]
        home = game["homeTeam"]
        away = game["awayTeam"]
        home_score = game["homeScore"]
        away_score = game["awayScore"]
        home_sp_no = game.get("homeSP")
        away_sp_no = game.get("awaySP")

        # 시즌 전환
        if prev_year is not None and year != prev_year:
            logger.info("시즌 전환: %d → %d", prev_year, year)
            elo.new_season()
            team_runs_scored.clear()
            team_runs_allowed.clear()
            team_games.clear()
            team_recent_results.clear()
            team_last_game_date.clear()
            team_runs_history.clear()
            team_lineup_wrc_history.clear()
        prev_year = year

        # ── 피처 생성 (경기 이전 데이터만) ──

        s_no = game.get("s_no")

        # 1. 타선 집약 피처 (6개 × 2팀 = 12)
        home_batting = _get_lineup_aggregated(
            s_no, home, away, year, lineup_db, batter_db,
        )
        away_batting = _get_lineup_aggregated(
            s_no, away, home, year, lineup_db, batter_db,
        )

        # 2. 선발투수 스탯 (5개 × 2팀 = 10)
        home_sp_data = pitcher_db.get((home_sp_no, year))
        away_sp_data = pitcher_db.get((away_sp_no, year))
        home_sp_stats = extract_sp_stats(
            home_sp_data["basic"] if home_sp_data else None,
            home_sp_data["deepen"] if home_sp_data else None,
        )
        away_sp_stats = extract_sp_stats(
            away_sp_data["basic"] if away_sp_data else None,
            away_sp_data["deepen"] if away_sp_data else None,
        )

        # 3. SP Elo 보정
        home_sp_fip = home_sp_stats["fip"]
        away_sp_fip = away_sp_stats["fip"]
        if home_sp_no:
            elo.update_sp_rating(home_sp_no, home, home_sp_fip)
        if away_sp_no:
            elo.update_sp_rating(away_sp_no, away, away_sp_fip)
        elo.update_team_sp_avg(home, 4.20)
        elo.update_team_sp_avg(away, 4.20)

        # 4. 팀 컨텍스트 (4개 × 2팀 = 8)
        home_elo_val = elo.get_rating(home)
        away_elo_val = elo.get_rating(away)

        h_games = max(team_games[home], 1)
        a_games = max(team_games[away], 1)
        home_rs = team_runs_scored[home] / h_games * 9 if h_games > 0 else 4.5
        home_ra = team_runs_allowed[home] / h_games * 9 if h_games > 0 else 4.5
        away_rs = team_runs_scored[away] / a_games * 9 if a_games > 0 else 4.5
        away_ra = team_runs_allowed[away] / a_games * 9 if a_games > 0 else 4.5

        home_pyth = _pythagorean(home_rs, home_ra)
        away_pyth = _pythagorean(away_rs, away_ra)

        home_wpct = _recent_win_pct(team_recent_results.get(home, []))
        away_wpct = _recent_win_pct(team_recent_results.get(away, []))

        game_date = game.get("_date")
        if game_date and home in team_last_game_date:
            home_rest = float((game_date - team_last_game_date[home]).days)
        else:
            home_rest = 3.0
        if game_date and away in team_last_game_date:
            away_rest = float((game_date - team_last_game_date[away]).days)
        else:
            away_rest = 3.0

        # 5. 시퀀스 (2개 × 2팀 = 4)
        home_trend = _compute_scoring_trend(
            home, team_runs_history, team_runs_scored, team_games,
        )
        away_trend = _compute_scoring_trend(
            away, team_runs_history, team_runs_scored, team_games,
        )

        # lineup_wrc_delta: 오늘 라인업 평균 wRC+ - 팀 시즌 평균 라인업 wRC+
        home_today_wrc = home_batting.get("top5_wrc", 100.0) * 5/9 + home_batting.get("bot4_wrc", 100.0) * 4/9
        away_today_wrc = away_batting.get("top5_wrc", 100.0) * 5/9 + away_batting.get("bot4_wrc", 100.0) * 4/9

        home_wrc_hist = team_lineup_wrc_history.get(home, [])
        away_wrc_hist = team_lineup_wrc_history.get(away, [])
        home_lineup_wrc_delta = home_today_wrc - np.mean(home_wrc_hist) if home_wrc_hist else 0.0
        away_lineup_wrc_delta = away_today_wrc - np.mean(away_wrc_hist) if away_wrc_hist else 0.0

        # 6. 환경 (3개)
        temp = game.get("temperature") or 15.0
        if temp == 0 or temp > 45 or temp < -15:
            temp = 15.0
        temp = max(-10.0, min(40.0, float(temp)))

        hm = str(game.get("hm", "18:00:00"))
        try:
            game_hour = int(hm.split(":")[0]) if ":" in hm else int(hm[:2])
        except (ValueError, IndexError):
            game_hour = 18
        is_night = 1.0 if game_hour >= NIGHT_GAME_HOUR else 0.0

        s_code = game.get("s_code")
        pf = park_factors.get(s_code, 1.0)

        # ── 피처 벡터 생성 (37개) ──
        row = build_game_features(
            home_batting=home_batting,
            away_batting=away_batting,
            home_sp=home_sp_stats,
            away_sp=away_sp_stats,
            home_elo=home_elo_val,
            away_elo=away_elo_val,
            home_pyth=home_pyth,
            away_pyth=away_pyth,
            home_recent_wpct=home_wpct,
            away_recent_wpct=away_wpct,
            home_rest=home_rest,
            away_rest=away_rest,
            home_scoring_trend=home_trend,
            away_scoring_trend=away_trend,
            home_lineup_wrc_delta=float(home_lineup_wrc_delta),
            away_lineup_wrc_delta=float(away_lineup_wrc_delta),
            park_factor=pf,
            temperature=temp,
            is_night=is_night,
        )

        # 라벨
        if home_score > away_score:
            label = 1
        elif home_score < away_score:
            label = 0
        else:
            continue  # 무승부 제외

        rows.append(row + [label])

        # ── 경기 결과 반영 ──
        elo.update(home, away, home_score, away_score, home_sp_no, away_sp_no)
        team_runs_scored[home] += home_score
        team_runs_allowed[home] += away_score
        team_runs_scored[away] += away_score
        team_runs_allowed[away] += home_score
        team_games[home] += 1
        team_games[away] += 1
        team_recent_results[home].append(home_score > away_score)
        team_recent_results[away].append(away_score > home_score)

        # 시퀀스 히스토리 갱신
        team_runs_history[home].append(float(home_score))
        team_runs_history[away].append(float(away_score))

        # 라인업 wRC+ 히스토리 갱신
        team_lineup_wrc_history[home].append(float(home_today_wrc))
        team_lineup_wrc_history[away].append(float(away_today_wrc))

        if game_date:
            team_last_game_date[home] = game_date
            team_last_game_date[away] = game_date

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

    # Elo 상태 저장
    elo.save()
    logger.info("Elo 최종 상태 저장")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="집약 학습 데이터셋 구축 (v4)")
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    parser.add_argument("--output", type=str, default="training_data_v4")
    args = parser.parse_args()

    build_dataset_v2(years=args.years, output_name=args.output)
