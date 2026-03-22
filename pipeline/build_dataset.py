"""
학습 데이터셋 구축.

과거 경기 데이터를 시간순으로 순회하면서:
1. Elo를 시뮬레이션 (각 경기 전 시점의 Elo로 피처 생성)
2. 10개 피처 + 정답 라벨(홈팀 승=1, 패=0) 생성
3. Parquet/CSV로 저장 → train.py에서 바로 사용

핵심 원칙: 각 경기의 피처는 해당 경기 이전 데이터만 사용 (미래 정보 누출 금지)
"""
import sys
import json
import logging
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
    BULLPEN_LOAD_DAYS,
    FIP_CONSTANT_DEFAULT,
)
from elo.engine import EloEngine
from features.builder import FeatureBuilder, GameFeatures

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _parse_schedules(years: list[int]) -> list[dict]:
    """연도별 일정 파일에서 완료된 경기만 시간순 정렬하여 반환."""
    all_games = []

    for year in years:
        f = DATA_DIR / f"schedules_{year}.json"
        if not f.exists():
            logger.warning("일정 파일 없음: %s", f)
            continue
        games = json.loads(f.read_text())
        for g in games:
            # 경기 완료 (점수가 있는 경기만)
            if g.get("homeScore") is not None and g.get("awayScore") is not None:
                g["_year"] = year
                g["_sort_key"] = g.get("gameDate", 0)
                all_games.append(g)

    # 시간순 정렬
    all_games.sort(key=lambda x: (x["_year"], x["_sort_key"], x.get("s_no", 0)))
    logger.info("완료된 경기 총 %d개 (years=%s)", len(all_games), years)
    return all_games


def _load_pitcher_data(years: list[int]) -> dict:
    """투수 시즌 기록 로드 → {(p_no, year): {"fip": float, "k_bb": float}}"""
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

            fip = 4.50
            k_bb = 2.0

            if basic_list:
                fip = basic_list[0].get("FIP", 4.50) or 4.50
            if deepen_list:
                k_bb = deepen_list[0].get("KBB", 2.0) or 2.0

            pitcher_db[(p_no, year)] = {"fip": fip, "k_bb": k_bb}

    logger.info("투수 데이터 로드: %d명-시즌", len(pitcher_db))
    return pitcher_db


def _load_team_wrc(years: list[int]) -> dict:
    """팀별 시즌 wRC+ 로드 → {(t_code, year): wrc_plus}"""
    team_wrc = {}
    for year in years:
        f = DATA_DIR / f"team_records_{year}.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        batting = data.get("batting", {})
        team_list = batting.get("list", [])
        if not team_list:
            # 중첩 구조 대응
            team_list = batting.get("data", batting.get("teams", []))

        for t in team_list if isinstance(team_list, list) else []:
            t_code = t.get("t_code")
            wrc = t.get("wRCplus", t.get("wrc_plus", 100.0))
            if t_code:
                team_wrc[(t_code, year)] = wrc or 100.0

    logger.info("팀 wRC+ 로드: %d팀-시즌", len(team_wrc))
    return team_wrc


def build_dataset(
    years: list[int] = None,
    output_name: str = "training_data",
) -> pd.DataFrame:
    """학습 데이터셋 구축.

    각 경기를 시간순으로 순회하며:
    - 경기 '이전' 시점의 Elo/통계로 피처 생성 (미래 누출 없음)
    - 경기 결과로 Elo 갱신
    - 피처 + 라벨 저장

    Returns:
        학습 데이터프레임 (N rows × 11 cols: 10 features + label)
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

    pitcher_db = _load_pitcher_data(years)
    team_wrc_db = _load_team_wrc(years)

    # ── Elo 엔진 (처음부터 시뮬레이션) ──
    elo = EloEngine()
    fb = FeatureBuilder()

    # ── 팀별 누적 통계 (시뮬레이션 중 갱신) ──
    team_runs_scored = defaultdict(float)     # 누적 득점
    team_runs_allowed = defaultdict(float)    # 누적 실점
    team_games = defaultdict(int)             # 누적 경기 수
    team_recent_results = defaultdict(list)   # 최근 경기 결과 [True/False, ...]
    team_bullpen_daily_ip = defaultdict(lambda: defaultdict(float))  # team → date → ip

    # ── 시즌 경계 감지 ──
    prev_year = None

    # ── 데이터셋 축적 ──
    rows = []

    for i, game in enumerate(all_games):
        year = game["_year"]
        home = game["homeTeam"]
        away = game["awayTeam"]
        home_score = game["homeScore"]
        away_score = game["awayScore"]
        home_sp = game.get("homeSP")
        away_sp = game.get("awaySP")

        # 시즌 전환 시 Elo 평균 회귀 + 누적 통계 리셋
        if prev_year is not None and year != prev_year:
            logger.info("시즌 전환: %d → %d (Elo 평균 회귀)", prev_year, year)
            elo.new_season()
            team_runs_scored.clear()
            team_runs_allowed.clear()
            team_games.clear()
            team_recent_results.clear()
            team_bullpen_daily_ip.clear()
        prev_year = year

        # ── 경기 이전 시점의 피처 생성 ──

        # 선발 투수 스탯
        home_sp_data = pitcher_db.get((home_sp, year), {"fip": 4.50, "k_bb": 2.0})
        away_sp_data = pitcher_db.get((away_sp, year), {"fip": 4.50, "k_bb": 2.0})

        # SP Elo 보정용 (간략화: FIP 기반)
        if home_sp:
            elo.update_sp_rating(home_sp, home, home_sp_data["fip"])
        if away_sp:
            elo.update_sp_rating(away_sp, away, away_sp_data["fip"])

        # 팀 투수 평균 (간략화: 리그 평균 사용)
        elo.update_team_sp_avg(home, 4.20)
        elo.update_team_sp_avg(away, 4.20)

        # Elo 차이 (SP 보정 포함)
        home_elo = elo.get_rating(home)
        away_elo = elo.get_rating(away)
        elo_diff = home_elo - away_elo + ELO_HOME_ADVANTAGE
        if home_sp:
            elo_diff += elo.get_sp_adjustment(home_sp, home)
        if away_sp:
            elo_diff -= elo.get_sp_adjustment(away_sp, away)

        # 누적 득실점 (피타고리안용)
        h_games = max(team_games[home], 1)
        a_games = max(team_games[away], 1)
        home_rs = team_runs_scored[home] / h_games * 9 if h_games > 0 else 4.5
        home_ra = team_runs_allowed[home] / h_games * 9 if h_games > 0 else 4.5
        away_rs = team_runs_scored[away] / a_games * 9 if a_games > 0 else 4.5
        away_ra = team_runs_allowed[away] / a_games * 9 if a_games > 0 else 4.5

        # 팀 wRC+
        home_wrc = team_wrc_db.get((home, year), 100.0)
        away_wrc = team_wrc_db.get((away, year), 100.0)

        # 최근 10경기 승률
        home_recent = team_recent_results.get(home, [])
        away_recent = team_recent_results.get(away, [])

        # 불펜 피로도 (간략화: 누적 데이터 없는 초기 버전에서는 0)
        home_bp = []
        away_bp = []

        # 휴식일 (간략화: 기본 1)
        home_rest = 1
        away_rest = 1

        # 기온
        temp = game.get("temperature", 15.0) or 15.0
        game_hour = int(str(game.get("hm", "1800"))[:2]) if game.get("hm") else 18

        # 피처 빌드
        features = fb.build(
            elo_diff=elo_diff,
            home_rs=home_rs, home_ra=home_ra,
            away_rs=away_rs, away_ra=away_ra,
            home_sp_fip=home_sp_data["fip"],
            away_sp_fip=away_sp_data["fip"],
            home_sp_k_bb=home_sp_data["k_bb"],
            away_sp_k_bb=away_sp_data["k_bb"],
            home_lineup=None,
            away_lineup=None,
            home_team_wrc=home_wrc,
            away_team_wrc=away_wrc,
            home_recent=home_recent,
            away_recent=away_recent,
            home_bp_ip=home_bp,
            away_bp_ip=away_bp,
            home_rest=home_rest,
            away_rest=away_rest,
            temperature=temp,
            game_hour=game_hour,
        )

        # 라벨
        if home_score > away_score:
            label = 1
        elif home_score < away_score:
            label = 0
        else:
            continue  # 무승부 제외 (KBO에서는 거의 없음)

        # 행 추가
        row = features.to_list() + [label]
        rows.append(row)

        # ── 경기 결과 반영 (피처 생성 후!) ──
        elo.update(home, away, home_score, away_score, home_sp, away_sp)

        # 누적 통계 갱신
        team_runs_scored[home] += home_score
        team_runs_allowed[home] += away_score
        team_runs_scored[away] += away_score
        team_runs_allowed[away] += home_score
        team_games[home] += 1
        team_games[away] += 1

        team_recent_results[home].append(home_score > away_score)
        team_recent_results[away].append(away_score > home_score)

        if (i + 1) % 500 == 0:
            logger.info("  진행: %d/%d 경기 처리", i + 1, len(all_games))

    # ── DataFrame 생성 ──
    columns = GameFeatures.feature_names() + ["label"]
    df = pd.DataFrame(rows, columns=columns)

    logger.info("데이터셋 생성 완료: %d행 × %d열", len(df), len(df.columns))
    logger.info("홈팀 승률: %.1f%%", df["label"].mean() * 100)

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

    # Elo 최종 상태 저장 (2025 시즌 말 → 2026 시즌 시작 Prior)
    elo.save()
    logger.info("Elo 최종 상태 저장 (2026 시즌 Prior용)")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="학습 데이터셋 구축")
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025],
                        help="대상 연도 (기본: 2023 2024 2025)")
    parser.add_argument("--output", type=str, default="training_data",
                        help="출력 파일명 (기본: training_data)")
    args = parser.parse_args()

    build_dataset(years=args.years, output_name=args.output)
