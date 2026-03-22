"""
과거 데이터 수집 (Backfill).

2023~2025 KBO 정규시즌 전체 경기 데이터를 API에서 수집하여 로컬 저장.
API 키 수령 후 1회 실행.
"""
import sys
import json
import logging
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.constants import LEAGUE_REGULAR, TEAM_CODES
from data.collector import (
    get_game_schedule,
    get_game_boxscore,
    get_player_season,
    get_player_day,
    get_team_record,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def backfill_schedules(years: list[int]):
    """연도별 전체 경기일정 수집 → schedules_{year}.json"""
    for year in years:
        out_file = DATA_DIR / f"schedules_{year}.json"
        if out_file.exists():
            logger.info("스킵 (이미 존재): %s", out_file.name)
            continue

        logger.info("경기일정 수집: %d", year)
        all_games = []

        for month in range(3, 11):  # 3월~10월
            for day in range(1, 32):
                try:
                    resp = get_game_schedule(year=year, month=month, day=day)
                    date_list = resp.get("date", [])
                    if date_list:
                        for g in date_list:
                            g["_date"] = f"{year}-{month:02d}-{day:02d}"
                        all_games.extend(date_list)
                except Exception as e:
                    pass  # 존재하지 않는 날짜 무시

        out_file.write_text(json.dumps(all_games, indent=2, ensure_ascii=False))
        logger.info("저장: %s (%d 경기)", out_file.name, len(all_games))


def backfill_boxscores(years: list[int]):
    """경기별 박스스코어 수집 → boxscores_{year}.json

    schedules 파일에서 s_no를 추출하여 개별 조회.
    """
    for year in years:
        schedule_file = DATA_DIR / f"schedules_{year}.json"
        if not schedule_file.exists():
            logger.warning("일정 파일 없음: %s — 먼저 backfill_schedules 실행", schedule_file)
            continue

        out_file = DATA_DIR / f"boxscores_{year}.json"
        if out_file.exists():
            logger.info("스킵 (이미 존재): %s", out_file.name)
            continue

        games = json.loads(schedule_file.read_text())
        s_nos = list({g["s_no"] for g in games if g.get("s_no")})
        s_nos.sort()

        logger.info("박스스코어 수집: %d (%d 경기)", year, len(s_nos))
        boxscores = []

        for i, s_no in enumerate(s_nos):
            try:
                resp = get_game_boxscore(s_no)
                resp["_s_no"] = s_no
                boxscores.append(resp)
            except Exception as e:
                logger.warning("박스스코어 실패 s_no=%d: %s", s_no, e)

            if (i + 1) % 100 == 0:
                logger.info("  진행: %d/%d", i + 1, len(s_nos))

        out_file.write_text(json.dumps(boxscores, indent=2, ensure_ascii=False))
        logger.info("저장: %s (%d 경기)", out_file.name, len(boxscores))


def backfill_team_records(years: list[int]):
    """팀별 시즌 기록 수집 → team_records_{year}.json"""
    for year in years:
        out_file = DATA_DIR / f"team_records_{year}.json"
        if out_file.exists():
            logger.info("스킵 (이미 존재): %s", out_file.name)
            continue

        logger.info("팀 기록 수집: %d", year)
        records = {}

        for m2 in ["batting", "pitching"]:
            try:
                resp = get_team_record(m2=m2, year=year)
                records[m2] = resp
            except Exception as e:
                logger.warning("팀 기록 실패 %d/%s: %s", year, m2, e)

        out_file.write_text(json.dumps(records, indent=2, ensure_ascii=False))
        logger.info("저장: %s", out_file.name)


def backfill_pitcher_seasons(years: list[int]):
    """주요 투수 시즌 기록 수집 → pitcher_seasons_{year}.json

    schedule에서 등장하는 선발 투수 목록 추출 후 조회.
    """
    for year in years:
        schedule_file = DATA_DIR / f"schedules_{year}.json"
        if not schedule_file.exists():
            continue

        out_file = DATA_DIR / f"pitcher_seasons_{year}.json"
        if out_file.exists():
            logger.info("스킵 (이미 존재): %s", out_file.name)
            continue

        games = json.loads(schedule_file.read_text())

        # 선발 투수 목록 추출
        sp_set = set()
        for g in games:
            if g.get("homeSP"):
                sp_set.add(g["homeSP"])
            if g.get("awaySP"):
                sp_set.add(g["awaySP"])

        sp_list = sorted(sp_set)
        logger.info("투수 시즌 기록 수집: %d (%d명)", year, len(sp_list))

        pitcher_data = {}
        for i, p_no in enumerate(sp_list):
            try:
                resp = get_player_season(p_no, m2="pitching", year=year)
                pitcher_data[p_no] = resp
            except Exception as e:
                logger.warning("투수 %d 실패: %s", p_no, e)

            if (i + 1) % 50 == 0:
                logger.info("  진행: %d/%d", i + 1, len(sp_list))

        out_file.write_text(json.dumps(pitcher_data, indent=2, ensure_ascii=False))
        logger.info("저장: %s (%d명)", out_file.name, len(pitcher_data))


def backfill_batter_seasons(years: list[int]):
    """팀별 타자 시즌 기록 (wRC+ 확보용) → batter_team_wrc_{year}.json

    개인별 전수 조회는 비효율 → 팀 기록실의 타자 데이터 사용.
    """
    # team_records에서 이미 포함됨 — 별도 파일 불필요
    logger.info("타자 wRC+는 team_records에서 추출 가능. 스킵.")


def run_backfill(years: list[int] = None):
    """전체 백필 실행."""
    if years is None:
        years = [2023, 2024, 2025]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    logger.info("=" * 60)
    logger.info("과거 데이터 백필 시작: %s", years)
    logger.info("=" * 60)

    # 순서 중요: schedule → boxscore, pitcher 순
    logger.info("\n[1/4] 경기일정 수집")
    backfill_schedules(years)

    logger.info("\n[2/4] 박스스코어 수집")
    backfill_boxscores(years)

    logger.info("\n[3/4] 팀 기록 수집")
    backfill_team_records(years)

    logger.info("\n[4/4] 투수 시즌 기록 수집")
    backfill_pitcher_seasons(years)

    logger.info("\n백필 완료!")
    logger.info("저장 위치: %s", DATA_DIR)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="과거 KBO 데이터 백필")
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025],
                        help="수집 대상 연도 (기본: 2023 2024 2025)")
    args = parser.parse_args()

    run_backfill(args.years)
