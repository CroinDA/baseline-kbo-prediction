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
    get_game_lineup,
    get_player_season,
    get_player_day,
    get_team_record,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def backfill_schedules(years: list[int]):
    """연도별 전체 경기일정 수집 → schedules_{year}.json

    API 응답 형식: {"0401": [games], "0402": [games], ...}
    월 단위로 조회 (일별 조회보다 효율적).
    """
    for year in years:
        out_file = DATA_DIR / f"schedules_{year}.json"
        if out_file.exists():
            logger.info("스킵 (이미 존재): %s", out_file.name)
            continue

        logger.info("경기일정 수집: %d", year)
        all_games = []

        for month in range(3, 11):  # 3월~10월
            try:
                resp = get_game_schedule(year=year, month=month)

                # API 응답: {"0401": [games], "0402": [...], ...}
                # result_cd, result_msg, update_time 키 제외
                for date_key, games in resp.items():
                    if not isinstance(games, list):
                        continue  # result_cd 등 메타 필드 스킵
                    for g in games:
                        g["_date"] = f"{year}-{month:02d}-{date_key[2:]}"
                        g["_year"] = year
                    all_games.extend(games)

                logger.info("  %d월: %d일치 데이터",
                            month,
                            sum(1 for k, v in resp.items() if isinstance(v, list)))
            except Exception as e:
                logger.warning("  %d월 수집 실패: %s", month, e)

        out_file.write_text(json.dumps(all_games, indent=2, ensure_ascii=False))
        logger.info("저장: %s (%d 경기)", out_file.name, len(all_games))


def backfill_boxscores(years: list[int]):
    """경기별 박스스코어 수집 → boxscores_{year}.json

    schedules 파일에서 s_no를 추출하여 개별 조회.
    완료된 경기(state=3)만 대상.
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
        # state=3 완료된 경기만 (미완료 경기는 박스스코어 없음)
        s_nos = list({g["s_no"] for g in games if g.get("s_no") and g.get("state") == 3})
        s_nos.sort()

        logger.info("박스스코어 수집: %d (%d 경기)", year, len(s_nos))
        boxscores = []

        for i, s_no in enumerate(s_nos):
            try:
                resp = get_game_boxscore(s_no)
                if resp.get("result_cd") == 100:
                    resp["_s_no"] = s_no
                    boxscores.append(resp)
                else:
                    logger.warning("박스스코어 API 실패 s_no=%d: %s",
                                   s_no, resp.get("result_msg"))
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
    API 응답에서 해당 연도 기록만 필터링하여 저장.
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
                filtered = _filter_year_records(resp, year)
                if filtered.get("basic", {}).get("list"):
                    pitcher_data[p_no] = filtered
            except Exception as e:
                logger.warning("투수 %d 실패: %s", p_no, e)

            if (i + 1) % 50 == 0:
                logger.info("  진행: %d/%d", i + 1, len(sp_list))

        out_file.write_text(json.dumps(pitcher_data, indent=2, ensure_ascii=False))
        logger.info("저장: %s (%d명)", out_file.name, len(pitcher_data))


def backfill_lineups(years: list[int]):
    """경기별 라인업 수집 → lineups_{year}.json

    schedules 파일에서 s_no를 추출하여 개별 조회.
    완료된 정규시즌 경기(state=3, leagueType=10100)만 대상.
    """
    for year in years:
        schedule_file = DATA_DIR / f"schedules_{year}.json"
        if not schedule_file.exists():
            logger.warning("일정 파일 없음: %s — 먼저 backfill_schedules 실행", schedule_file)
            continue

        out_file = DATA_DIR / f"lineups_{year}.json"
        if out_file.exists():
            logger.info("스킵 (이미 존재): %s", out_file.name)
            continue

        games = json.loads(schedule_file.read_text())
        # 완료된 정규시즌 경기만
        s_nos = sorted({
            g["s_no"] for g in games
            if g.get("s_no") and g.get("state") == 3
            and g.get("leagueType") == LEAGUE_REGULAR
        })

        logger.info("라인업 수집: %d (%d 경기)", year, len(s_nos))
        lineups = {}

        for i, s_no in enumerate(s_nos):
            try:
                resp = get_game_lineup(s_no)
                if resp.get("result_cd") == 100:
                    lineups[str(s_no)] = resp
                else:
                    logger.debug("라인업 API 실패 s_no=%d: %s",
                                 s_no, resp.get("result_msg"))
            except Exception as e:
                logger.warning("라인업 실패 s_no=%d: %s", s_no, e)

            if (i + 1) % 100 == 0:
                logger.info("  진행: %d/%d", i + 1, len(s_nos))
                # 중간 저장 (크래시 방지)
                out_file.write_text(json.dumps(lineups, indent=2, ensure_ascii=False))

        out_file.write_text(json.dumps(lineups, indent=2, ensure_ascii=False))
        logger.info("저장: %s (%d 경기)", out_file.name, len(lineups))


def _filter_year_records(resp: dict, year: int) -> dict:
    """API 응답에서 해당 연도 레코드만 필터링.

    API가 전 시즌 기록을 반환하므로, 저장 전에 해당 연도만 남긴다.
    """
    filtered = {}
    for section in ["basic", "deepen", "fielding"]:
        section_data = resp.get(section)
        if not section_data:
            continue
        if isinstance(section_data, dict) and "list" in section_data:
            year_records = [
                r for r in section_data["list"]
                if str(r.get("year")) == str(year)
            ]
            if year_records:
                filtered[section] = {"list": year_records}
        else:
            filtered[section] = section_data
    filtered["result_cd"] = resp.get("result_cd", 100)
    filtered["result_msg"] = resp.get("result_msg", "Success")
    return filtered


def backfill_batter_seasons(years: list[int]):
    """라인업에 등장하는 타자들의 시즌 기록 수집 → batter_seasons_{year}.json

    lineups 파일에서 타자 p_no 추출 후, playerSeason batting 조회.
    API 응답에서 해당 연도 기록만 필터링하여 저장.
    """
    for year in years:
        lineup_file = DATA_DIR / f"lineups_{year}.json"
        if not lineup_file.exists():
            logger.warning("라인업 파일 없음: %s — 먼저 backfill_lineups 실행", lineup_file)
            continue

        out_file = DATA_DIR / f"batter_seasons_{year}.json"
        if out_file.exists():
            logger.info("스킵 (이미 존재): %s", out_file.name)
            continue

        # 라인업에서 고유 타자 ID 추출
        lineups = json.loads(lineup_file.read_text())
        batter_set = set()
        for s_no_str, resp in lineups.items():
            for key, val in resp.items():
                if isinstance(val, list):
                    for p in val:
                        p_no = p.get("p_no") or p.get("pNo")
                        if p_no:
                            batter_set.add(int(p_no))

        batter_list = sorted(batter_set)
        logger.info("타자 시즌 기록 수집: %d (%d명)", year, len(batter_list))

        batter_data = {}
        for i, p_no in enumerate(batter_list):
            try:
                resp = get_player_season(p_no, m2="batting", year=year)
                filtered = _filter_year_records(resp, year)
                if filtered.get("basic", {}).get("list"):
                    batter_data[str(p_no)] = filtered
            except Exception as e:
                logger.warning("타자 %d 실패: %s", p_no, e)

            if (i + 1) % 100 == 0:
                logger.info("  진행: %d/%d", i + 1, len(batter_list))
                out_file.write_text(json.dumps(batter_data, indent=2, ensure_ascii=False))

        out_file.write_text(json.dumps(batter_data, indent=2, ensure_ascii=False))
        logger.info("저장: %s (%d명)", out_file.name, len(batter_data))


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

    # 순서 중요: schedule → lineup → batter → boxscore, pitcher 순
    logger.info("\n[1/6] 경기일정 수집")
    backfill_schedules(years)

    logger.info("\n[2/6] 팀 기록 수집")
    backfill_team_records(years)

    logger.info("\n[3/6] 투수 시즌 기록 수집")
    backfill_pitcher_seasons(years)

    logger.info("\n[4/6] 라인업 수집")
    backfill_lineups(years)

    logger.info("\n[5/6] 타자 시즌 기록 수집")
    backfill_batter_seasons(years)

    logger.info("\n[6/6] 박스스코어 수집")
    backfill_boxscores(years)

    logger.info("\n백필 완료!")
    logger.info("저장 위치: %s", DATA_DIR)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="과거 KBO 데이터 백필")
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025],
                        help="수집 대상 연도 (기본: 2023 2024 2025)")
    args = parser.parse_args()

    run_backfill(args.years)
