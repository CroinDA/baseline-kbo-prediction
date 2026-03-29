"""
2026 실시간 데이터 일일 동기화.

목적:
1. 당일 종료된 정규시즌 경기의 schedule/lineup/roster 캐시 반영
2. 당일 사용 선수들의 regular season playerSeason/playerDay 캐시 새로고침

주의:
- 시범경기 개인 스탯은 현재 API 조합으로 안정적으로 복원되지 않으므로 본 sync는 정규시즌 live 누적만 갱신한다.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.constants import LEAGUE_REGULAR
from data.collector import get_game_lineup, get_player_day, get_player_season
from pipeline.daily_run import collect_daily_roster, collect_daily_schedule, get_today_games

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _save_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


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


def _collect_lineups_for_games(games: list[dict], year: int) -> set[int]:
    out_file = DATA_DIR / f"lineups_{year}.json"
    lineups = _load_json(out_file, {})
    player_ids: set[int] = set()

    for game in games:
        s_no = int(game["s_no"])
        resp = get_game_lineup(s_no)
        if resp.get("result_cd") == 100:
            lineups[str(s_no)] = resp
            for rows in resp.values():
                if not isinstance(rows, list):
                    continue
                for rec in rows:
                    p_no = rec.get("p_no")
                    if p_no:
                        player_ids.add(int(p_no))

    _save_json(out_file, lineups)
    logger.info("live lineup sync 저장: %s (%d 경기)", out_file.name, len(games))
    return player_ids


def _collect_roster_players(date_str: str, year: int) -> tuple[set[int], set[int]]:
    collect_daily_roster(date_str)
    roster_file = DATA_DIR / f"rosters_{year}.json"
    rosters = _load_json(roster_file, {})
    players = rosters.get(date_str, [])
    batter_ids: set[int] = set()
    pitcher_ids: set[int] = set()
    for rec in players:
        p_no = rec.get("p_no")
        if not p_no:
            continue
        pos = rec.get("position")
        p_no = int(p_no)
        if pos == 1:
            pitcher_ids.add(p_no)
        else:
            batter_ids.add(p_no)
    return batter_ids, pitcher_ids


def _refresh_player_season_cache(player_ids: set[int], mode: str, year: int):
    prefix = "batter" if mode == "batting" else "pitcher"
    out_file = DATA_DIR / f"{prefix}_seasons_full_{year}.json"
    data = _load_json(out_file, {})
    for i, p_no in enumerate(sorted(player_ids), start=1):
        resp = get_player_season(p_no, m2=mode, year=year, league_type=LEAGUE_REGULAR)
        data[str(p_no)] = _filter_year_records(resp, year)
        if i % 50 == 0:
            _save_json(out_file, data)
            logger.info("%s season refresh %d/%d", mode, i, len(player_ids))
    _save_json(out_file, data)
    logger.info("%s season refresh 완료: %d명", mode, len(player_ids))


def _refresh_player_day_cache(player_ids: set[int], mode: str, year: int, month: int):
    prefix = "batter" if mode == "batting" else "pitcher"
    out_file = DATA_DIR / f"{prefix}_days_{year}.json"
    data = _load_json(out_file, {})
    month_key = f"{month:02d}"
    for i, p_no in enumerate(sorted(player_ids), start=1):
        p_key = str(p_no)
        player_months = data.get(p_key, {})
        resp = get_player_day(p_no, m2=mode, year=year, month=month)
        player_months[month_key] = resp
        data[p_key] = player_months
        if i % 50 == 0:
            _save_json(out_file, data)
            logger.info("%s day refresh %d/%d", mode, i, len(player_ids))
    _save_json(out_file, data)
    logger.info("%s day refresh 완료: %d명", mode, len(player_ids))


def sync_live_day(date_str: str) -> dict:
    target_dt = datetime.strptime(date_str, "%Y-%m-%d")
    year, month, day = target_dt.year, target_dt.month, target_dt.day

    collect_daily_schedule(date_str)
    games = get_today_games(year, month, day)
    regular_games = [g for g in games if g.get("state") and g.get("s_no")]

    if not regular_games:
        logger.info("live sync 대상 경기 없음: %s", date_str)
        return {"date": date_str, "games": 0, "batters": 0, "pitchers": 0}

    lineup_player_ids = _collect_lineups_for_games(regular_games, year)
    roster_batters, roster_pitchers = _collect_roster_players(date_str, year)

    # 타자는 실제 라인업 중심, 투수는 active roster 전체를 갱신해 불펜 상태를 살린다.
    batter_ids = set(lineup_player_ids) | roster_batters
    pitcher_ids = set(roster_pitchers)
    for game in regular_games:
        if game.get("home_sp"):
            pitcher_ids.add(int(game["home_sp"]))
        if game.get("away_sp"):
            pitcher_ids.add(int(game["away_sp"]))

    _refresh_player_season_cache(batter_ids, "batting", year)
    _refresh_player_season_cache(pitcher_ids, "pitching", year)
    _refresh_player_day_cache(batter_ids, "batting", year, month)
    _refresh_player_day_cache(pitcher_ids, "pitching", year, month)

    return {
        "date": date_str,
        "games": len(regular_games),
        "batters": len(batter_ids),
        "pitchers": len(pitcher_ids),
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="live regular-season cache sync")
    parser.add_argument("--date", required=True, help="대상 날짜 YYYY-MM-DD")
    args = parser.parse_args()

    result = sync_live_day(args.date)
    print(json.dumps(result, ensure_ascii=False, indent=2))
