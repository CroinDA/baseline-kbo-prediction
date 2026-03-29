"""
v6 지원 데이터 백필.

목적:
1. 경기일 로스터 기반으로 "전 선수" 시즌 기록 수집
2. 선수별 날짜 로그(playerDay) 수집

v6는 시점 누적 스탯과 불펜 연투 판정이 필요하므로,
기존 v2/v5보다 더 세밀한 원천 데이터가 필요하다.
"""
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.collector import get_player_day, get_player_season
from config.constants import LEAGUE_EXHIBITION, LEAGUE_REGULAR

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _iter_roster_player_ids(year: int) -> list[int]:
    roster_file = DATA_DIR / f"rosters_{year}.json"
    if not roster_file.exists():
        logger.warning("로스터 파일 없음: %s", roster_file.name)
        return []

    rosters = json.loads(roster_file.read_text())
    player_ids = set()
    for players in rosters.values():
        if not isinstance(players, list):
            continue
        for rec in players:
            p_no = rec.get("p_no")
            if p_no:
                player_ids.add(int(p_no))
    return sorted(player_ids)


def _iter_early_season_core_player_ids(year: int, days: int = 14) -> list[int]:
    sched_file = DATA_DIR / f"schedules_{year}.json"
    lineup_file = DATA_DIR / f"lineups_{year}.json"
    if not sched_file.exists():
        return _iter_roster_player_ids(year)

    sched = json.loads(sched_file.read_text())
    regular = [g for g in sched if g.get("leagueType") == LEAGUE_REGULAR and g.get("gameDate")]
    if not regular:
        return _iter_roster_player_ids(year)

    start = min(datetime.fromtimestamp(g["gameDate"]).date() for g in regular)
    end = start + timedelta(days=days - 1)
    early_games = {
        int(g["s_no"]) for g in regular
        if start <= datetime.fromtimestamp(g["gameDate"]).date() <= end
    }

    player_ids = set()
    if lineup_file.exists():
        lineup_db = json.loads(lineup_file.read_text())
        for s_no in early_games:
            resp = lineup_db.get(str(s_no), {})
            for rows in resp.values():
                if not isinstance(rows, list):
                    continue
                for rec in rows:
                    p_no = rec.get("p_no")
                    order = str(rec.get("battingOrder", ""))
                    if p_no and (order == "P" or (order.isdigit() and 1 <= int(order) <= 9)):
                        player_ids.add(int(p_no))

    if not player_ids:
        return _iter_roster_player_ids(year)
    return sorted(player_ids)


def _filter_year_records(resp: dict, year: int) -> dict:
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


def _has_basic_records(resp: dict) -> bool:
    return bool(resp.get("basic", {}).get("list"))


def _has_day_rows(resp: dict) -> bool:
    return any(k.isdigit() and isinstance(v, dict) for k, v in resp.items())


def _collect_player_seasons_for_ids(
    player_ids: list[int],
    year: int,
    league_type: int,
    batter_out: Path,
    pitcher_out: Path,
):
    batter_data = _load_json(batter_out, {})
    pitcher_data = _load_json(pitcher_out, {})

    logger.info(
        "%d season collect(leagueType=%s): players=%d, existing batter=%d, pitcher=%d",
        year, league_type, len(player_ids), len(batter_data), len(pitcher_data),
    )

    for i, p_no in enumerate(player_ids):
        p_key = str(p_no)

        if p_key not in batter_data:
            try:
                resp = get_player_season(p_no, m2="batting", year=year, league_type=league_type)
                filtered = _filter_year_records(resp, year)
                if _has_basic_records(filtered):
                    batter_data[p_key] = filtered
            except Exception as e:
                logger.warning("batting season 실패 p_no=%d year=%d lt=%s: %s", p_no, year, league_type, e)

        if p_key not in pitcher_data:
            try:
                resp = get_player_season(p_no, m2="pitching", year=year, league_type=league_type)
                filtered = _filter_year_records(resp, year)
                if _has_basic_records(filtered):
                    pitcher_data[p_key] = filtered
            except Exception as e:
                logger.warning("pitching season 실패 p_no=%d year=%d lt=%s: %s", p_no, year, league_type, e)

        if (i + 1) % 50 == 0:
            _save_json(batter_out, batter_data)
            _save_json(pitcher_out, pitcher_data)
            logger.info("  %d/%d 진행", i + 1, len(player_ids))

    _save_json(batter_out, batter_data)
    _save_json(pitcher_out, pitcher_data)
    logger.info("%d 완료(leagueType=%s): batter=%d, pitcher=%d", year, league_type, len(batter_data), len(pitcher_data))


def backfill_full_player_seasons(
    years: list[int] = None,
    league_type: int = LEAGUE_REGULAR,
):
    """로스터 기반으로 전 선수 batting/pitching 시즌 기록 수집."""
    if years is None:
        years = [2023, 2024, 2025]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    for year in years:
        player_ids = _iter_roster_player_ids(year)
        if not player_ids:
            continue

        suffix = "_spring" if league_type == LEAGUE_EXHIBITION else "_full"
        batter_out = DATA_DIR / f"batter_seasons{suffix}_{year}.json"
        pitcher_out = DATA_DIR / f"pitcher_seasons{suffix}_{year}.json"
        _collect_player_seasons_for_ids(player_ids, year, league_type, batter_out, pitcher_out)


def backfill_spring_core_seasons(years: list[int] = None, days: int = 14):
    if years is None:
        years = [2023, 2024, 2025, 2026]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    for year in years:
        if year >= 2026:
            player_ids = _iter_roster_player_ids(year)
        else:
            player_ids = _iter_early_season_core_player_ids(year, days=days)
        if not player_ids:
            continue

        batter_out = DATA_DIR / f"batter_seasons_spring_{year}.json"
        pitcher_out = DATA_DIR / f"pitcher_seasons_spring_{year}.json"
        _collect_player_seasons_for_ids(player_ids, year, LEAGUE_EXHIBITION, batter_out, pitcher_out)


def backfill_player_days(
    years: list[int] = None,
    mode: str = "batting",
    months: list[int] = None,
):
    """선수별 월간 playerDay 로그 수집."""
    if years is None:
        years = [2023, 2024, 2025]
    if months is None:
        months = list(range(3, 11))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if mode not in {"batting", "pitching"}:
        raise ValueError("mode must be batting or pitching")

    season_file_name = (
        f"batter_seasons_full_{{year}}.json"
        if mode == "batting"
        else f"pitcher_seasons_full_{{year}}.json"
    )
    fallback_file_name = (
        f"batter_seasons_{{year}}.json"
        if mode == "batting"
        else f"pitcher_seasons_{{year}}.json"
    )
    out_file_name = (
        f"batter_days_{{year}}.json"
        if mode == "batting"
        else f"pitcher_days_{{year}}.json"
    )

    for year in years:
        season_file = DATA_DIR / season_file_name.format(year=year)
        if not season_file.exists():
            season_file = DATA_DIR / fallback_file_name.format(year=year)
        if not season_file.exists():
            logger.warning("%d %s season 파일 없음", year, mode)
            continue

        source_data = json.loads(season_file.read_text())
        player_ids = sorted(int(p_no) for p_no in source_data.keys())
        out_file = DATA_DIR / out_file_name.format(year=year)
        day_data = _load_json(out_file, {})

        logger.info(
            "%d %s day backfill: players=%d existing=%d",
            year, mode, len(player_ids), len(day_data),
        )

        for i, p_no in enumerate(player_ids):
            p_key = str(p_no)
            player_months = day_data.get(p_key, {})

            for month in months:
                month_key = f"{month:02d}"
                if month_key in player_months:
                    continue

                try:
                    resp = get_player_day(p_no, m2=mode, year=year, month=month)
                    player_months[month_key] = resp if _has_day_rows(resp) else {}
                except Exception as e:
                    logger.warning(
                        "%s day 실패 p_no=%d year=%d month=%02d: %s",
                        mode, p_no, year, month, e,
                    )

            day_data[p_key] = player_months

            if (i + 1) % 25 == 0:
                _save_json(out_file, day_data)
                logger.info("  %d/%d 진행", i + 1, len(player_ids))

        _save_json(out_file, day_data)
        logger.info("%d %s day 완료", year, mode)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="v6 지원 데이터 백필")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_full = sub.add_parser("full-seasons", help="로스터 기반 season 수집")
    p_full.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    p_full.add_argument("--league-type", type=int, default=LEAGUE_REGULAR)

    p_days = sub.add_parser("player-days", help="playerDay 로그 수집")
    p_days.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    p_days.add_argument("--mode", choices=["batting", "pitching"], required=True)
    p_days.add_argument("--months", nargs="+", type=int, default=list(range(3, 11)))

    p_spring = sub.add_parser("spring-core", help="early-season 핵심 선수 시범경기 요약 수집")
    p_spring.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025, 2026])
    p_spring.add_argument("--days", type=int, default=14)

    args = parser.parse_args()

    if args.cmd == "full-seasons":
        backfill_full_player_seasons(years=args.years, league_type=args.league_type)
    elif args.cmd == "player-days":
        backfill_player_days(years=args.years, mode=args.mode, months=args.months)
    elif args.cmd == "spring-core":
        backfill_spring_core_seasons(years=args.years, days=args.days)
