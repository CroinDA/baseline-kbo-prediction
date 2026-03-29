"""
1군 로스터 백필 — 경기일별 로스터 스냅샷 수집.

경기가 있는 날짜별로 player_roster API를 호출하여 저장.
저장 형식: {date_str: [{name, p_no, t_code, pj_date}, ...]}
"""
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.constants import LEAGUE_REGULAR
from data.collector import get_player_roster

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _get_game_dates(year: int) -> list[str]:
    """해당 연도의 고유 경기일 목록 (YYYY-MM-DD)."""
    f = DATA_DIR / f"schedules_{year}.json"
    if not f.exists():
        return []
    games = json.loads(f.read_text())
    dates = set()
    for g in games:
        if g.get("leagueType") != LEAGUE_REGULAR:
            continue
        if g.get("homeScore") is None:
            continue
        gd = g.get("gameDate", 0)
        if gd > 0:
            dates.add(datetime.fromtimestamp(gd).strftime("%Y-%m-%d"))
    return sorted(dates)


def backfill_roster(years: list[int] = None):
    """경기일별 1군 로스터 스냅샷 수집."""
    if years is None:
        years = [2023, 2024, 2025]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    for year in years:
        out_file = DATA_DIR / f"rosters_{year}.json"

        # 기존 데이터 로드 (이어받기)
        if out_file.exists():
            existing = json.loads(out_file.read_text())
            logger.info("%d: 기존 %d일 로스터 데이터 있음", year, len(existing))
        else:
            existing = {}

        dates = _get_game_dates(year)
        logger.info("%d: %d 경기일 중 %d일 수집 필요",
                     year, len(dates), len(dates) - len(existing))

        new_count = 0
        for i, date_str in enumerate(dates):
            if date_str in existing:
                continue

            try:
                resp = get_player_roster(date_str)
                players = []
                for k, v in resp.items():
                    if k.isdigit() and isinstance(v, dict):
                        players.append({
                            "name": v.get("name"),
                            "p_no": v.get("p_no"),
                            "t_code": v.get("t_code"),
                            "pj_date": v.get("pj_date"),
                        })

                existing[date_str] = players
                new_count += 1

                if new_count % 20 == 0:
                    # 중간 저장
                    out_file.write_text(
                        json.dumps(existing, ensure_ascii=False, indent=1)
                    )
                    logger.info("  %d: %d/%d 완료 (중간저장)", year, new_count,
                                len(dates) - len(existing) + new_count)

            except Exception as e:
                logger.warning("로스터 조회 실패 (%s): %s", date_str, e)

        # 최종 저장
        out_file.write_text(json.dumps(existing, ensure_ascii=False, indent=1))
        logger.info("%d: 완료 — 총 %d일 로스터 저장 (신규 %d)",
                     year, len(existing), new_count)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="1군 로스터 백필")
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    args = parser.parse_args()
    backfill_roster(years=args.years)
