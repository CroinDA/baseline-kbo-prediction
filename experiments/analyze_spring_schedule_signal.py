"""
2023~2025 시범경기 일정/결과 기반 팀-level 신호가
정규시즌 초반/전체 성적과 얼마나 연관되는지 점검한다.

개인 spring 스탯 API는 신뢰할 수 없으므로 사용하지 않는다.
사용 데이터:
- data/raw/schedules_2023.json
- data/raw/schedules_2024.json
- data/raw/schedules_2025.json

출력:
- experiments/spring_signal_team_rows.csv
- experiments/spring_signal_summary.json
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.constants import LEAGUE_EXHIBITION, LEAGUE_REGULAR, TEAM_CODES


DATA_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "experiments"
YEARS = [2023, 2024, 2025]


def _load_schedule_rows(year: int) -> list[dict]:
    path = DATA_DIR / f"schedules_{year}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for g in data:
        state = int(g.get("state") or 0)
        if state not in {3, 4, 5}:
            continue
        hs = g.get("homeScore")
        aw = g.get("awayScore")
        if hs is None or aw is None:
            continue
        rows.append(
            {
                "year": year,
                "date": int(g["gameDate"]),
                "s_no": int(g["s_no"]),
                "leagueType": int(g["leagueType"]),
                "home": int(g["homeTeam"]),
                "away": int(g["awayTeam"]),
                "homeScore": float(hs),
                "awayScore": float(aw),
            }
        )
    rows.sort(key=lambda x: (x["date"], x["s_no"]))
    return rows


def _team_game_rows(schedule_rows: list[dict], team_code: int) -> list[tuple[float, float, float]]:
    out = []
    for g in schedule_rows:
        if g["home"] == team_code:
            rs, ra = g["homeScore"], g["awayScore"]
        elif g["away"] == team_code:
            rs, ra = g["awayScore"], g["homeScore"]
        else:
            continue
        win_value = 1.0 if rs > ra else 0.5 if rs == ra else 0.0
        out.append((rs, ra, win_value))
    return out


def _summarize_team_games(schedule_rows: list[dict], team_code: int) -> dict:
    team_games = _team_game_rows(schedule_rows, team_code)
    games = len(team_games)
    if games == 0:
        return {
            "G": 0,
            "W": 0.0,
            "wpct": None,
            "rs_pg": None,
            "ra_pg": None,
            "rd_pg": None,
            "pyth": None,
        }
    rs = sum(x[0] for x in team_games)
    ra = sum(x[1] for x in team_games)
    wins = sum(x[2] for x in team_games)
    pyth = (rs**1.83) / ((rs**1.83) + (ra**1.83)) if (rs > 0 or ra > 0) else 0.5
    return {
        "G": games,
        "W": wins,
        "wpct": wins / games,
        "rs_pg": rs / games,
        "ra_pg": ra / games,
        "rd_pg": (rs - ra) / games,
        "pyth": pyth,
    }


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def _rankdata(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    return _pearson(_rankdata(xs), _rankdata(ys))


def build_rows() -> list[dict]:
    rows = []
    for year in YEARS:
        sched = _load_schedule_rows(year)
        spring = [g for g in sched if g["leagueType"] == LEAGUE_EXHIBITION]
        regular = [g for g in sched if g["leagueType"] == LEAGUE_REGULAR]
        for team_code, team_name in TEAM_CODES.items():
            reg_team_games = [g for g in regular if g["home"] == team_code or g["away"] == team_code]
            row = {
                "year": year,
                "team_code": team_code,
                "team_name": team_name,
            }
            row.update({f"spring_{k}": v for k, v in _summarize_team_games(spring, team_code).items()})
            row.update({f"reg10_{k}": v for k, v in _summarize_team_games(reg_team_games[:10], team_code).items()})
            row.update({f"reg20_{k}": v for k, v in _summarize_team_games(reg_team_games[:20], team_code).items()})
            row.update({f"regfull_{k}": v for k, v in _summarize_team_games(reg_team_games, team_code).items()})
            rows.append(row)
    return rows


def build_summary(rows: list[dict]) -> dict:
    spring_metrics = ["spring_wpct", "spring_rs_pg", "spring_ra_pg", "spring_rd_pg", "spring_pyth"]
    targets = ["reg10_wpct", "reg20_wpct", "regfull_wpct", "regfull_rd_pg", "regfull_pyth"]
    correlations = {}
    ranked = []
    for src in spring_metrics:
        correlations[src] = {}
        xs = [float(r[src]) for r in rows]
        for dst in targets:
            ys = [float(r[dst]) for r in rows]
            pearson = _pearson(xs, ys)
            spearman = _spearman(xs, ys)
            correlations[src][dst] = {"pearson": pearson, "spearman": spearman}
            ranked.append(
                {
                    "source": src,
                    "target": dst,
                    "abs_pearson": abs(pearson) if pearson is not None else None,
                    "pearson": pearson,
                    "spearman": spearman,
                }
            )
    ranked = sorted(
        [x for x in ranked if x["abs_pearson"] is not None],
        key=lambda x: x["abs_pearson"],
        reverse=True,
    )
    return {
        "sample_n": len(rows),
        "correlations": correlations,
        "top_abs_pearson_pairs": ranked[:10],
    }


def save_rows_csv(rows: list[dict]) -> None:
    out_path = OUT_DIR / "spring_signal_team_rows.csv"
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_json(summary: dict) -> None:
    out_path = OUT_DIR / "spring_signal_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    rows = build_rows()
    summary = build_summary(rows)
    save_rows_csv(rows)
    save_summary_json(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
