"""
v6 학습 데이터셋 구축 — 시점 누적 선수스탯 + 로스터 집계.

핵심 변화:
1. 선수 개인 스탯을 연도 최종값이 아닌 "경기 직전까지의 누적"으로 계산
2. 이전 시즌 prior와 당해 season-to-date를 블렌딩
3. 1군 로스터 전체를 활용해 불펜/벤치 집계 피처 생성
4. 불펜은 필승조/추격조/롱릴리프로 구분하고 연투 패널티 반영
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
    SPRING_FULL_WEIGHT_DAYS,
    SPRING_HALF_WEIGHT_DAYS,
    SPRING_BATTER_PA_EQUIV,
    SPRING_PITCHER_IP_EQUIV,
    SPRING_BATTER_ANCHOR,
    SPRING_PITCHER_ANCHOR,
)
from elo.engine import EloEngine
from features.expanded import extract_batter_stats_v5, extract_sp_stats_v5

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
_season_start_cache: dict[int, date] = {}

PRIOR_YEAR_WEIGHTS = [1.0, 0.7, 0.4]
BENCH_WEIGHTS = [1.0, 0.7, 0.4]
BULLPEN_GROUP_WEIGHTS = {
    "core": [1.0, 0.8, 0.6],
    "chase": [1.0, 0.7, 0.5, 0.3],
    "long": [1.0, 0.6],
}
TUNED_BULLPEN_GROUP_WEIGHTS = {
    "core": [1.2, 0.9, 0.6],
    "chase": [1.0, 0.6, 0.35, 0.2],
    "long": [1.0, 0.5],
}
CONSEC_MULTIPLIERS = {
    1: 1.00,  # 정상
    2: 0.90,  # 2연투 예정
    3: 0.60,  # 3연투 예정
    4: 0.35,  # 4연투 이상 예정
}
TUNED_CONSEC_MULTIPLIERS = {
    1: 1.00,
    2: 0.93,
    3: 0.45,
    4: 0.15,
}
ROLE_QUALITY_MULTIPLIERS = {
    "core": 1.00,
    "chase": 0.90,
    "long": 0.82,
}

BATTER_PRIOR_SHRINK_PA = 250.0
BATTER_CURRENT_SHRINK_PA = 180.0
PITCHER_PRIOR_SHRINK_OUTS = 270.0
PITCHER_CURRENT_SHRINK_OUTS = 180.0

LEAGUE_AVG_BATTER = {
    "wrcplus": 100.0,
    "avg": 0.260,
    "obp": 0.330,
    "slg": 0.400,
    "hr_rate": 0.025,
    "bb_rate": 0.085,
    "k_rate": 0.200,
    "bbk_ratio": 0.425,
    "pa": 0.0,
}

LEAGUE_AVG_SP = {
    "era": 4.50,
    "fip": 4.50,
    "whip": 1.40,
    "k9": 7.0,
    "bb9": 3.5,
    "hr9": 1.0,
    "obp_against": 0.330,
    "kbb_ratio": 2.0,
    "np_per_app": 85.0,
    "ip": 0.0,
    "war": 1.0,
    "quality": 100.0,
}

VARIANT_CONFIGS = {
    "v6": {
        "include_bench": True,
        "include_sp_war": False,
        "stabilize_rates": False,
        "tune_bullpen": False,
    },
    "v61": {
        "include_bench": False,
        "include_sp_war": False,
        "stabilize_rates": False,
        "tune_bullpen": False,
    },
    "v62": {
        "include_bench": False,
        "include_sp_war": False,
        "stabilize_rates": True,
        "tune_bullpen": False,
    },
    "v63": {
        "include_bench": False,
        "include_sp_war": True,
        "stabilize_rates": True,
        "tune_bullpen": False,
    },
    "v64": {
        "include_bench": False,
        "include_sp_war": True,
        "stabilize_rates": True,
        "tune_bullpen": True,
        "use_spring": False,
    },
    "v65": {
        "include_bench": False,
        "include_sp_war": True,
        "stabilize_rates": True,
        "tune_bullpen": True,
        "use_spring": False,
    },
}


def feature_names_v6(include_bench: bool = True, include_sp_war: bool = False) -> list[str]:
    names = []

    batter_stats = ["wrcplus", "avg", "obp", "slg", "hr_rate", "bb_rate", "k_rate", "bbk_ratio", "pa"]
    sp_stats = ["era", "fip", "whip", "k9", "bb9", "hr9", "obp_against", "kbb_ratio", "np_per_app", "ip"]
    if include_sp_war:
        sp_stats.append("war")
    team_stats = ["elo", "pyth_exp", "wpct", "bp_load", "rest_days"]
    seq_stats = ["scoring_trend", "lineup_wrc_delta"]
    env_stats = ["temperature", "humidity", "wind_speed", "is_night", "park_factor"]
    bullpen_stats = [
        "bp_core_strength",
        "bp_chase_strength",
        "bp_long_strength",
        "bp_fatigue_index",
        "bp_3plus_count",
    ]
    availability_stats = ["lineup_gap", "roster_loss_hit", "roster_loss_pitch"]
    bench_stats = ["bench_offense_top3", "bench_depth"]

    for side in ["home", "away"]:
        for i in range(1, 10):
            for stat in batter_stats:
                names.append(f"{side}_b{i}_{stat}")

    for side in ["home", "away"]:
        for stat in sp_stats:
            names.append(f"{side}_sp_{stat}")

    for side in ["home", "away"]:
        for stat in team_stats:
            names.append(f"{side}_{stat}")

    for side in ["home", "away"]:
        for stat in seq_stats:
            names.append(f"{side}_{stat}")

    names.extend(env_stats)

    for side in ["home", "away"]:
        for stat in bullpen_stats:
            names.append(f"{side}_{stat}")

    for side in ["home", "away"]:
        for stat in availability_stats:
            names.append(f"{side}_{stat}")

    if include_bench:
        for side in ["home", "away"]:
            for stat in bench_stats:
                names.append(f"{side}_{stat}")

    return names


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _safe_int(value, default=0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ip_to_outs(ip_value) -> int:
    if ip_value in (None, "", 0):
        return 0

    s = str(ip_value)
    if "." not in s:
        return int(float(s) * 3)

    whole, frac = s.split(".", 1)
    outs = int(whole) * 3
    if frac == "1":
        outs += 1
    elif frac == "2":
        outs += 2
    else:
        outs += int(round(float("0." + frac) * 3))
    return outs


def _outs_to_ip(outs: float) -> float:
    if outs <= 0:
        return 0.0
    whole = int(outs // 3)
    rem = int(outs % 3)
    return whole + rem / 3.0


def _parse_game_date(year: int, mmdd: str) -> date | None:
    if not mmdd:
        return None
    try:
        month, day = str(mmdd).split("-")
        return date(year, int(month), int(day))
    except (TypeError, ValueError):
        return None


def _load_schedules(years: list[int]) -> list[dict]:
    all_games = []
    for year in years:
        f = DATA_DIR / f"schedules_{year}.json"
        if not f.exists():
            logger.warning("일정 파일 없음: %s", f.name)
            continue
        games = json.loads(f.read_text())
        for g in games:
            if g.get("leagueType") != LEAGUE_REGULAR:
                continue
            if g.get("homeScore") is None or g.get("awayScore") is None:
                continue
            gd = g.get("gameDate", 0)
            g["_year"] = int(g.get("year", year))
            g["_sort_key"] = gd
            g["_date"] = datetime.fromtimestamp(gd).date() if gd else None
            all_games.append(g)

    all_games.sort(key=lambda x: (x["_year"], x["_sort_key"], x.get("s_no", 0)))
    logger.info("완료 경기 로드: %d", len(all_games))
    return all_games


def _load_lineup_db(years: list[int]) -> dict[int, dict[int, list[dict]]]:
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
    logger.info("라인업 로드: %d경기", len(lineup_db))
    return lineup_db


def _load_roster_db(years: list[int]) -> dict[str, dict[int, list[int]]]:
    roster_db = {}
    for year in years:
        f = DATA_DIR / f"rosters_{year}.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        for date_str, players in data.items():
            by_team = defaultdict(list)
            if isinstance(players, list):
                for rec in players:
                    p_no = rec.get("p_no")
                    t_code = rec.get("t_code")
                    if p_no and t_code:
                        by_team[int(t_code)].append(int(p_no))
            roster_db[date_str] = dict(by_team)
    logger.info("로스터 스냅샷 로드: %d일", len(roster_db))
    return roster_db


def _load_batter_summary_db(years: list[int]) -> dict[tuple[int, int], dict]:
    db = {}
    for year in years:
        f = DATA_DIR / f"batter_seasons_full_{year}.json"
        if not f.exists():
            f = DATA_DIR / f"batter_seasons_{year}.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        for p_no_str, resp in data.items():
            basic_list = resp.get("basic", {}).get("list", [])
            for rec in basic_list:
                if str(rec.get("year")) == str(year):
                    db[(int(p_no_str), year)] = rec
                    break
    logger.info("타자 시즌 요약 로드: %d명-시즌", len(db))
    return db


def _load_pitcher_summary_db(years: list[int]) -> dict[tuple[int, int], dict]:
    db = {}
    for year in years:
        f = DATA_DIR / f"pitcher_seasons_full_{year}.json"
        if not f.exists():
            f = DATA_DIR / f"pitcher_seasons_{year}.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        for p_no_str, resp in data.items():
            basic = resp.get("basic", {}).get("list", [])
            deepen = resp.get("deepen", {}).get("list", [])
            basic_rec = next((r for r in basic if str(r.get("year")) == str(year)), None)
            deepen_rec = next((r for r in deepen if str(r.get("year")) == str(year)), None)
            if basic_rec:
                db[(int(p_no_str), year)] = {"basic": basic_rec, "deepen": deepen_rec or {}}
    logger.info("투수 시즌 요약 로드: %d명-시즌", len(db))
    return db


def _load_batter_spring_summary_db(years: list[int]) -> dict[tuple[int, int], dict]:
    db = {}
    for year in years:
        f = DATA_DIR / f"batter_seasons_spring_{year}.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        for p_no_str, resp in data.items():
            basic_list = resp.get("basic", {}).get("list", [])
            deepen_list = resp.get("deepen", {}).get("list", [])
            basic_rec = next((r for r in basic_list if str(r.get("year")) == str(year)), None)
            deepen_rec = next((r for r in deepen_list if str(r.get("year")) == str(year)), None)
            if basic_rec or deepen_rec:
                db[(int(p_no_str), year)] = {"basic": basic_rec or {}, "deepen": deepen_rec or {}}
    logger.info("타자 시범경기 요약 로드: %d명-시즌", len(db))
    return db


def _load_pitcher_spring_summary_db(years: list[int]) -> dict[tuple[int, int], dict]:
    db = {}
    for year in years:
        f = DATA_DIR / f"pitcher_seasons_spring_{year}.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        for p_no_str, resp in data.items():
            basic_list = resp.get("basic", {}).get("list", [])
            deepen_list = resp.get("deepen", {}).get("list", [])
            basic_rec = next((r for r in basic_list if str(r.get("year")) == str(year)), None)
            deepen_rec = next((r for r in deepen_list if str(r.get("year")) == str(year)), None)
            if basic_rec or deepen_rec:
                db[(int(p_no_str), year)] = {"basic": basic_rec or {}, "deepen": deepen_rec or {}}
    logger.info("투수 시범경기 요약 로드: %d명-시즌", len(db))
    return db


def _get_regular_season_start(year: int) -> date:
    cached = _season_start_cache.get(year)
    if cached is not None:
        return cached
    f = DATA_DIR / f"schedules_{year}.json"
    if f.exists():
        rows = json.loads(f.read_text())
        timestamps = [r.get("gameDate") for r in rows if r.get("leagueType") == LEAGUE_REGULAR and r.get("gameDate")]
        if timestamps:
            season_start = datetime.fromtimestamp(min(timestamps)).date()
            _season_start_cache[year] = season_start
            return season_start
    season_start = date(year, 3, 28)
    _season_start_cache[year] = season_start
    return season_start


def _get_spring_decay(year: int, game_date: date | None) -> float:
    if not game_date:
        return 0.0
    season_start = _get_regular_season_start(year)
    delta_days = (game_date - season_start).days
    if delta_days < 0:
        return 0.0
    if delta_days < SPRING_FULL_WEIGHT_DAYS:
        return 1.0
    if delta_days < SPRING_HALF_WEIGHT_DAYS:
        return 0.5
    return 0.0


def _blend_rate_dicts(base_stats: dict, spring_stats: dict, spring_weight: float) -> dict:
    if spring_weight <= 0.0 or not spring_stats:
        return dict(base_stats)
    out = dict(base_stats)
    for key, base_val in base_stats.items():
        if key not in spring_stats:
            continue
        out[key] = (1.0 - spring_weight) * float(base_val) + spring_weight * float(spring_stats[key])
    return out


def _parse_player_day_response(resp: dict, year: int, p_no: int) -> list[dict]:
    rows = []
    for key, value in resp.items():
        if not (str(key).isdigit() and isinstance(value, dict)):
            continue
        s_no = _safe_int(value.get("s_no"), _safe_int(key))
        if not s_no:
            continue
        record = dict(value)
        record["s_no"] = s_no
        record["p_no"] = _safe_int(record.get("p_no"), p_no)
        record["_date"] = _parse_game_date(year, record.get("gameDate"))
        rows.append(record)
    rows.sort(key=lambda r: (r.get("_date") or date(year, 1, 1), r["s_no"]))
    return rows


def _load_day_index(years: list[int], mode: str) -> dict[int, list[dict]]:
    file_name = "batter_days_{year}.json" if mode == "batting" else "pitcher_days_{year}.json"
    by_game = defaultdict(list)

    for year in years:
        f = DATA_DIR / file_name.format(year=year)
        if not f.exists():
            logger.warning("%s day 파일 없음: %s", mode, f.name)
            continue
        data = json.loads(f.read_text())
        for p_no_str, month_map in data.items():
            for resp in month_map.values():
                if not isinstance(resp, dict):
                    continue
                for rec in _parse_player_day_response(resp, year, int(p_no_str)):
                    by_game[rec["s_no"]].append(rec)

    total_rows = sum(len(v) for v in by_game.values())
    logger.info("%s day index 로드: %d games / %d rows", mode, len(by_game), total_rows)
    return dict(by_game)


def _batter_state_default():
    return {
        "pa": 0.0,
        "epa": 0.0,
        "ab": 0.0,
        "h": 0.0,
        "double": 0.0,
        "triple": 0.0,
        "hr": 0.0,
        "bb": 0.0,
        "hp": 0.0,
        "so": 0.0,
        "sf": 0.0,
        "tb": 0.0,
    }


def _pitcher_state_default():
    return {
        "g": 0.0,
        "gs": 0.0,
        "gr": 0.0,
        "gf": 0.0,
        "s": 0.0,
        "hd": 0.0,
        "outs": 0.0,
        "relief_outs": 0.0,
        "er": 0.0,
        "ab": 0.0,
        "h": 0.0,
        "double": 0.0,
        "triple": 0.0,
        "hr": 0.0,
        "bb": 0.0,
        "hp": 0.0,
        "so": 0.0,
        "tb": 0.0,
        "np": 0.0,
        "war": 0.0,
        "appearance_dates": set(),
    }


def _empty_batter_counts():
    return _batter_state_default()


def _empty_pitcher_counts():
    state = _pitcher_state_default()
    state["appearance_dates"] = set()
    return state


def _extract_batter_prior_counts(rec: dict) -> dict:
    return {
        "pa": _safe_float(rec.get("PA")),
        "epa": _safe_float(rec.get("ePA") or rec.get("PA")),
        "ab": _safe_float(rec.get("AB")),
        "h": _safe_float(rec.get("H")),
        "double": _safe_float(rec.get("2B")),
        "triple": _safe_float(rec.get("3B")),
        "hr": _safe_float(rec.get("HR")),
        "bb": _safe_float(rec.get("BB")),
        "hp": _safe_float(rec.get("HP")),
        "so": _safe_float(rec.get("SO")),
        "sf": _safe_float(rec.get("SF")),
        "tb": _safe_float(rec.get("TB")),
    }


def _extract_pitcher_prior_counts(rec: dict, deepen_rec: dict) -> dict:
    g = _safe_float(rec.get("G"))
    gs = _safe_float(rec.get("GS"))
    gr = _safe_float(rec.get("GR") or max(g - gs, 0.0))
    outs = float(_ip_to_outs(rec.get("IP")))
    relief_outs = outs * (gr / g) if g > 0 else 0.0

    double = _safe_float(rec.get("2B"))
    triple = _safe_float(rec.get("3B"))
    hr = _safe_float(rec.get("HR"))
    h = _safe_float(rec.get("H"))
    tb = h + double + 2 * triple + 3 * hr

    return {
        "g": g,
        "gs": gs,
        "gr": gr,
        "gf": _safe_float(rec.get("GF")),
        "s": _safe_float(rec.get("S")),
        "hd": _safe_float(rec.get("HD")),
        "outs": outs,
        "relief_outs": relief_outs,
        "er": _safe_float(rec.get("ER")),
        "ab": _safe_float(rec.get("AB") or rec.get("TBF")),
        "h": h,
        "double": double,
        "triple": triple,
        "hr": hr,
        "bb": _safe_float(rec.get("BB")),
        "hp": _safe_float(rec.get("HP")),
        "so": _safe_float(rec.get("SO")),
        "tb": tb,
        "np": 0.0,
        "war": _safe_float(rec.get("WAR"), LEAGUE_AVG_SP["war"]),
        "appearance_dates": set(),
    }


def _scaled_add(target: dict, source: dict, weight: float):
    for key, value in source.items():
        if key == "appearance_dates":
            continue
        target[key] += float(value) * weight


def _league_wrcplus_from_ops(ops: float) -> float:
    if ops <= 0:
        return 80.0
    return max(40.0, min(220.0, ops / 0.730 * 100.0))


def _shrink_toward_anchor(value: float, sample: float, anchor: float, scale: float) -> float:
    if sample <= 0:
        return float(anchor)
    weight = sample / (sample + scale)
    return float(anchor * (1.0 - weight) + value * weight)


def _combine_counts(prior: dict, current: dict) -> dict:
    combined = {}
    for key in prior.keys():
        if key == "appearance_dates":
            continue
        combined[key] = float(prior.get(key, 0.0)) + float(current.get(key, 0.0))
    return combined


def _build_batter_prior_cache(years: list[int], summary_db: dict) -> dict:
    cache = {}
    years_set = set(years)

    for p_no, year in sorted({(p_no, year) for p_no, year in summary_db.keys() if year in years_set}):
        pass

    player_ids = {p_no for p_no, _ in summary_db.keys()}
    for year in years:
        for p_no in player_ids:
            agg = _empty_batter_counts()
            for offset, weight in enumerate(PRIOR_YEAR_WEIGHTS, start=1):
                prev_year = year - offset
                rec = summary_db.get((p_no, prev_year))
                if not rec:
                    continue
                _scaled_add(agg, _extract_batter_prior_counts(rec), weight)
            cache[(p_no, year)] = agg
    return cache


def _build_pitcher_prior_cache(years: list[int], summary_db: dict) -> dict:
    cache = {}
    player_ids = {p_no for p_no, _ in summary_db.keys()}
    for year in years:
        for p_no in player_ids:
            agg = _empty_pitcher_counts()
            for offset, weight in enumerate(PRIOR_YEAR_WEIGHTS, start=1):
                prev_year = year - offset
                rec = summary_db.get((p_no, prev_year))
                if not rec:
                    continue
                _scaled_add(agg, _extract_pitcher_prior_counts(rec["basic"], rec.get("deepen", {})), weight)
            cache[(p_no, year)] = agg
    return cache


def _batter_rates_from_counts(counts: dict) -> tuple[dict, float]:
    pa = float(counts.get("epa") or counts.get("pa") or 0.0)
    ab = float(counts.get("ab", 0.0))
    h = float(counts.get("h", 0.0))
    bb = float(counts.get("bb", 0.0))
    hp = float(counts.get("hp", 0.0))
    sf = float(counts.get("sf", 0.0))
    tb = float(counts.get("tb", 0.0))

    if pa <= 0 or ab <= 0:
        return dict(LEAGUE_AVG_BATTER), 0.0

    obp_denom = ab + bb + hp + sf
    rates = {
        "wrcplus": _league_wrcplus_from_ops(
            ((h + bb + hp) / obp_denom if obp_denom > 0 else LEAGUE_AVG_BATTER["obp"])
            + (tb / ab if ab > 0 else LEAGUE_AVG_BATTER["slg"])
        ),
        "avg": h / ab if ab > 0 else LEAGUE_AVG_BATTER["avg"],
        "obp": (h + bb + hp) / obp_denom if obp_denom > 0 else LEAGUE_AVG_BATTER["obp"],
        "slg": tb / ab if ab > 0 else LEAGUE_AVG_BATTER["slg"],
        "hr_rate": float(counts.get("hr", 0.0)) / pa if pa > 0 else LEAGUE_AVG_BATTER["hr_rate"],
        "bb_rate": bb / pa if pa > 0 else LEAGUE_AVG_BATTER["bb_rate"],
        "k_rate": float(counts.get("so", 0.0)) / pa if pa > 0 else LEAGUE_AVG_BATTER["k_rate"],
        "bbk_ratio": min(3.0, bb / max(float(counts.get("so", 0.0)), 1.0)),
        "pa": pa,
    }
    return rates, pa


def _pitcher_rates_from_counts(counts: dict) -> tuple[dict, float]:
    outs = float(counts.get("outs", 0.0))
    ip = _outs_to_ip(outs)
    if outs <= 0 or ip <= 0:
        return dict(LEAGUE_AVG_SP), 0.0

    h = float(counts.get("h", 0.0))
    bb = float(counts.get("bb", 0.0))
    hp = float(counts.get("hp", 0.0))
    ab = float(counts.get("ab", 0.0))
    er = float(counts.get("er", 0.0))
    hr = float(counts.get("hr", 0.0))
    so = float(counts.get("so", 0.0))
    tb = float(counts.get("tb", 0.0))

    era = er * 9.0 / ip
    whip = (h + bb) / ip
    k9 = so * 9.0 / ip
    bb9 = bb * 9.0 / ip
    hr9 = hr * 9.0 / ip
    fip = ((13 * hr) + (3 * (bb + hp)) - (2 * so)) / ip + FIP_CONSTANT_DEFAULT
    obp_denom = ab + bb + hp
    obp_against = (h + bb + hp) / obp_denom if obp_denom > 0 else LEAGUE_AVG_SP["obp_against"]
    quality = 100.0 - 10.0 * (fip - 4.20) - 8.0 * (whip - 1.35) + 1.2 * (k9 - bb9)
    quality = max(20.0, min(180.0, quality))
    g = float(counts.get("g", 0.0))
    np_total = float(counts.get("np", 0.0))
    np_per_app = np_total / g if g > 0 and np_total > 0 else LEAGUE_AVG_SP["np_per_app"]

    rates = {
        "era": float(era),
        "fip": float(fip),
        "whip": float(whip),
        "k9": float(k9),
        "bb9": float(bb9),
        "hr9": float(hr9),
        "obp_against": float(obp_against),
        "kbb_ratio": min(10.0, so / max(bb, 1.0)),
        "np_per_app": float(np_per_app),
        "ip": float(ip),
        "war": float(counts.get("war", LEAGUE_AVG_SP["war"])),
        "quality": float(quality),
    }
    return rates, outs


def _batter_snapshot(
    p_no: int,
    year: int,
    prior_cache: dict,
    current_state: dict,
    stabilize_rates: bool = False,
) -> dict:
    prior = prior_cache.get((p_no, year), _empty_batter_counts())
    current = current_state.get(p_no, _batter_state_default())
    if not stabilize_rates:
        combined = _combine_counts(prior, current)
        pa = combined["epa"] if combined["epa"] > 0 else combined["pa"]
        ab = combined["ab"]
        h = combined["h"]
        bb = combined["bb"]
        hp = combined["hp"]
        sf = combined["sf"]
        tb = combined["tb"]

        if pa <= 0 or ab <= 0:
            return dict(LEAGUE_AVG_BATTER)

        avg = h / ab if ab > 0 else LEAGUE_AVG_BATTER["avg"]
        obp_denom = ab + bb + hp + sf
        obp = (h + bb + hp) / obp_denom if obp_denom > 0 else LEAGUE_AVG_BATTER["obp"]
        slg = tb / ab if ab > 0 else LEAGUE_AVG_BATTER["slg"]
        hr_rate = combined["hr"] / pa if pa > 0 else LEAGUE_AVG_BATTER["hr_rate"]
        bb_rate = bb / pa if pa > 0 else LEAGUE_AVG_BATTER["bb_rate"]
        k_rate = combined["so"] / pa if pa > 0 else LEAGUE_AVG_BATTER["k_rate"]
        bbk_ratio = min(3.0, bb / max(combined["so"], 1.0))
        wrcplus = _league_wrcplus_from_ops(obp + slg)

        return {
            "wrcplus": float(wrcplus),
            "avg": float(avg),
            "obp": float(obp),
            "slg": float(slg),
            "hr_rate": float(hr_rate),
            "bb_rate": float(bb_rate),
            "k_rate": float(k_rate),
            "bbk_ratio": float(bbk_ratio),
            "pa": float(current.get("epa") or current.get("pa") or 0.0),
        }

    prior_rates, prior_pa = _batter_rates_from_counts(prior)
    current_rates, current_pa = _batter_rates_from_counts(current)

    out = {}
    for stat in ["wrcplus", "avg", "obp", "slg", "hr_rate", "bb_rate", "k_rate", "bbk_ratio"]:
        prior_anchor = _shrink_toward_anchor(
            prior_rates[stat], prior_pa, LEAGUE_AVG_BATTER[stat], BATTER_PRIOR_SHRINK_PA
        )
        out[stat] = _shrink_toward_anchor(
            current_rates[stat], current_pa, prior_anchor, BATTER_CURRENT_SHRINK_PA
        )
    out["pa"] = float(current.get("epa") or current.get("pa") or 0.0)
    return out


def _pitcher_snapshot(
    p_no: int,
    year: int,
    prior_cache: dict,
    current_state: dict,
    stabilize_rates: bool = False,
) -> dict:
    prior = prior_cache.get((p_no, year), _empty_pitcher_counts())
    current = current_state.get(p_no, _pitcher_state_default())
    if not stabilize_rates:
        combined = _combine_counts(prior, current)
        outs = combined["outs"]
        ip = _outs_to_ip(outs)
        if outs <= 0 or ip <= 0:
            return dict(LEAGUE_AVG_SP)

        h = combined["h"]
        bb = combined["bb"]
        hp = combined["hp"]
        ab = combined["ab"]
        er = combined["er"]
        hr = combined["hr"]
        so = combined["so"]
        era = er * 9.0 / ip
        whip = (h + bb) / ip
        k9 = so * 9.0 / ip
        bb9 = bb * 9.0 / ip
        hr9 = hr * 9.0 / ip
        fip = ((13 * hr) + (3 * (bb + hp)) - (2 * so)) / ip + FIP_CONSTANT_DEFAULT
        obp_denom = ab + bb + hp
        obp_against = (h + bb + hp) / obp_denom if obp_denom > 0 else LEAGUE_AVG_SP["obp_against"]
        g = current.get("g", 0.0)
        np_total = current.get("np", 0.0)
        np_per_app = np_total / g if g > 0 and np_total > 0 else LEAGUE_AVG_SP["np_per_app"]
        quality = 100.0 - 10.0 * (fip - 4.20) - 8.0 * (whip - 1.35) + 1.2 * (k9 - bb9)
        quality = max(20.0, min(180.0, quality))

        return {
            "era": float(era),
            "fip": float(fip),
            "whip": float(whip),
            "k9": float(k9),
            "bb9": float(bb9),
            "hr9": float(hr9),
            "obp_against": float(obp_against),
            "kbb_ratio": float(min(10.0, so / max(bb, 1.0))),
            "np_per_app": float(np_per_app),
            "ip": float(_outs_to_ip(current.get("outs", 0.0))),
            "war": float(prior.get("war", LEAGUE_AVG_SP["war"])),
            "quality": float(quality),
        }

    prior_rates, prior_outs = _pitcher_rates_from_counts(prior)
    current_rates, current_outs = _pitcher_rates_from_counts(current)

    out = {}
    for stat in ["era", "fip", "whip", "k9", "bb9", "hr9", "obp_against", "kbb_ratio", "quality"]:
        prior_anchor = _shrink_toward_anchor(
            prior_rates[stat], prior_outs, LEAGUE_AVG_SP[stat], PITCHER_PRIOR_SHRINK_OUTS
        )
        out[stat] = _shrink_toward_anchor(
            current_rates[stat], current_outs, prior_anchor, PITCHER_CURRENT_SHRINK_OUTS
        )
    current_g = float(current.get("g", 0.0))
    current_np = float(current.get("np", 0.0))
    raw_np_per_app = current_np / current_g if current_g > 0 and current_np > 0 else LEAGUE_AVG_SP["np_per_app"]
    out["np_per_app"] = _shrink_toward_anchor(
        raw_np_per_app, current_g, LEAGUE_AVG_SP["np_per_app"], 4.0
    )
    prior_war_anchor = _shrink_toward_anchor(
        prior_rates["war"], prior_outs, LEAGUE_AVG_SP["war"], PITCHER_PRIOR_SHRINK_OUTS
    )
    workload_factor = min(current_outs / 540.0, 0.35)
    quality_adjust = (out["quality"] - 100.0) / 40.0
    out["war"] = float(max(-2.0, min(8.0, prior_war_anchor + workload_factor * quality_adjust)))
    out["ip"] = float(_outs_to_ip(current.get("outs", 0.0)))
    return out


def _spring_batter_snapshot(rec: dict | None) -> tuple[dict, float]:
    if not rec:
        return dict(LEAGUE_AVG_BATTER), 0.0
    basic_rec = rec.get("basic", {})
    deepen_rec = rec.get("deepen", {})
    stats = extract_batter_stats_v5(basic_rec, deepen_rec)
    pa = _safe_float(basic_rec.get("PA") or basic_rec.get("ePA") or deepen_rec.get("PA"))
    stats["pa"] = pa
    return stats, pa


def _spring_pitcher_snapshot(rec: dict | None) -> tuple[dict, float]:
    if not rec:
        return dict(LEAGUE_AVG_SP), 0.0
    basic_rec = rec.get("basic", {})
    deepen_rec = rec.get("deepen", {})
    stats = extract_sp_stats_v5(basic_rec, deepen_rec)
    ip = _safe_float(basic_rec.get("IP") or deepen_rec.get("IP"))
    stats["whip"] = _safe_float(basic_rec.get("WHIP"), LEAGUE_AVG_SP["whip"])
    stats["hr9"] = (_safe_float(basic_rec.get("HR")) / ip * 9.0) if ip > 0 else LEAGUE_AVG_SP["hr9"]
    stats["obp_against"] = _safe_float(stats.get("obp_against"), LEAGUE_AVG_SP["obp_against"])
    stats["ip"] = ip
    stats["war"] = _safe_float(basic_rec.get("WAR"), LEAGUE_AVG_SP["war"])
    return stats, ip


def _maybe_blend_spring_batter_snapshot(base_stats: dict, spring_db: dict, p_no: int, year: int, game_date: date | None) -> dict:
    decay = _get_spring_decay(year, game_date)
    if decay <= 0.0:
        return dict(base_stats)
    spring_stats, spring_pa = _spring_batter_snapshot(spring_db.get((p_no, year)))
    effective_pa = spring_pa * SPRING_BATTER_PA_EQUIV
    if effective_pa <= 0.0:
        return dict(base_stats)
    spring_weight = decay * (effective_pa / (effective_pa + SPRING_BATTER_ANCHOR))
    return _blend_rate_dicts(base_stats, spring_stats, spring_weight)


def _maybe_blend_spring_pitcher_snapshot(base_stats: dict, spring_db: dict, p_no: int, year: int, game_date: date | None) -> dict:
    decay = _get_spring_decay(year, game_date)
    if decay <= 0.0:
        return dict(base_stats)
    spring_stats, spring_ip = _spring_pitcher_snapshot(spring_db.get((p_no, year)))
    effective_ip = spring_ip * SPRING_PITCHER_IP_EQUIV
    if effective_ip <= 0.0:
        return dict(base_stats)
    spring_weight = decay * (effective_ip / (effective_ip + SPRING_PITCHER_ANCHOR))
    return _blend_rate_dicts(base_stats, spring_stats, spring_weight)


def _projected_consecutive(relief_dates: set[date], current_date: date | None) -> int:
    if not current_date:
        return 1
    streak = 0
    cursor = current_date - timedelta(days=1)
    while cursor in relief_dates:
        streak += 1
        cursor -= timedelta(days=1)
    return min(streak + 1, 4)


def _bullpen_role(snapshot: dict, combined_counts: dict) -> str:
    relief_app = combined_counts.get("gr", 0.0)
    saves = combined_counts.get("s", 0.0)
    holds = combined_counts.get("hd", 0.0)
    gf = combined_counts.get("gf", 0.0)
    relief_outs = combined_counts.get("relief_outs", 0.0)
    high_lev = saves + holds + 0.5 * gf
    avg_outs = relief_outs / relief_app if relief_app > 0 else 0.0

    if relief_app <= 0:
        return "chase"
    if avg_outs >= 4.5 and high_lev < 8:
        return "long"
    if high_lev >= 8 or saves >= 4 or holds >= 6:
        return "core"
    return "chase"


def _weighted_top(values: list[float], weights: list[float]) -> float:
    if not values:
        return 100.0
    values = sorted(values, reverse=True)[: len(weights)]
    use_weights = weights[: len(values)]
    denom = sum(use_weights)
    if denom <= 0:
        return float(np.mean(values))
    return float(sum(v * w for v, w in zip(values, use_weights)) / denom)


def _build_bullpen_features(
    team_code: int,
    starting_pitcher: int | None,
    year: int,
    roster_players: list[int],
    current_date: date | None,
    pitcher_prior_cache: dict,
    pitcher_state: dict,
    stabilize_rates: bool = False,
    tuned: bool = False,
) -> dict:
    grouped = defaultdict(list)
    fatigue_losses = []
    three_plus = 0
    weight_map = TUNED_BULLPEN_GROUP_WEIGHTS if tuned else BULLPEN_GROUP_WEIGHTS
    consec_multipliers = TUNED_CONSEC_MULTIPLIERS if tuned else CONSEC_MULTIPLIERS

    for p_no in roster_players:
        if p_no == starting_pitcher:
            continue
        snapshot = _pitcher_snapshot(
            p_no, year, pitcher_prior_cache, pitcher_state, stabilize_rates=stabilize_rates
        )
        prior = pitcher_prior_cache.get((p_no, year), _empty_pitcher_counts())
        current = pitcher_state.get(p_no, _pitcher_state_default())
        combined = _combine_counts(prior, current)

        relief_app = combined.get("gr", 0.0)
        if relief_app <= 0:
            continue

        role = _bullpen_role(snapshot, combined)
        consec = _projected_consecutive(current.get("appearance_dates", set()), current_date)
        fatigue_mult = consec_multipliers.get(consec, consec_multipliers[4])
        adjusted_quality = snapshot["quality"] * ROLE_QUALITY_MULTIPLIERS[role] * fatigue_mult

        grouped[role].append(adjusted_quality)
        fatigue_losses.append(1.0 - fatigue_mult)
        if consec >= 3:
            three_plus += 1

    return {
        "bp_core_strength": _weighted_top(grouped.get("core", []), weight_map["core"]),
        "bp_chase_strength": _weighted_top(grouped.get("chase", []), weight_map["chase"]),
        "bp_long_strength": _weighted_top(grouped.get("long", []), weight_map["long"]),
        "bp_fatigue_index": float(np.mean(fatigue_losses)) if fatigue_losses else 0.0,
        "bp_3plus_count": float(three_plus),
    }


def _build_bench_features(
    team_code: int,
    lineup_players: list[int],
    year: int,
    roster_players: list[int],
    batter_prior_cache: dict,
    batter_state: dict,
    stabilize_rates: bool = False,
) -> dict:
    lineup_set = set(lineup_players)
    scores = []
    for p_no in roster_players:
        if p_no in lineup_set:
            continue
        snapshot = _batter_snapshot(
            p_no, year, batter_prior_cache, batter_state, stabilize_rates=stabilize_rates
        )
        if snapshot["pa"] <= 0 and snapshot["wrcplus"] == LEAGUE_AVG_BATTER["wrcplus"]:
            continue
        scores.append(snapshot["wrcplus"])

    scores.sort(reverse=True)
    top = scores[: len(BENCH_WEIGHTS)]
    if top:
        bench_offense = float(
            sum(v * w for v, w in zip(top, BENCH_WEIGHTS[: len(top)]))
            / sum(BENCH_WEIGHTS[: len(top)])
        )
    else:
        bench_offense = 100.0

    bench_depth = float(np.mean(scores[:5])) if scores else 100.0
    return {
        "bench_offense_top3": bench_offense,
        "bench_depth": bench_depth,
    }


def _get_previous_roster_players(
    roster_db: dict[str, dict[int, list[int]]],
    team_code: int,
    game_date: date | None,
    lookback_days: int = 7,
) -> list[int]:
    if not game_date:
        return []
    for days_back in range(1, lookback_days + 1):
        prev_date = (game_date - timedelta(days=days_back)).isoformat()
        players = roster_db.get(prev_date, {}).get(team_code)
        if players:
            return list(players)
    return []


def _build_availability_features(
    lineup_players: list[int],
    current_roster: list[int],
    previous_roster: list[int],
    year: int,
    batter_prior_cache: dict,
    batter_state: dict,
    pitcher_prior_cache: dict,
    pitcher_state: dict,
    batter_player_ids: set[int],
    pitcher_player_ids: set[int],
    stabilize_rates: bool = False,
) -> dict:
    lineup_gap = 0.0
    if len(lineup_players) >= 7:
        active_batters = [p for p in current_roster if p in batter_player_ids]
        active_scores = sorted(
            [
                _batter_snapshot(
                    p_no, year, batter_prior_cache, batter_state, stabilize_rates=stabilize_rates
                )["wrcplus"]
                for p_no in active_batters
            ],
            reverse=True,
        )
        lineup_scores = [
            _batter_snapshot(
                p_no, year, batter_prior_cache, batter_state, stabilize_rates=stabilize_rates
            )["wrcplus"]
            for p_no in lineup_players
            if p_no in batter_player_ids
        ]
        if active_scores and lineup_scores:
            lineup_gap = max(0.0, float(np.mean(active_scores[:9]) - np.mean(lineup_scores)))

    removed = set(previous_roster) - set(current_roster)
    removed_hitter_scores = sorted(
        [
            _batter_snapshot(
                p_no, year, batter_prior_cache, batter_state, stabilize_rates=stabilize_rates
            )["wrcplus"]
            for p_no in removed
            if p_no in batter_player_ids
        ],
        reverse=True,
    )
    removed_pitcher_scores = sorted(
        [
            _pitcher_snapshot(
                p_no, year, pitcher_prior_cache, pitcher_state, stabilize_rates=stabilize_rates
            )["quality"]
            for p_no in removed
            if p_no in pitcher_player_ids
        ],
        reverse=True,
    )

    roster_loss_hit = max(0.0, _weighted_top(removed_hitter_scores, [0.6, 0.3, 0.1]) - 100.0) if removed_hitter_scores else 0.0
    roster_loss_pitch = max(0.0, _weighted_top(removed_pitcher_scores, [0.7, 0.3]) - 100.0) if removed_pitcher_scores else 0.0

    return {
        "lineup_gap": float(lineup_gap),
        "roster_loss_hit": float(roster_loss_hit),
        "roster_loss_pitch": float(roster_loss_pitch),
    }


def _pythagorean(rs: float, ra: float) -> float:
    if rs <= 0 and ra <= 0:
        return 0.5
    rs_e = rs ** PYTHAGOREAN_EXPONENT
    ra_e = ra ** PYTHAGOREAN_EXPONENT
    denom = rs_e + ra_e
    return rs_e / denom if denom > 0 else 0.5


def _compute_bp_load(team_code, game_date, team_game_dates) -> float:
    dates = team_game_dates.get(team_code, set())
    if not game_date:
        return 0.0
    count = 0
    for d in range(1, BULLPEN_LOAD_DAYS + 1):
        if (game_date - timedelta(days=d)) in dates:
            count += 1
    return float(count * 3)


def _get_lineup_players(s_no: int, team_code: int, lineup_db: dict) -> list[int]:
    game_lineups = lineup_db.get(s_no)
    if not game_lineups:
        return []
    players = game_lineups.get(team_code, [])
    order_map = {}
    for rec in players:
        p_no = rec.get("p_no")
        order_raw = rec.get("battingOrder", "0")
        if not p_no:
            continue
        try:
            order = int(order_raw)
        except (TypeError, ValueError):
            continue
        if 1 <= order <= 9:
            order_map[order] = int(p_no)
    return [order_map[i] for i in range(1, 10) if i in order_map]


def _update_batter_state(rec: dict, batter_state: dict):
    p_no = int(rec.get("p_no"))
    st = batter_state[p_no]
    st["pa"] += _safe_float(rec.get("PA"))
    st["epa"] += _safe_float(rec.get("ePA") or rec.get("PA"))
    st["ab"] += _safe_float(rec.get("AB"))
    st["h"] += _safe_float(rec.get("H"))
    st["double"] += _safe_float(rec.get("2B"))
    st["triple"] += _safe_float(rec.get("3B"))
    st["hr"] += _safe_float(rec.get("HR"))
    st["bb"] += _safe_float(rec.get("BB"))
    st["hp"] += _safe_float(rec.get("HP"))
    st["so"] += _safe_float(rec.get("SO"))
    st["sf"] += _safe_float(rec.get("SF"))
    st["tb"] += _safe_float(rec.get("TB"))


def _update_pitcher_state(rec: dict, pitcher_state: dict):
    p_no = int(rec.get("p_no"))
    st = pitcher_state[p_no]
    gs = _safe_float(rec.get("GS"))
    outs = float(_ip_to_outs(rec.get("IP")))

    st["g"] += _safe_float(rec.get("G", 1))
    st["gs"] += gs
    if gs <= 0:
        st["gr"] += 1.0
        st["relief_outs"] += outs
        if rec.get("_date"):
            st["appearance_dates"].add(rec["_date"])
    st["outs"] += outs
    st["gf"] += _safe_float(rec.get("GF"))
    st["s"] += _safe_float(rec.get("S"))
    st["hd"] += _safe_float(rec.get("HD"))
    st["er"] += _safe_float(rec.get("ER"))
    st["ab"] += _safe_float(rec.get("AB") or rec.get("TBF"))
    st["h"] += _safe_float(rec.get("H"))
    st["double"] += _safe_float(rec.get("2B"))
    st["triple"] += _safe_float(rec.get("3B"))
    st["hr"] += _safe_float(rec.get("HR"))
    st["bb"] += _safe_float(rec.get("BB"))
    st["hp"] += _safe_float(rec.get("HP"))
    st["so"] += _safe_float(rec.get("SO"))
    st["tb"] += _safe_float(rec.get("TB"))
    st["np"] += _safe_float(rec.get("NP"))


def build_dataset_v6(
    years: list[int] = None,
    output_name: str = "training_data_v6",
    include_bench: bool = True,
    include_sp_war: bool = False,
    stabilize_rates: bool = False,
    tune_bullpen: bool = False,
    use_spring: bool = False,
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
    batter_spring_db = _load_batter_spring_summary_db(years) if use_spring else {}
    pitcher_spring_db = _load_pitcher_spring_summary_db(years) if use_spring else {}
    batter_by_game = _load_day_index(years, "batting")
    pitcher_by_game = _load_day_index(years, "pitching")

    if not batter_by_game or not pitcher_by_game:
        raise RuntimeError(
            "v6는 batter_days_*.json / pitcher_days_*.json 이 필요합니다. "
            "먼저 pipeline/backfill_v6_support.py 로 playerDay 로그를 수집하세요."
        )

    batter_prior_cache = _build_batter_prior_cache(years, batter_summary_db)
    pitcher_prior_cache = _build_pitcher_prior_cache(years, pitcher_summary_db)
    batter_player_ids = {p_no for p_no, _ in batter_summary_db.keys()}
    pitcher_player_ids = {p_no for p_no, _ in pitcher_summary_db.keys()}

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
    feat_names = feature_names_v6(include_bench=include_bench, include_sp_war=include_sp_war)

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

        row = []

        home_batters = [
            _batter_snapshot(p_no, year, batter_prior_cache, batter_state, stabilize_rates=stabilize_rates)
            for p_no in home_lineup
        ]
        away_batters = [
            _batter_snapshot(p_no, year, batter_prior_cache, batter_state, stabilize_rates=stabilize_rates)
            for p_no in away_lineup
        ]
        if use_spring:
            home_batters = [
                _maybe_blend_spring_batter_snapshot(stats, batter_spring_db, p_no, year, game_date)
                for stats, p_no in zip(home_batters, home_lineup)
            ]
            away_batters = [
                _maybe_blend_spring_batter_snapshot(stats, batter_spring_db, p_no, year, game_date)
                for stats, p_no in zip(away_batters, away_lineup)
            ]

        while len(home_batters) < 9:
            home_batters.append(dict(LEAGUE_AVG_BATTER))
        while len(away_batters) < 9:
            away_batters.append(dict(LEAGUE_AVG_BATTER))

        batter_stats_order = ["wrcplus", "avg", "obp", "slg", "hr_rate", "bb_rate", "k_rate", "bbk_ratio", "pa"]
        for batters in [home_batters, away_batters]:
            for b in batters:
                for stat in batter_stats_order:
                    row.append(float(b.get(stat, LEAGUE_AVG_BATTER[stat])))

        home_sp_stats = _pitcher_snapshot(
            home_sp_no, year, pitcher_prior_cache, pitcher_state, stabilize_rates=stabilize_rates
        )
        away_sp_stats = _pitcher_snapshot(
            away_sp_no, year, pitcher_prior_cache, pitcher_state, stabilize_rates=stabilize_rates
        )
        if use_spring:
            home_sp_stats = _maybe_blend_spring_pitcher_snapshot(
                home_sp_stats, pitcher_spring_db, home_sp_no, year, game_date
            )
            away_sp_stats = _maybe_blend_spring_pitcher_snapshot(
                away_sp_stats, pitcher_spring_db, away_sp_no, year, game_date
            )
        sp_stats_order = ["era", "fip", "whip", "k9", "bb9", "hr9", "obp_against", "kbb_ratio", "np_per_app", "ip"]
        if include_sp_war:
            sp_stats_order.append("war")
        for sp in [home_sp_stats, away_sp_stats]:
            for stat in sp_stats_order:
                row.append(float(sp.get(stat, LEAGUE_AVG_SP[stat])))

        if home_sp_no:
            elo.update_sp_rating(home_sp_no, home, home_sp_stats["fip"])
        if away_sp_no:
            elo.update_sp_rating(away_sp_no, away, away_sp_stats["fip"])
        elo.update_team_sp_avg(home, 4.20)
        elo.update_team_sp_avg(away, 4.20)

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
        home_bp_load = _compute_bp_load(home, game_date, team_game_dates)
        away_bp_load = _compute_bp_load(away, game_date, team_game_dates)
        home_rest = float((game_date - team_last_game_date[home]).days) if game_date and home in team_last_game_date else 3.0
        away_rest = float((game_date - team_last_game_date[away]).days) if game_date and away in team_last_game_date else 3.0
        row.extend([
            home_elo_val, home_pyth, home_wpct, home_bp_load, home_rest,
            away_elo_val, away_pyth, away_wpct, away_bp_load, away_rest,
        ])

        home_today_wrc = float(np.mean([b["wrcplus"] for b in home_batters]))
        away_today_wrc = float(np.mean([b["wrcplus"] for b in away_batters]))
        home_rs_hist = team_rs_history.get(home, [])
        away_rs_hist = team_rs_history.get(away, [])
        home_trend = float(np.mean(home_rs_hist[-5:]) - team_runs_scored[home] / team_games[home]) if len(home_rs_hist) >= 5 and team_games[home] > 0 else 0.0
        away_trend = float(np.mean(away_rs_hist[-5:]) - team_runs_scored[away] / team_games[away]) if len(away_rs_hist) >= 5 and team_games[away] > 0 else 0.0
        home_wrc_hist = team_lineup_wrc_history.get(home, [])
        away_wrc_hist = team_lineup_wrc_history.get(away, [])
        home_wrc_delta = float(home_today_wrc - np.mean(home_wrc_hist)) if home_wrc_hist else 0.0
        away_wrc_delta = float(away_today_wrc - np.mean(away_wrc_hist)) if away_wrc_hist else 0.0
        row.extend([home_trend, home_wrc_delta, away_trend, away_wrc_delta])

        temp = game.get("temperature") or 15.0
        temp = max(-10.0, min(40.0, float(temp)))
        humidity = max(0.0, min(100.0, float(game.get("humidity") or 50.0)))
        wind_speed = max(0.0, min(30.0, float(game.get("windSpeed") or 0.0)))
        is_night = 1.0 if game_hour >= NIGHT_GAME_HOUR else 0.0
        row.extend([temp, humidity, wind_speed, is_night, 1.0])

        home_bp = _build_bullpen_features(
            home,
            home_sp_no,
            year,
            home_roster,
            game_date,
            pitcher_prior_cache,
            pitcher_state,
            stabilize_rates=stabilize_rates,
            tuned=tune_bullpen,
        )
        away_bp = _build_bullpen_features(
            away,
            away_sp_no,
            year,
            away_roster,
            game_date,
            pitcher_prior_cache,
            pitcher_state,
            stabilize_rates=stabilize_rates,
            tuned=tune_bullpen,
        )
        for bp in [home_bp, away_bp]:
            row.extend([
                bp["bp_core_strength"],
                bp["bp_chase_strength"],
                bp["bp_long_strength"],
                bp["bp_fatigue_index"],
                bp["bp_3plus_count"],
            ])

        home_prev_roster = _get_previous_roster_players(roster_db, home, game_date)
        away_prev_roster = _get_previous_roster_players(roster_db, away, game_date)
        home_availability = _build_availability_features(
            home_lineup,
            home_roster,
            home_prev_roster,
            year,
            batter_prior_cache,
            batter_state,
            pitcher_prior_cache,
            pitcher_state,
            batter_player_ids,
            pitcher_player_ids,
            stabilize_rates=stabilize_rates,
        )
        away_availability = _build_availability_features(
            away_lineup,
            away_roster,
            away_prev_roster,
            year,
            batter_prior_cache,
            batter_state,
            pitcher_prior_cache,
            pitcher_state,
            batter_player_ids,
            pitcher_player_ids,
            stabilize_rates=stabilize_rates,
        )
        for availability in [home_availability, away_availability]:
            row.extend([
                availability["lineup_gap"],
                availability["roster_loss_hit"],
                availability["roster_loss_pitch"],
            ])

        if include_bench:
            home_bench = _build_bench_features(
                home, home_lineup, year, home_roster, batter_prior_cache, batter_state,
                stabilize_rates=stabilize_rates,
            )
            away_bench = _build_bench_features(
                away, away_lineup, year, away_roster, batter_prior_cache, batter_state,
                stabilize_rates=stabilize_rates,
            )
            for bench in [home_bench, away_bench]:
                row.extend([bench["bench_offense_top3"], bench["bench_depth"]])

        assert len(row) == len(feat_names), f"expected {len(feat_names)}, got {len(row)}"

        if home_score > away_score:
            label = 1
        elif home_score < away_score:
            label = 0
        else:
            label = None

        if label is not None:
            rows.append(row + [label])

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

    columns = feat_names + ["label"]
    df = pd.DataFrame(rows, columns=columns)
    logger.info("v6 데이터셋 생성 완료: %d행 × %d열", len(df), len(df.columns))
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

    parser = argparse.ArgumentParser(description="v6 데이터셋 구축")
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    parser.add_argument("--output", type=str, default="training_data_v6")
    parser.add_argument("--variant", type=str, default="v6", choices=sorted(VARIANT_CONFIGS.keys()))
    args = parser.parse_args()

    cfg = VARIANT_CONFIGS[args.variant]
    build_dataset_v6(
        years=args.years,
        output_name=args.output,
        include_bench=cfg["include_bench"],
        include_sp_war=cfg["include_sp_war"],
        stabilize_rates=cfg["stabilize_rates"],
        tune_bullpen=cfg["tune_bullpen"],
        use_spring=cfg.get("use_spring", False),
    )
