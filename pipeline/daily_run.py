"""
일일 자동 파이프라인 (v5 — 221개 피처).

매일 실행 흐름:
1. 당일 경기일정 조회
2. 선발투수 + 라인업 확인
3. 221개 피처 산출 (v2 159 + situational 15 + expanded 47)
4. Elo + XGBoost 블렌딩 예측
5. API 제출
6. 전일 결과 반영 (Elo 갱신)
"""
import sys
import json
import math
import logging
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.constants import (
    TEAM_CODES, NIGHT_GAME_HOUR, LEAGUE_REGULAR,
    PYTHAGOREAN_EXPONENT, STADIUM_COORDS, BULLPEN_LOAD_DAYS,
    LEAGUE_EXHIBITION,
    SPRING_FULL_WEIGHT_DAYS,
    SPRING_HALF_WEIGHT_DAYS,
    SPRING_BATTER_PA_EQUIV,
    SPRING_PITCHER_IP_EQUIV,
    SPRING_BATTER_ANCHOR,
    SPRING_PITCHER_ANCHOR,
)
from data.collector import (
    get_game_schedule,
    get_game_lineup,
    get_player_season,
    get_player_roster,
)
from elo.engine import EloEngine
from features.expanded import (
    extract_batter_stats,
    extract_batter_stats_v5,
    extract_sp_stats,
    extract_sp_stats_v5,
    compute_platoon_advantage,
    encode_weather,
    encode_wind_direction,
    LEAGUE_AVG_BATTER,
    LEAGUE_AVG_BATTER_V5,
)
from models.predict import predict_game, batch_predict
from models.train import load_model
from pipeline.submit import submit_batch

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)
_player_season_cache: dict[tuple[int, str, int | None, int], dict] = {}
_season_start_cache: dict[int, date] = {}

# ── 시즌 컨텍스트 상태 (세션 동안 유지) ──
_season_state = {
    "team_runs_history": defaultdict(list),
    "team_runs_scored": defaultdict(float),
    "team_games": defaultdict(int),
    "team_lineup_wrc_history": defaultdict(list),
    # v5 추가
    "team_results": defaultdict(list),       # [True/False, ...]
    "team_prev_game": {},                    # {team: (date, hour)}
    "team_game_dates": defaultdict(set),     # {team: {date, ...}}
    "team_prev_s_code": {},                  # {team: s_code}
    "team_runs_allowed": defaultdict(float),  # 피실점 누적
    "h2h_tracker": defaultdict(lambda: defaultdict(int)),
    "sp_vs_team_tracker": defaultdict(lambda: {"wins": 0, "total": 0}),
}


def _parse_game_clock(hm_value) -> tuple[int, int]:
    """hm 값을 (hour, minute)로 파싱."""
    hm = str(hm_value or "18:00:00")
    try:
        if ":" in hm:
            parts = hm.split(":")
            hour = int(parts[0])
            minute = int(parts[1]) if len(parts) > 1 else 0
        else:
            digits = hm.strip()
            if len(digits) >= 4:
                hour = int(digits[:2])
                minute = int(digits[2:4])
            else:
                hour = int(digits[:2])
                minute = 0
    except (ValueError, IndexError, TypeError):
        hour, minute = 18, 0
    hour = max(0, min(23, hour))
    minute = max(0, min(59, minute))
    return hour, minute


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 좌표 간 거리(km)."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _get_stadium_distance(s_code_from: int, s_code_to: int) -> float:
    c1 = STADIUM_COORDS.get(s_code_from)
    c2 = STADIUM_COORDS.get(s_code_to)
    if not c1 or not c2:
        return 0.0
    return _haversine(c1[0], c1[1], c2[0], c2[1])


def _load_park_factors() -> dict:
    """사전 계산된 Park Factor 로드 (3년 박스스코어 기반)."""
    pf = {}
    for year in [2023, 2024, 2025]:
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
                pf.setdefault(s_code, []).append(hs + aws)

    all_runs = []
    for runs in pf.values():
        all_runs.extend(runs)
    league_avg = np.mean(all_runs) if all_runs else 9.5

    result = {}
    for s_code, runs in pf.items():
        if len(runs) >= 10:
            result[s_code] = np.mean(runs) / league_avg
    return result


def setup_logging():
    """로깅 설정."""
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"daily_{datetime.now():%Y%m%d}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def _parse_schedule_response(resp: dict) -> list[dict]:
    """API 스케줄 응답에서 경기 리스트 추출."""
    games = []
    for key, value in resp.items():
        if isinstance(value, list):
            games.extend(value)
    return games


def _find_year_record(records: list[dict], year: int):
    """리스트에서 해당 연도 레코드 찾기."""
    for rec in records:
        rec_year = rec.get("year")
        if rec_year is not None and str(rec_year) == str(year):
            return rec
    return None


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_regular_season_start(year: int) -> date:
    cached = _season_start_cache.get(year)
    if cached is not None:
        return cached

    sched_file = DATA_DIR / f"schedules_{year}.json"
    if sched_file.exists():
        rows = json.loads(sched_file.read_text())
        regular_rows = [r for r in rows if r.get("leagueType") == LEAGUE_REGULAR]
        timestamps = [r.get("gameDate") for r in regular_rows if r.get("gameDate")]
        if timestamps:
            season_start = datetime.fromtimestamp(min(timestamps)).date()
            _season_start_cache[year] = season_start
            return season_start

    season_start = date(year, 3, 28)
    _season_start_cache[year] = season_start
    return season_start


def _get_spring_decay(year: int, target_date: date | None) -> float:
    if target_date is None:
        return 0.0
    season_start = _get_regular_season_start(year)
    delta_days = (target_date - season_start).days
    if delta_days < 0:
        return 0.0
    if delta_days < SPRING_FULL_WEIGHT_DAYS:
        return 1.0
    if delta_days < SPRING_HALF_WEIGHT_DAYS:
        return 0.5
    return 0.0


def _get_player_season_cached(
    p_no: int,
    m2: str,
    year: int | None,
    league_type: int,
) -> dict:
    key = (int(p_no), m2, year, league_type)
    cached = _player_season_cache.get(key)
    if cached is not None:
        return cached
    resp = get_player_season(p_no, m2=m2, year=year, league_type=league_type)
    _player_season_cache[key] = resp
    return resp


def _blend_stat_dicts(base_stats: dict, spring_stats: dict, spring_weight: float) -> dict:
    if spring_weight <= 0.0 or not spring_stats:
        return dict(base_stats)
    blended = dict(base_stats)
    for key, base_val in base_stats.items():
        if key not in spring_stats:
            continue
        blended[key] = (1.0 - spring_weight) * float(base_val) + spring_weight * float(spring_stats[key])
    return blended


def _extract_batter_pa(basic_rec: dict | None, deepen_rec: dict | None) -> float:
    if basic_rec:
        for key in ("PA", "TBF", "ab", "AB"):
            if key in basic_rec:
                val = _safe_float(basic_rec.get(key), 0.0)
                if val > 0:
                    return val
    if deepen_rec:
        val = _safe_float(deepen_rec.get("PA"), 0.0)
        if val > 0:
            return val
    return 0.0


def _extract_pitcher_ip(basic_rec: dict | None, deepen_rec: dict | None) -> float:
    if basic_rec:
        val = _safe_float(basic_rec.get("IP"), 0.0)
        if val > 0:
            return val
    if deepen_rec:
        val = _safe_float(deepen_rec.get("IP"), 0.0)
        if val > 0:
            return val
    return 0.0


def _maybe_blend_spring_batter(
    base_stats: dict,
    spring_basic: dict | None,
    spring_deepen: dict | None,
    year: int,
    target_date: date | None,
) -> dict:
    decay = _get_spring_decay(year, target_date)
    if decay <= 0.0 or not (spring_basic or spring_deepen):
        return dict(base_stats)
    spring_pa = _extract_batter_pa(spring_basic, spring_deepen)
    effective_pa = spring_pa * SPRING_BATTER_PA_EQUIV
    if effective_pa <= 0.0:
        return dict(base_stats)
    spring_stats = extract_batter_stats_v5(spring_basic, spring_deepen)
    spring_weight = decay * (effective_pa / (effective_pa + SPRING_BATTER_ANCHOR))
    return _blend_stat_dicts(base_stats, spring_stats, spring_weight)


def _maybe_blend_spring_pitcher(
    base_stats: dict,
    spring_basic: dict | None,
    spring_deepen: dict | None,
    year: int,
    target_date: date | None,
) -> dict:
    decay = _get_spring_decay(year, target_date)
    if decay <= 0.0 or not (spring_basic or spring_deepen):
        return dict(base_stats)
    spring_ip = _extract_pitcher_ip(spring_basic, spring_deepen)
    effective_ip = spring_ip * SPRING_PITCHER_IP_EQUIV
    if effective_ip <= 0.0:
        return dict(base_stats)
    spring_stats = extract_sp_stats_v5(spring_basic, spring_deepen)
    if spring_basic:
        spring_stats["whip"] = _safe_float(spring_basic.get("WHIP"), 1.40)
        ip = _safe_float(spring_basic.get("IP"), 0.0)
        spring_stats["hr9"] = _safe_float(spring_basic.get("HR"), 0.0) / ip * 9 if ip > 0 else 1.0
    else:
        spring_stats["whip"] = 1.40
        spring_stats["hr9"] = 1.0
    spring_weight = decay * (effective_ip / (effective_ip + SPRING_PITCHER_ANCHOR))
    return _blend_stat_dicts(base_stats, spring_stats, spring_weight)


# ── 경기 조회 ──

def get_today_games(year: int, month: int, day: int) -> list[dict]:
    """당일 경기 목록 조회."""
    resp = get_game_schedule(year=year, month=month, day=day)
    games_raw = _parse_schedule_response(resp)

    if not games_raw:
        logger.warning("당일 경기 없음: %d-%02d-%02d", year, month, day)
        return []

    games = []
    for g in games_raw:
        if g.get("leagueType") != LEAGUE_REGULAR:
            continue
        game_hour, game_minute = _parse_game_clock(g.get("hm"))
        game_dt = datetime(year, month, day, game_hour, game_minute)

        games.append({
            "s_no": g["s_no"],
            "home_team": g["homeTeam"],
            "away_team": g["awayTeam"],
            "home_score": g.get("homeScore"),
            "away_score": g.get("awayScore"),
            "state": g.get("state"),
            "home_sp": g.get("homeSP"),
            "away_sp": g.get("awaySP"),
            "home_sp_name": g.get("homeSPName", ""),
            "away_sp_name": g.get("awaySPName", ""),
            "temperature": g.get("temperature") or 15.0,
            "humidity": g.get("humidity"),
            "windSpeed": g.get("windSpeed"),
            "weather": g.get("weather"),
            "windDirection": g.get("windDirection"),
            "game_hour": game_hour,
            "game_minute": game_minute,
            "hm": g.get("hm"),
            "s_code": g.get("s_code"),
            "_datetime": game_dt,
        })

    logger.info("당일 경기 %d개 조회", len(games))
    return games


# ── 선수 스탯 조회 ──

LEAGUE_AVG_SP_V2 = {
    "era": 4.50, "fip": 4.50, "whip": 1.40, "k9": 7.0,
    "bb9": 3.5, "hr9": 1.0, "war": 1.0,
}


def get_sp_full_stats(
    p_no: int,
    year: int,
    team_code: int | None = None,
    target_date: date | None = None,
) -> dict:
    """선발투수 시즌 전체 스탯 조회 → 9개 stats (v5 호환: 7개 + obp_against, slg_against)."""
    if not p_no:
        return {**LEAGUE_AVG_SP_V2, "obp_against": 0.330, "slg_against": 0.400}

    try:
        resp = _get_player_season_cached(p_no, m2="pitching", year=year, league_type=LEAGUE_REGULAR)
        basic_list = resp.get("basic", {}).get("list", [])
        deepen_list = resp.get("deepen", {}).get("list", [])
        basic_rec = _find_year_record(basic_list, year)
        deepen_rec = _find_year_record(deepen_list, year)

        # 이전 시즌 폴백 (시즌 초 데이터 없을 때)
        if not basic_rec and basic_list:
            basic_rec = basic_list[-1]
        if not deepen_rec and deepen_list:
            deepen_rec = deepen_list[-1]

        stats = extract_sp_stats_v5(basic_rec, deepen_rec)
        # v2 추가: whip, hr9
        if basic_rec:
            stats["whip"] = float(basic_rec.get("WHIP") or 1.40)
            ip = float(basic_rec.get("IP") or 1)
            stats["hr9"] = float(basic_rec.get("HR") or 0) / ip * 9 if ip > 0 else 1.0
        else:
            stats["whip"] = 1.40
            stats["hr9"] = 1.0
        if team_code is not None:
            spring_resp = _get_player_season_cached(
                p_no, m2="pitching", year=year, league_type=LEAGUE_EXHIBITION
            )
            spring_basic = _find_year_record(spring_resp.get("basic", {}).get("list", []), year)
            spring_deepen = _find_year_record(spring_resp.get("deepen", {}).get("list", []), year)
            stats = _maybe_blend_spring_pitcher(stats, spring_basic, spring_deepen, year, target_date)
        return stats

    except Exception as e:
        logger.warning("투수 %d 스탯 조회 실패: %s", p_no, e)
        return {**LEAGUE_AVG_SP_V2, "obp_against": 0.330, "slg_against": 0.400}


def get_lineup_individual(
    s_no: int,
    home_team: int,
    away_team: int,
    year: int,
    target_date: date | None = None,
) -> tuple[list[dict], list[dict], list[int], list[int], int, int]:
    """경기 라인업에서 홈/원정 각 9명 개인 타자 스탯(v5) + p_bat + SP p_throw 반환.

    Returns:
        (home_batters, away_batters, home_hands, away_hands,
         away_sp_throw, home_sp_throw)
    """
    default_lineup = [dict(LEAGUE_AVG_BATTER_V5) for _ in range(9)]
    default_hands = [2] * 9
    try:
        resp = get_game_lineup(s_no)
    except Exception as e:
        logger.warning("라인업 조회 실패 (s_no=%d): %s", s_no, e)
        return list(default_lineup), list(default_lineup), default_hands, default_hands, 1, 1

    def _get_sp_throw(team_code: int) -> int:
        """팀 라인업에서 SP의 p_throw 찾기."""
        players = resp.get(str(team_code), [])
        if not isinstance(players, list):
            return 1
        for p in players:
            if str(p.get("battingOrder", "")) == "P":
                return p.get("p_throw", 1)
        return 1

    def _process_team(team_code: int) -> tuple[list[dict], list[int]]:
        players = resp.get(str(team_code), [])
        if not isinstance(players, list):
            return list(default_lineup), list(default_hands)

        order_map = {}
        hand_map = {}
        for p in players:
            p_no = p.get("p_no")
            order_raw = p.get("battingOrder", "0")
            if not p_no:
                continue
            try:
                order = int(order_raw)
            except (ValueError, TypeError):
                continue
            if not (1 <= order <= 9):
                continue

            hand_map[order] = p.get("p_bat", 2)
            try:
                ps = _get_player_season_cached(int(p_no), m2="batting", year=year, league_type=LEAGUE_REGULAR)
                basic_list = ps.get("basic", {}).get("list", [])
                deepen_list = ps.get("deepen", {}).get("list", [])
                basic_rec = _find_year_record(basic_list, year)
                deepen_rec = _find_year_record(deepen_list, year)
                if not basic_rec and basic_list:
                    basic_rec = basic_list[-1]
                if not deepen_rec and deepen_list:
                    deepen_rec = deepen_list[-1]
                batter_stats = extract_batter_stats_v5(basic_rec, deepen_rec)
                spring_ps = _get_player_season_cached(int(p_no), m2="batting", year=year, league_type=LEAGUE_EXHIBITION)
                spring_basic = _find_year_record(spring_ps.get("basic", {}).get("list", []), year)
                spring_deepen = _find_year_record(spring_ps.get("deepen", {}).get("list", []), year)
                order_map[order] = _maybe_blend_spring_batter(
                    batter_stats, spring_basic, spring_deepen, year, target_date
                )
            except Exception:
                order_map[order] = dict(LEAGUE_AVG_BATTER_V5)

        batters = [order_map.get(i, dict(LEAGUE_AVG_BATTER_V5)) for i in range(1, 10)]
        hands = [hand_map.get(i, 2) for i in range(1, 10)]
        return batters, hands

    away_sp_throw = _get_sp_throw(away_team)
    home_sp_throw = _get_sp_throw(home_team)
    home_batters, home_hands = _process_team(home_team)
    away_batters, away_hands = _process_team(away_team)
    logger.info("라인업 개인 스탯 조회 완료 (v5)")
    return home_batters, away_batters, home_hands, away_hands, away_sp_throw, home_sp_throw


# ── 컨텍스트 조회 ──

def get_rest_days(
    team_code: int, target_date: datetime, year: int, month: int,
) -> float:
    """팀의 휴식일 계산."""
    try:
        resp = get_game_schedule(year=year, month=month)
        games = _parse_schedule_response(resp)
        games = [g for g in games if g.get("leagueType") == LEAGUE_REGULAR]

        team_dates = []
        for g in games:
            gd = g.get("gameDate", 0)
            if gd <= 0:
                continue
            game_date = datetime.fromtimestamp(gd).date()
            if game_date < target_date.date():
                if g.get("homeTeam") == team_code or g.get("awayTeam") == team_code:
                    team_dates.append(game_date)

        if not team_dates:
            return 3.0

        team_dates.sort()
        return float((target_date.date() - team_dates[-1]).days)

    except Exception as e:
        logger.warning("휴식일 조회 실패 (team=%d): %s", team_code, e)
        return 1.0


def _pythagorean(rs: float, ra: float) -> float:
    if rs <= 0 and ra <= 0:
        return 0.5
    rs_e = rs ** PYTHAGOREAN_EXPONENT
    ra_e = ra ** PYTHAGOREAN_EXPONENT
    denom = rs_e + ra_e
    return rs_e / denom if denom > 0 else 0.5


# ── 피처 빌드 ──

def _compute_live_context(team_code: int, today_lineup_wrc: float) -> dict:
    """라이브 예측용 시퀀스 컨텍스트 계산."""
    st = _season_state
    rs_hist = st["team_runs_history"].get(team_code, [])
    total_games = st["team_games"].get(team_code, 0)

    # scoring_trend
    if len(rs_hist) >= 5 and total_games > 0:
        rs_5 = float(np.mean(rs_hist[-5:]))
        season_avg = st["team_runs_scored"].get(team_code, 0) / total_games
        scoring_trend = rs_5 - season_avg
    else:
        scoring_trend = 0.0

    # lineup_wrc_delta
    history = st["team_lineup_wrc_history"].get(team_code, [])
    if history:
        lineup_wrc_delta = today_lineup_wrc - float(np.mean(history))
    else:
        lineup_wrc_delta = 0.0

    return {
        "scoring_trend": scoring_trend,
        "lineup_wrc_delta": lineup_wrc_delta,
    }


def build_game_features_v5(
    game: dict,
    elo_engine: EloEngine,
    year: int,
    park_factors: dict = None,
    use_confirmed_lineup: bool = True,
) -> dict:
    """단일 경기의 221개 피처 산출 (v5 확장)."""

    st = _season_state
    home = game["home_team"]
    away = game["away_team"]
    target_dt = game.get("_datetime", datetime(year, 3, 28))
    game_date = target_dt.date() if target_dt else None
    game_hour = game.get("game_hour", 18)
    s_code = game.get("s_code")

    # 1. 선발투수 스탯 (9개 × 2)
    home_sp_stats = get_sp_full_stats(game.get("home_sp"), year, team_code=home, target_date=game_date)
    away_sp_stats = get_sp_full_stats(game.get("away_sp"), year, team_code=away, target_date=game_date)

    # SP Elo 보정
    if game.get("home_sp"):
        elo_engine.update_sp_rating(game["home_sp"], home, home_sp_stats["fip"])
    if game.get("away_sp"):
        elo_engine.update_sp_rating(game["away_sp"], away, away_sp_stats["fip"])
    elo_engine.update_team_sp_avg(home, 4.20)
    elo_engine.update_team_sp_avg(away, 4.20)

    # 2. 라인업 개인 타자 스탯 (v5: 9개 × 9명 × 2팀) + p_bat + SP p_throw
    if use_confirmed_lineup:
        home_batters, away_batters, home_hands, away_hands, away_sp_throw, home_sp_throw = \
            get_lineup_individual(game["s_no"], home, away, year, target_date=game_date)
    else:
        home_batters = [dict(LEAGUE_AVG_BATTER_V5) for _ in range(9)]
        away_batters = [dict(LEAGUE_AVG_BATTER_V5) for _ in range(9)]
        home_hands = [2] * 9
        away_hands = [2] * 9
        away_sp_throw = 1
        home_sp_throw = 1

    # 3. 팀 컨텍스트 (5개 × 2 = 10)
    home_elo = elo_engine.get_rating(home)
    away_elo = elo_engine.get_rating(away)

    h_games = max(st["team_games"].get(home, 0), 1)
    a_games = max(st["team_games"].get(away, 0), 1)
    home_rs_avg = st["team_runs_scored"].get(home, 0) / h_games
    home_ra = st.get("team_runs_allowed", {}).get(home, 0) / h_games
    home_pyth = _pythagorean(home_rs_avg * 9, home_ra * 9) if h_games > 1 else 0.5

    away_rs_avg = st["team_runs_scored"].get(away, 0) / a_games
    away_ra = st.get("team_runs_allowed", {}).get(away, 0) / a_games
    away_pyth = _pythagorean(away_rs_avg * 9, away_ra * 9) if a_games > 1 else 0.5

    home_results = st["team_results"].get(home, [])
    away_results = st["team_results"].get(away, [])
    home_wpct = sum(home_results) / len(home_results) if home_results else 0.5
    away_wpct = sum(away_results) / len(away_results) if away_results else 0.5

    home_rest = get_rest_days(home, target_dt, year, target_dt.month)
    away_rest = get_rest_days(away, target_dt, year, target_dt.month)

    # bp_load
    home_dates = st["team_game_dates"].get(home, set())
    away_dates = st["team_game_dates"].get(away, set())
    home_bp_load = 0.0
    away_bp_load = 0.0
    if game_date:
        for d in range(1, BULLPEN_LOAD_DAYS + 1):
            if (game_date - timedelta(days=d)) in home_dates:
                home_bp_load += 3.0
            if (game_date - timedelta(days=d)) in away_dates:
                away_bp_load += 3.0

    # 4. 시퀀스 (2개 × 2 = 4)
    home_today_wrc = float(np.mean([b["wrcplus"] for b in home_batters]))
    away_today_wrc = float(np.mean([b["wrcplus"] for b in away_batters]))
    home_ctx = _compute_live_context(home, home_today_wrc)
    away_ctx = _compute_live_context(away, away_today_wrc)

    # 5. 환경 (5개)
    temp = game.get("temperature", 15.0)
    if temp == 0 or temp > 45 or temp < -15:
        temp = 15.0
    temp = max(-10.0, min(40.0, float(temp)))
    is_night = 1.0 if game_hour >= NIGHT_GAME_HOUR else 0.0
    pf = 1.0
    if park_factors and s_code:
        pf = park_factors.get(s_code, 1.0)

    humidity = float(game.get("humidity") or 50.0)
    humidity = max(0.0, min(100.0, humidity))

    wind_speed = float(game.get("wind_speed") or game.get("windSpeed") or 0.0)
    wind_speed = max(0.0, min(30.0, wind_speed))

    # ── v2 159 피처 벡터 조립 ──
    row = []

    batter_stats_order = ["wrcplus", "avg", "obp", "slg", "hr_rate", "bb_rate", "k_rate"]
    for batters in [home_batters, away_batters]:
        for b in batters:
            for stat in batter_stats_order:
                row.append(float(b.get(stat, LEAGUE_AVG_BATTER.get(stat, 0.0))))

    sp_stats_order = ["era", "fip", "whip", "k9", "bb9", "hr9", "war"]
    for sp in [home_sp_stats, away_sp_stats]:
        for stat in sp_stats_order:
            row.append(float(sp.get(stat, LEAGUE_AVG_SP_V2.get(stat, 0.0))))

    row.extend([
        home_elo, home_pyth, home_wpct, home_bp_load, home_rest,
        away_elo, away_pyth, away_wpct, away_bp_load, away_rest,
    ])

    row.extend([
        home_ctx["scoring_trend"], home_ctx["lineup_wrc_delta"],
        away_ctx["scoring_trend"], away_ctx["lineup_wrc_delta"],
    ])

    row.extend([temp, humidity, wind_speed, is_night, pf])

    # ── situational 15 피처 ──

    # H2H (3)
    h2h_key = frozenset({home, away})
    h2h_rec = st["h2h_tracker"].get(h2h_key)
    if h2h_rec and h2h_rec.get("total", 0) > 0:
        h2h_home = h2h_rec.get(home, 0) / h2h_rec["total"]
        h2h_away = h2h_rec.get(away, 0) / h2h_rec["total"]
        h2h_n = float(h2h_rec["total"])
    else:
        h2h_home, h2h_away, h2h_n = 0.5, 0.5, 0.0
    row.extend([h2h_home, h2h_away, h2h_n])

    # DAGN (2)
    dagn_home = 0.0
    dagn_away = 0.0
    if game_hour < NIGHT_GAME_HOUR:
        prev_h = st["team_prev_game"].get(home)
        if prev_h and game_date:
            if (game_date - prev_h[0]).days == 1 and prev_h[1] >= NIGHT_GAME_HOUR:
                dagn_home = 1.0
        prev_a = st["team_prev_game"].get(away)
        if prev_a and game_date:
            if (game_date - prev_a[0]).days == 1 and prev_a[1] >= NIGHT_GAME_HOUR:
                dagn_away = 1.0
    row.extend([dagn_home, dagn_away])

    # 이동거리 (2)
    travel_home = _get_stadium_distance(st["team_prev_s_code"].get(home, s_code), s_code) if s_code else 0.0
    travel_away = _get_stadium_distance(st["team_prev_s_code"].get(away, s_code), s_code) if s_code else 0.0
    row.extend([travel_home, travel_away])

    # 불펜 연투 (2)
    bp_consec_home = 0.0
    bp_consec_away = 0.0
    if game_date:
        check = game_date - timedelta(days=1)
        for _ in range(5):
            if check in home_dates:
                bp_consec_home += 1.0
                check -= timedelta(days=1)
            else:
                break
        bp_consec_home = min(bp_consec_home, 3.0)
        check = game_date - timedelta(days=1)
        for _ in range(5):
            if check in away_dates:
                bp_consec_away += 1.0
                check -= timedelta(days=1)
            else:
                break
        bp_consec_away = min(bp_consec_away, 3.0)
    row.extend([bp_consec_home, bp_consec_away])

    # 최근 5경기 폼 (4)
    r5_h = home_results[-5:] if len(home_results) >= 5 else home_results
    r5_a = away_results[-5:] if len(away_results) >= 5 else away_results
    r5_wpct_home = sum(r5_h) / len(r5_h) if r5_h else 0.5
    r5_wpct_away = sum(r5_a) / len(r5_a) if r5_a else 0.5
    rs_hist_h = st["team_runs_history"].get(home, [])
    rs_hist_a = st["team_runs_history"].get(away, [])
    r5_rs_home = float(np.mean(rs_hist_h[-5:])) if len(rs_hist_h) >= 5 else (float(np.mean(rs_hist_h)) if rs_hist_h else 4.5)
    r5_rs_away = float(np.mean(rs_hist_a[-5:])) if len(rs_hist_a) >= 5 else (float(np.mean(rs_hist_a)) if rs_hist_a else 4.5)
    row.extend([r5_wpct_home, r5_wpct_away, r5_rs_home, r5_rs_away])

    # SP vs 상대팀 (2)
    sp_home_key = (game.get("home_sp"), away)
    sp_home_rec = st["sp_vs_team_tracker"].get(sp_home_key)
    sp_vs_home = sp_home_rec["wins"] / sp_home_rec["total"] if sp_home_rec and sp_home_rec["total"] > 0 else 0.5
    sp_away_key = (game.get("away_sp"), home)
    sp_away_rec = st["sp_vs_team_tracker"].get(sp_away_key)
    sp_vs_away = sp_away_rec["wins"] / sp_away_rec["total"] if sp_away_rec and sp_away_rec["total"] > 0 else 0.5
    row.extend([sp_vs_home, sp_vs_away])

    # ── expanded 47 피처 (v5 신규) ──

    # platoon_adv (2)
    platoon_home = compute_platoon_advantage(home_hands, away_sp_throw)
    platoon_away = compute_platoon_advantage(away_hands, home_sp_throw)
    row.extend([platoon_home, platoon_away])

    # weather one-hot (3)
    row.extend(encode_weather(game.get("weather")))

    # wind direction sin/cos (2)
    row.extend(encode_wind_direction(game.get("windDirection")))

    # batter wOBA (18)
    for batters in [home_batters, away_batters]:
        for b in batters:
            row.append(float(b.get("woba", 0.320)))

    # batter BABIP (18)
    for batters in [home_batters, away_batters]:
        for b in batters:
            row.append(float(b.get("babip", 0.300)))

    # pitcher OBP/SLG against (4)
    row.append(float(home_sp_stats.get("obp_against", 0.330)))
    row.append(float(home_sp_stats.get("slg_against", 0.400)))
    row.append(float(away_sp_stats.get("obp_against", 0.330)))
    row.append(float(away_sp_stats.get("slg_against", 0.400)))

    assert len(row) == 221, f"Expected 221, got {len(row)}"

    return {
        "s_no": game["s_no"],
        "home_team": home,
        "away_team": away,
        "home_sp": game.get("home_sp"),
        "away_sp": game.get("away_sp"),
        "features": row,
    }


# ── Elo 갱신 ──

def collect_daily_roster(date_str: str):
    """당일 1군 로스터 수집 및 저장."""
    year = int(date_str[:4])
    roster_file = DATA_DIR / f"rosters_{year}.json"

    if roster_file.exists():
        existing = json.loads(roster_file.read_text())
    else:
        existing = {}

    if date_str in existing and existing[date_str]:
        logger.info("로스터 이미 수집됨: %s", date_str)
        return existing[date_str]

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
        roster_file.write_text(json.dumps(existing, ensure_ascii=False, indent=1))
        logger.info("로스터 수집: %s — %d명", date_str, len(players))
        return players

    except Exception as e:
        logger.warning("로스터 수집 실패 (%s): %s", date_str, e)
        return []


def collect_daily_schedule(date_str: str):
    """당일 일정 API 응답을 연도 schedule 캐시에 병합 저장."""
    target_dt = datetime.strptime(date_str, "%Y-%m-%d")
    year = target_dt.year
    sched_file = DATA_DIR / f"schedules_{year}.json"

    if sched_file.exists():
        existing = json.loads(sched_file.read_text())
    else:
        existing = []

    existing_by_s_no = {}
    for row in existing:
        s_no = row.get("s_no")
        if s_no:
            existing_by_s_no[int(s_no)] = row

    try:
        resp = get_game_schedule(year=target_dt.year, month=target_dt.month, day=target_dt.day)
        games_raw = _parse_schedule_response(resp)
    except Exception as e:
        logger.warning("일정 수집 실패 (%s): %s", date_str, e)
        return []

    merged = 0
    touched = []
    for g in games_raw:
        s_no = g.get("s_no")
        if not s_no:
            continue
        existing_by_s_no[int(s_no)] = g
        touched.append(g)
        merged += 1

    rows = sorted(
        existing_by_s_no.values(),
        key=lambda x: (x.get("year", 0), x.get("gameDate", 0), x.get("s_no", 0)),
    )
    sched_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    logger.info("일정 캐시 갱신: %s — %d경기 병합", date_str, merged)
    return touched


def update_elo_from_yesterday(
    elo_engine: EloEngine,
    year: int, month: int, day: int,
):
    """전일 경기 결과로 Elo 갱신."""
    yesterday = datetime(year, month, day) - timedelta(days=1)
    resp = get_game_schedule(
        year=yesterday.year, month=yesterday.month, day=yesterday.day
    )

    games_raw = _parse_schedule_response(resp)
    if not games_raw:
        return

    updated = 0
    for g in games_raw:
        home_score = g.get("homeScore")
        away_score = g.get("awayScore")
        if home_score is None or away_score is None:
            continue

        elo_engine.update(
            home_team=g["homeTeam"],
            away_team=g["awayTeam"],
            home_score=home_score,
            away_score=away_score,
            home_sp=g.get("homeSP"),
            away_sp=g.get("awaySP"),
        )
        updated += 1

    if updated > 0:
        elo_engine.save()
        logger.info("Elo 갱신: %d경기 반영 (총 %d경기)", updated, elo_engine.games_played)


# ── 시즌 상태 초기화 ──

def _reset_season_state():
    """세션 상태 초기화."""
    global _season_state
    _season_state = {
        "team_runs_history": defaultdict(list),
        "team_runs_scored": defaultdict(float),
        "team_games": defaultdict(int),
        "team_lineup_wrc_history": defaultdict(list),
        "team_results": defaultdict(list),
        "team_prev_game": {},
        "team_game_dates": defaultdict(set),
        "team_prev_s_code": {},
        "team_runs_allowed": defaultdict(float),
        "h2h_tracker": defaultdict(lambda: defaultdict(int)),
        "sp_vs_team_tracker": defaultdict(lambda: {"wins": 0, "total": 0}),
    }


def _is_completed_before(row: dict, before_dt: datetime | None) -> bool:
    """해당 schedule row가 before_dt 이전에 완료된 경기인지 판정."""
    if row.get("leagueType") != LEAGUE_REGULAR:
        return False
    if row.get("homeScore") is None or row.get("awayScore") is None:
        return False
    gd_ts = row.get("gameDate", 0)
    if gd_ts <= 0:
        return False
    game_date = datetime.fromtimestamp(gd_ts).date()
    hour, minute = _parse_game_clock(row.get("hm"))
    game_dt = datetime.combine(game_date, datetime.min.time()).replace(hour=hour, minute=minute)
    if before_dt and game_dt >= before_dt:
        return False
    return True


def _init_season_state(year: int, before_dt: datetime | None = None):
    """schedule 캐시에서 시즌 기록을 로드하여 _season_state 초기화."""
    st = _season_state
    sched_file = DATA_DIR / f"schedules_{year}.json"

    if not sched_file.exists():
        logger.info("schedule 파일 없음 — 시즌 상태 초기화 스킵")
        return

    scheds = json.loads(sched_file.read_text())

    for g in sorted(scheds, key=lambda x: (x.get("gameDate", 0), x.get("s_no", 0))):
        if not _is_completed_before(g, before_dt):
            continue

        gd_ts = g.get("gameDate", 0)
        game_date = datetime.fromtimestamp(gd_ts).date()
        home = g.get("homeTeam")
        away = g.get("awayTeam")
        hs = g.get("homeScore")
        aws = g.get("awayScore")
        s_no = g.get("s_no")

        if home is None or away is None or hs is None or aws is None:
            continue

        home_won = hs > aws

        # 득점/실점
        st["team_runs_scored"][home] += float(hs)
        st["team_runs_scored"][away] += float(aws)
        st["team_runs_allowed"][home] += float(aws)
        st["team_runs_allowed"][away] += float(hs)
        st["team_runs_history"][home].append(float(hs))
        st["team_runs_history"][away].append(float(aws))
        st["team_games"][home] += 1
        st["team_games"][away] += 1

        # 승패
        if hs != aws:
            st["team_results"][home].append(home_won)
            st["team_results"][away].append(not home_won)

        # 경기 날짜
        st["team_game_dates"][home].add(game_date)
        st["team_game_dates"][away].add(game_date)

        hour, _ = _parse_game_clock(g.get("hm"))
        s_code = g.get("s_code")

        st["team_prev_game"][home] = (game_date, hour)
        st["team_prev_game"][away] = (game_date, hour)
        if s_code:
            st["team_prev_s_code"][home] = s_code
            st["team_prev_s_code"][away] = s_code

        # H2H
        h2h_key = frozenset({home, away})
        st["h2h_tracker"][h2h_key]["total"] += 1
        if home_won:
            st["h2h_tracker"][h2h_key][home] += 1
        else:
            st["h2h_tracker"][h2h_key][away] += 1

        # SP vs Team
        home_sp = g.get("homeSP")
        away_sp = g.get("awaySP")
        if home_sp:
            key = (home_sp, away)
            st["sp_vs_team_tracker"][key]["total"] += 1
            if home_won:
                st["sp_vs_team_tracker"][key]["wins"] += 1
        if away_sp:
            key = (away_sp, home)
            st["sp_vs_team_tracker"][key]["total"] += 1
            if not home_won:
                st["sp_vs_team_tracker"][key]["wins"] += 1

    total = sum(st["team_games"].values()) // 2
    logger.info("시즌 상태 초기화: %d경기 로드 (%d년)", total, year)


# ── 메인 파이프라인 ──

def run(target_date: str = None, dry_run: bool = False):
    """일일 파이프라인 실행.

    Args:
        target_date: "2026-03-28" 형식. None이면 오늘.
        dry_run: True면 예측만 하고 제출하지 않음.
    """
    setup_logging()

    if target_date:
        dt = datetime.strptime(target_date, "%Y-%m-%d")
    else:
        dt = datetime.now()

    year, month, day = dt.year, dt.month, dt.day
    logger.info("=" * 60)
    logger.info("일일 파이프라인 시작: %d-%02d-%02d (v65 live 피처)", year, month, day)

    # ── 1. Elo 엔진 로드 ──
    elo = EloEngine()
    if not elo.load():
        logger.info("Elo 초기 상태 — 신규 시즌 시작")

    # ── 2. 전일 결과 반영 ──
    update_elo_from_yesterday(elo, year, month, day)

    # ── 2.5 시즌 상태 초기화 (v5 피처용) ──
    _reset_season_state()
    _init_season_state(year, before_dt=dt)

    # ── 3. XGBoost 모델 로드 (v65 → v64 → v63 → v62 → v61 → v6) ──
    xgb_model = load_model("xgb_v65")
    if xgb_model is None:
        xgb_model = load_model("xgb_v64")
    if xgb_model is None:
        xgb_model = load_model("xgb_v63")
    if xgb_model is None:
        xgb_model = load_model("xgb_v62")
    if xgb_model is None:
        xgb_model = load_model("xgb_v61")
    if xgb_model is None:
        xgb_model = load_model("xgb_v6")
    if xgb_model is None:
        logger.warning("XGBoost 모델 없음 — Elo 단독 예측 모드")

    # ── 4. 당일 1군 로스터 수집 ──
    date_str = f"{year}-{month:02d}-{day:02d}"
    collect_daily_roster(date_str)

    # ── 5. 당일 경기 조회 ──
    games = get_today_games(year, month, day)
    if not games:
        logger.info("당일 경기 없음. 종료.")
        return

    # ── 6. 피처 산출 + 예측 ──
    from pipeline.live_v65 import build_game_features_v65_live
    game_inputs = []
    for game in games:
        game["_datetime"] = dt
        gi = build_game_features_v65_live(game, elo, year, use_confirmed_lineup=True)
        game_inputs.append(gi)

    predictions = batch_predict(elo, xgb_model, game_inputs)

    # ── 7. 결과 출력 ──
    logger.info("─" * 40)
    for pred, game in zip(predictions, games):
        home_name = TEAM_CODES.get(game["home_team"], "?")
        away_name = TEAM_CODES.get(game["away_team"], "?")
        logger.info(
            "경기 %d: %s(홈) vs %s(원정) | %s vs %s → 홈 승률 %.2f%%",
            pred["s_no"], home_name, away_name,
            game.get("home_sp_name", "?"), game.get("away_sp_name", "?"),
            pred["percent"],
        )
    logger.info("─" * 40)

    # ── 8. 제출 ──
    if dry_run:
        logger.info("DRY RUN — 제출 스킵")
        out_file = DATA_DIR / f"predictions_{year}{month:02d}{day:02d}.json"
        out_file.write_text(json.dumps(predictions, indent=2, ensure_ascii=False))
        logger.info("예측 결과 저장: %s", out_file)
    else:
        results = submit_batch(predictions)
        logger.info("제출 완료")

    logger.info("일일 파이프라인 종료")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="KBO 승부예측 일일 파이프라인 (v5)")
    parser.add_argument("--date", type=str, default=None,
                        help="대상 날짜 (YYYY-MM-DD). 기본: 오늘")
    parser.add_argument("--dry-run", action="store_true",
                        help="제출하지 않고 예측만 수행")
    args = parser.parse_args()

    run(target_date=args.date, dry_run=args.dry_run)
