"""
일일 자동 파이프라인.

매일 실행 흐름:
1. 당일 경기일정 조회
2. 선발투수 + 라인업 확인
3. 피처 산출
4. Elo + XGBoost 블렌딩 예측
5. API 제출
6. 전일 결과 반영 (Elo 갱신)
"""
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.constants import (
    TEAM_CODES, NIGHT_GAME_HOUR, BULLPEN_LOAD_DAYS, LEAGUE_REGULAR,
)
from data.collector import (
    get_game_schedule,
    get_game_lineup,
    get_game_boxscore,
    get_player_season,
)
from elo.engine import EloEngine
from features.builder import FeatureBuilder, GameFeatures
from models.predict import predict_game, batch_predict
from models.train import load_model
from pipeline.submit import submit_batch

logger = logging.getLogger(__name__)

# ── 데이터 캐시 경로 ──
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


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
    """API 스케줄 응답에서 경기 리스트 추출.

    API 응답 형식: {"0401": [games], "result_cd": 100, ...}
    날짜 키(숫자4자리)의 값이 list인 항목만 추출.
    """
    games = []
    for key, value in resp.items():
        if isinstance(value, list):
            games.extend(value)
    return games


def get_today_games(year: int, month: int, day: int) -> list[dict]:
    """당일 경기 목록 조회."""
    resp = get_game_schedule(year=year, month=month, day=day)
    games_raw = _parse_schedule_response(resp)

    if not games_raw:
        logger.warning("당일 경기 없음: %d-%02d-%02d", year, month, day)
        return []

    games = []
    for g in games_raw:
        # 정규시즌만 (시범경기 제외)
        if g.get("leagueType") != LEAGUE_REGULAR:
            continue
        # hm 파싱: "18:30:00" 또는 "1800" 형식
        hm = str(g.get("hm", "18:00:00"))
        try:
            game_hour = int(hm.split(":")[0]) if ":" in hm else int(hm[:2])
        except (ValueError, IndexError):
            game_hour = 18

        games.append({
            "s_no": g["s_no"],
            "home_team": g["homeTeam"],
            "away_team": g["awayTeam"],
            "home_sp": g.get("homeSP"),
            "away_sp": g.get("awaySP"),
            "home_sp_name": g.get("homeSPName", ""),
            "away_sp_name": g.get("awaySPName", ""),
            "temperature": g.get("temperature") or 15.0,
            "humidity": g.get("humidity", 50),
            "game_hour": game_hour,
            "s_code": g.get("s_code"),
        })

    logger.info("당일 경기 %d개 조회", len(games))
    return games


def _find_year_record(records: list[dict], year: int) -> dict | None:
    """리스트에서 해당 연도 레코드를 찾아 반환.

    API year 필드가 문자열("2023") 또는 정수(2023)일 수 있으므로 둘 다 비교.
    """
    for rec in records:
        rec_year = rec.get("year")
        if rec_year is not None and str(rec_year) == str(year):
            return rec
    return None


def get_sp_stats(p_no: int, year: int) -> dict:
    """선발 투수 시즌 스탯 조회.

    API 응답은 전 시즌 데이터를 모두 반환하므로,
    반드시 year로 필터링하여 해당 시즌 레코드만 사용.
    """
    if not p_no:
        return {"fip": 4.50, "k_bb": 2.0}  # 리그 평균 폴백

    try:
        resp = get_player_season(p_no, m2="pitching", year=year)

        fip = 4.50
        k_bb = 2.0

        # basic/deepen 구조 (해당 year 레코드만 추출)
        basic_list = resp.get("basic", {}).get("list", [])
        deepen_list = resp.get("deepen", {}).get("list", [])

        basic_rec = _find_year_record(basic_list, year)
        deepen_rec = _find_year_record(deepen_list, year)

        if basic_rec:
            fip = float(basic_rec.get("FIP") or basic_rec.get("fip") or 4.50)
            so = float(basic_rec.get("SO") or basic_rec.get("K") or 0)
            bb = float(basic_rec.get("BB") or 1)
            if bb > 0 and so > 0:
                k_bb = so / bb
        if deepen_rec:
            k_bb = float(deepen_rec.get("KBB", k_bb) or k_bb)

        # fallback: 최상위 list 구조
        if not basic_rec:
            main_list = resp.get("list", [])
            if isinstance(main_list, list):
                main_rec = _find_year_record(main_list, year)
                if main_rec:
                    fip = float(main_rec.get("FIP") or main_rec.get("fip") or fip)
                    so = float(main_rec.get("SO") or main_rec.get("K") or 0)
                    bb = float(main_rec.get("BB") or 1)
                    if bb > 0 and so > 0:
                        k_bb = so / bb

        return {"fip": float(fip), "k_bb": float(k_bb)}

    except Exception as e:
        logger.warning("투수 %d 스탯 조회 실패: %s", p_no, e)
        return {"fip": 4.50, "k_bb": 2.0}


def get_lineup_wrc(s_no: int, year: int) -> tuple[list[dict], list[dict]]:
    """경기 라인업에서 각 선수의 OPS→wRC+ 조회.

    Returns:
        (home_lineup, away_lineup) — 각각 [{"batting_order": int, "wrc_plus": float}, ...]
    """
    try:
        resp = get_game_lineup(s_no)
        home_lineup = []
        away_lineup = []

        # API 응답 구조: {"list": [...], ...} 또는 {"home": [...], "away": [...]}
        lineup_list = resp.get("list", [])
        if not isinstance(lineup_list, list):
            lineup_list = []

        # home/away 분리된 구조도 확인
        home_list = resp.get("home", lineup_list)
        away_list = resp.get("away", [])

        if not isinstance(home_list, list):
            home_list = []
        if not isinstance(away_list, list):
            away_list = []

        # 통합 리스트에서 homeAway 필드로 분리
        if lineup_list and not away_list:
            for p in lineup_list:
                ha = p.get("homeAway", p.get("home_away", ""))
                if str(ha).upper() in ("H", "HOME", "1"):
                    home_list.append(p)
                elif str(ha).upper() in ("A", "AWAY", "2"):
                    away_list.append(p)

        def _process_lineup(players: list[dict]) -> list[dict]:
            result = []
            for p in players:
                p_no = p.get("p_no") or p.get("pNo")
                order = p.get("battingOrder") or p.get("batting_order") or p.get("batOrder", 5)
                if not p_no:
                    continue
                # OPS 조회 → wRC+ 변환
                try:
                    ps = get_player_season(p_no, m2="batting", year=year)
                    ops = 0.730  # 리그 평균 폴백
                    pl = ps.get("list", [])
                    if isinstance(pl, list) and pl:
                        ops = pl[0].get("OPS") or pl[0].get("ops") or 0.730
                    bl = ps.get("basic", {}).get("list", [])
                    if bl:
                        ops = bl[0].get("OPS", ops) or ops
                    wrc_plus = (float(ops) / 0.730) * 100.0
                except Exception:
                    wrc_plus = 100.0
                result.append({"batting_order": int(order), "wrc_plus": wrc_plus})
            return result

        home_lineup = _process_lineup(home_list)
        away_lineup = _process_lineup(away_list)

        if home_lineup:
            logger.info("라인업 wRC+ 조회: 홈 %d명, 원정 %d명", len(home_lineup), len(away_lineup))

        return home_lineup, away_lineup

    except Exception as e:
        logger.warning("라인업 조회 실패 (s_no=%d): %s", s_no, e)
        return [], []


def get_rest_days_and_bullpen(
    team_code: int, target_date: datetime, year: int, month: int,
) -> tuple[int, list[float]]:
    """팀의 휴식일과 불펜 피로도(최근 3일 경기 수) 계산.

    해당 월 일정에서 target_date 이전 경기를 찾아 계산.

    Returns:
        (rest_days, bullpen_ip_list)
    """
    try:
        resp = get_game_schedule(year=year, month=month)
        games = _parse_schedule_response(resp)

        # 정규시즌만 필터
        games = [g for g in games if g.get("leagueType") == LEAGUE_REGULAR]

        team_dates = []
        for g in games:
            if g.get("homeScore") is None and g.get("awayScore") is None:
                # 아직 완료 안 된 경기도 일정에 포함 (예정 경기)
                pass
            gd = g.get("gameDate", 0)
            if gd <= 0:
                continue
            game_dt = datetime.fromtimestamp(gd)
            game_date = game_dt.date()
            # target_date 이전 + 이 팀 경기
            if game_date < target_date.date():
                if g.get("homeTeam") == team_code or g.get("awayTeam") == team_code:
                    team_dates.append(game_date)

        if not team_dates:
            return 3, []  # 시즌 초 기본값

        team_dates.sort()
        last_game = team_dates[-1]
        rest_days = (target_date.date() - last_game).days

        # 불펜 피로도: 최근 3일 내 경기 수 × 평균 불펜 3.0IP
        bp_games = sum(
            1 for d in team_dates
            if 0 < (target_date.date() - d).days <= BULLPEN_LOAD_DAYS
        )
        bullpen_ip = [3.0] * bp_games

        return rest_days, bullpen_ip

    except Exception as e:
        logger.warning("휴식일/불펜 조회 실패 (team=%d): %s", team_code, e)
        return 1, []


def build_game_features(
    game: dict,
    elo_engine: EloEngine,
    year: int,
) -> dict:
    """단일 경기의 피처를 산출.

    Args:
        game: 경기 정보 dict
        elo_engine: Elo 엔진
        year: 시즌 연도
    """
    fb = FeatureBuilder()

    # 선발 투수 스탯
    home_sp_stats = get_sp_stats(game.get("home_sp"), year)
    away_sp_stats = get_sp_stats(game.get("away_sp"), year)

    # Elo 차이 (SP 보정 포함)
    home_elo = elo_engine.get_rating(game["home_team"])
    away_elo = elo_engine.get_rating(game["away_team"])
    elo_base_diff = home_elo - away_elo + 30  # 홈 이점

    sp_adj_home = elo_engine.get_sp_adjustment(
        game.get("home_sp", 0), game["home_team"]
    )
    sp_adj_away = elo_engine.get_sp_adjustment(
        game.get("away_sp", 0), game["away_team"]
    )
    elo_diff = elo_base_diff + sp_adj_home - sp_adj_away

    # 기온: 이상치 클램핑 (-10~40°C)
    temp = game.get("temperature", 15.0)
    if temp == 0 or temp > 45 or temp < -15:
        temp = 15.0
    temp = max(-10.0, min(40.0, float(temp)))

    # 라인업 wRC+ (API 호출)
    home_lineup, away_lineup = get_lineup_wrc(game["s_no"], year)

    # 휴식일 + 불펜 피로도
    target_dt = game.get("_datetime", datetime(year, game.get("_month", 3), game.get("_day", 28)))
    home_rest, home_bp = get_rest_days_and_bullpen(
        game["home_team"], target_dt, year, target_dt.month,
    )
    away_rest, away_bp = get_rest_days_and_bullpen(
        game["away_team"], target_dt, year, target_dt.month,
    )

    # 피처 빌드
    features = fb.build(
        elo_diff=elo_diff,
        home_rs=4.5, home_ra=4.5,
        away_rs=4.5, away_ra=4.5,
        home_sp_fip=home_sp_stats["fip"],
        away_sp_fip=away_sp_stats["fip"],
        home_sp_k_bb=home_sp_stats["k_bb"],
        away_sp_k_bb=away_sp_stats["k_bb"],
        home_lineup=home_lineup or None,
        away_lineup=away_lineup or None,
        home_team_wrc=100.0,
        away_team_wrc=100.0,
        home_recent=None,
        away_recent=None,
        home_bp_ip=home_bp or None,
        away_bp_ip=away_bp or None,
        home_rest=home_rest,
        away_rest=away_rest,
        temperature=temp,
        game_hour=game.get("game_hour", 18),
    )

    return {
        "s_no": game["s_no"],
        "home_team": game["home_team"],
        "away_team": game["away_team"],
        "home_sp": game.get("home_sp"),
        "away_sp": game.get("away_sp"),
        "features": features,
    }


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

        # 경기 완료 확인
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
    logger.info("일일 파이프라인 시작: %d-%02d-%02d", year, month, day)

    # ── 1. Elo 엔진 로드 ──
    elo = EloEngine()
    if not elo.load():
        logger.info("Elo 초기 상태 — 신규 시즌 시작")

    # ── 2. 전일 결과 반영 ──
    update_elo_from_yesterday(elo, year, month, day)

    # ── 3. XGBoost 모델 로드 ──
    xgb_model = load_model()
    if xgb_model is None:
        logger.warning("XGBoost 모델 없음 — Elo 단독 예측 모드")

    # ── 4. 당일 경기 조회 ──
    games = get_today_games(year, month, day)
    if not games:
        logger.info("당일 경기 없음. 종료.")
        return

    # ── 5. 피처 산출 + 예측 ──
    game_inputs = []
    for game in games:
        # 날짜 컨텍스트 주입 (rest_days/bullpen_load 계산용)
        game["_datetime"] = dt
        game["_month"] = month
        game["_day"] = day
        gi = build_game_features(game, elo, year)
        game_inputs.append(gi)

    predictions = batch_predict(elo, xgb_model, game_inputs)

    # ── 6. 결과 출력 ──
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

    # ── 7. 제출 ──
    if dry_run:
        logger.info("DRY RUN — 제출 스킵")
        # 파일로 저장
        out_file = DATA_DIR / f"predictions_{year}{month:02d}{day:02d}.json"
        out_file.write_text(json.dumps(predictions, indent=2, ensure_ascii=False))
        logger.info("예측 결과 저장: %s", out_file)
    else:
        results = submit_batch(predictions)
        logger.info("제출 완료")

    logger.info("일일 파이프라인 종료")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="KBO 승부예측 일일 파이프라인")
    parser.add_argument("--date", type=str, default=None,
                        help="대상 날짜 (YYYY-MM-DD). 기본: 오늘")
    parser.add_argument("--dry-run", action="store_true",
                        help="제출하지 않고 예측만 수행")
    args = parser.parse_args()

    run(target_date=args.date, dry_run=args.dry_run)
