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

from config.constants import TEAM_CODES, NIGHT_GAME_HOUR
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


def get_today_games(year: int, month: int, day: int) -> list[dict]:
    """당일 경기 목록 조회."""
    resp = get_game_schedule(year=year, month=month, day=day)
    games = []

    date_list = resp.get("date", [])
    if not date_list:
        logger.warning("당일 경기 없음: %d-%02d-%02d", year, month, day)
        return games

    for g in date_list:
        games.append({
            "s_no": g["s_no"],
            "home_team": g["homeTeam"],
            "away_team": g["awayTeam"],
            "home_sp": g.get("homeSP"),
            "away_sp": g.get("awaySP"),
            "home_sp_name": g.get("homeSPName", ""),
            "away_sp_name": g.get("awaySPName", ""),
            "temperature": g.get("temperature", 15.0),
            "humidity": g.get("humidity", 50),
            "game_hour": int(g.get("hm", "1800")[:2]) if g.get("hm") else 18,
            "s_code": g.get("s_code"),
        })

    logger.info("당일 경기 %d개 조회", len(games))
    return games


def get_sp_stats(p_no: int, year: int) -> dict:
    """선발 투수 시즌 스탯 조회."""
    if not p_no:
        return {"fip": 4.50, "k_bb": 2.0}  # 리그 평균 폴백

    try:
        resp = get_player_season(p_no, m2="pitching", year=year)
        basic_list = resp.get("basic", {}).get("list", [])
        deepen_list = resp.get("deepen", {}).get("list", [])

        if basic_list:
            b = basic_list[0]
            fip = b.get("FIP", 4.50)
        else:
            fip = 4.50

        if deepen_list:
            d = deepen_list[0]
            k_bb = d.get("KBB", 2.0)
        else:
            k_bb = 2.0

        return {"fip": fip or 4.50, "k_bb": k_bb or 2.0}

    except Exception as e:
        logger.warning("투수 %d 스탯 조회 실패: %s", p_no, e)
        return {"fip": 4.50, "k_bb": 2.0}


def build_game_features(
    game: dict,
    elo_engine: EloEngine,
    year: int,
) -> dict:
    """단일 경기의 피처를 산출."""
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

    # 피처 빌드 (일부 데이터는 시즌 초반 부족 → 기본값 사용)
    features = fb.build(
        elo_diff=elo_diff,
        # 팀 득실점 (시즌 초반에는 기본값)
        home_rs=4.5, home_ra=4.5,
        away_rs=4.5, away_ra=4.5,
        # 선발 투수
        home_sp_fip=home_sp_stats["fip"],
        away_sp_fip=away_sp_stats["fip"],
        home_sp_k_bb=home_sp_stats["k_bb"],
        away_sp_k_bb=away_sp_stats["k_bb"],
        # 라인업 (폴백: 팀 평균)
        home_lineup=None,
        away_lineup=None,
        home_team_wrc=100.0,
        away_team_wrc=100.0,
        # 최근 폼 (시즌 초반 부족)
        home_recent=None,
        away_recent=None,
        # 불펜 (기본값)
        home_bp_ip=None,
        away_bp_ip=None,
        # 휴식일 (기본값)
        home_rest=1,
        away_rest=1,
        # 환경
        temperature=game.get("temperature", 15.0),
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

    date_list = resp.get("date", [])
    if not date_list:
        return

    updated = 0
    for g in date_list:
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
