"""
실전 제출 자동화 오케스트레이터.

운영 규칙:
1. 아침 첫 tick 때 당일 schedule 캐시 갱신
2. 경기 시작 50분 전 1차 제출
   - stage 시작 시 roster 갱신
   - 확정 라인업이 있으면 사용하고, 없으면 fallback 라인업 사용
3. 경기 시작 20분 전 라인업 확정 시 2차 제출
4. 더블헤더/동일 팀의 당일 선행 경기가 안 끝났으면 후속 경기 제출 보류
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.constants import ELO_INITIAL, ELO_REVERT_FACTOR
from data.collector import get_game_lineup
from elo.engine import EloEngine
import xgboost as xgb
from models.predict import batch_predict
from pipeline.build_v65_plus import add_comparison_features_live, COMPARISON_FEATURES
from pipeline.build_v6_timeaware import feature_names_v6
from pipeline.daily_run import (
    DATA_DIR,
    _init_season_state,
    _is_completed_before,
    _parse_game_clock,
    _reset_season_state,
    collect_daily_roster,
    collect_daily_schedule,
    get_today_games,
    setup_logging,
)
from pipeline.live_v65 import build_game_features_v65_live
from pipeline.live_retrain import retrain_live_v65
from pipeline.live_retrain_v65plus import retrain_live_v65plus
from pipeline.live_sync import sync_live_day
from pipeline.submit import submit_batch

logger = logging.getLogger(__name__)

STATE_DIR = DATA_DIR / "automation"
STATE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "saved"
LIVE_PIN_FILE = MODEL_DIR / "live_model.json"


MORNING_CHECK_HOUR = 10  # 오전 10시에 스케줄 fetch


def _state_path(date_str: str) -> Path:
    return STATE_DIR / f"auto_submit_state_{date_str.replace('-', '')}.json"


def _load_state(date_str: str) -> dict:
    path = _state_path(date_str)
    if not path.exists():
        return {
            "date": date_str,
            "schedule_refresh": {},
            "roster_refresh": {},
            "games": {},
        }
    return json.loads(path.read_text())


def _save_state(date_str: str, state: dict):
    _state_path(date_str).write_text(json.dumps(state, ensure_ascii=False, indent=2))


def _read_cached_game_times(date_str: str) -> list[datetime] | None:
    """로컬 캐시에서 당일 경기 시각 목록을 읽는다. API 호출 없음."""
    y, m, d = map(int, date_str.split("-"))
    sched_file = DATA_DIR / f"schedules_{y}.json"
    if not sched_file.exists():
        return None
    try:
        rows = json.loads(sched_file.read_text())
    except Exception:
        return None
    game_times = []
    for row in rows:
        if row.get("month") != m or row.get("day") != d:
            continue
        if row.get("leagueType") != 10100:  # LEAGUE_REGULAR
            continue
        hm = row.get("hm", "14:00:00")
        try:
            parts = hm.split(":")
            game_times.append(datetime(y, m, d, int(parts[0]), int(parts[1])))
        except Exception:
            game_times.append(datetime(y, m, d, 14, 0))
    return game_times if game_times else None


def _should_tick(current_dt: datetime, date_str: str) -> tuple[bool, str]:
    """API 호출 없이 지금 tick을 실행해야 하는지 판단.

    Returns:
        (should_run, reason)
    """
    state = _load_state(date_str)

    # 1) 스케줄 아직 미수집 → 오전 10시 이후 첫 1회 허용
    if not state.get("schedule_refresh", {}).get("completed_at"):
        if current_dt.hour >= MORNING_CHECK_HOUR:
            return True, "morning_schedule_fetch"
        return False, "too_early_no_schedule"

    # 2) 로컬 캐시에서 경기 시각 읽기
    game_times = _read_cached_game_times(date_str)
    if game_times is None:
        return False, "no_games_today"

    # 3) postgame sync 미완료 + 마지막 경기 종료 추정(+3h) 이후 → 허용
    if not state.get("postgame_sync", {}).get("completed_at"):
        latest_game = max(game_times)
        if current_dt >= latest_game + timedelta(hours=3):
            return True, "postgame_sync"

    # 4) T-50 / T-20 윈도우 체크 (각 ±5분 버퍼)
    for gt in game_times:
        # T-50: game-55min ~ game-15min
        if gt - timedelta(minutes=55) <= current_dt <= gt - timedelta(minutes=15):
            return True, f"t50_window({gt:%H:%M})"
        # T-20: game-25min ~ game+5min
        if gt - timedelta(minutes=25) <= current_dt <= gt + timedelta(minutes=5):
            return True, f"t20_window({gt:%H:%M})"

    return False, "outside_all_windows"


def _manual_new_season(elo: EloEngine):
    for code in list(elo.ratings.keys()):
        prev = elo.ratings[code]
        elo.ratings[code] = prev + ELO_REVERT_FACTOR * (ELO_INITIAL - prev)
    elo.games_played = 0


def _replay_elo_until(target_dt: datetime, start_year: int = 2023) -> EloEngine:
    elo = EloEngine(ratings_path=str(STATE_DIR / "ignore_ratings.json"))

    first_year = True
    for year in range(start_year, target_dt.year + 1):
        sched_file = DATA_DIR / f"schedules_{year}.json"
        if not sched_file.exists():
            continue
        if not first_year:
            _manual_new_season(elo)
        first_year = False

        rows = json.loads(sched_file.read_text())
        for row in sorted(rows, key=lambda x: (x.get("gameDate", 0), x.get("s_no", 0))):
            if not _is_completed_before(row, target_dt):
                continue
            elo.update(
                home_team=row["homeTeam"],
                away_team=row["awayTeam"],
                home_score=row["homeScore"],
                away_score=row["awayScore"],
                home_sp=row.get("homeSP"),
                away_sp=row.get("awaySP"),
            )
    return elo


def _get_model_num_features(model_path: Path) -> int | None:
    try:
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        return int(model.get_booster().num_features())
    except Exception:
        logger.exception("모델 피처 수 판독 실패: %s", model_path.name)
        return None


def _load_model_by_path(model_path: Path):
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    return model


def _load_live_model(expected_feature_count: int):
    candidates: list[Path] = []

    if LIVE_PIN_FILE.exists():
        try:
            pinned = json.loads(LIVE_PIN_FILE.read_text())
            model_name = pinned.get("model_name")
            if model_name:
                path = MODEL_DIR / f"{model_name}.json"
                if path.exists():
                    candidates.append(path)
        except Exception:
            logger.exception("live_model.json 파싱 실패")

    versioned = [
        p for p in MODEL_DIR.glob("xgb_v*.json")
        if re.fullmatch(r"xgb_v\d+\w*\.json", p.name)
    ]
    versioned.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for p in versioned:
        if p not in candidates:
            candidates.append(p)

    for path in candidates:
        feature_count = _get_model_num_features(path)
        if feature_count != expected_feature_count:
            continue
        logger.info("live 모델 선택: %s (%d features)", path.stem, feature_count)
        return _load_model_by_path(path), path.stem

    logger.warning("호환되는 live 모델 없음 — Elo 단독 예측 사용 (expected=%d)", expected_feature_count)
    return None, "elo_only"


def _team_lineup_ready(players: list[dict]) -> tuple[bool, dict]:
    orders = set()
    confirmed_orders = set()
    sp_ready = False

    for row in players:
        order_raw = str(row.get("battingOrder", ""))
        state = row.get("lineupState")
        if order_raw == "P":
            if state == "Y":
                sp_ready = True
            continue
        try:
            order = int(order_raw)
        except (TypeError, ValueError):
            continue
        if 1 <= order <= 9 and row.get("starting") == "Y":
            orders.add(order)
            if state == "Y":
                confirmed_orders.add(order)

    ready = orders == set(range(1, 10)) and confirmed_orders == set(range(1, 10)) and sp_ready
    return ready, {
        "starting_orders": sorted(orders),
        "confirmed_orders": sorted(confirmed_orders),
        "sp_ready": sp_ready,
    }


def _get_lineup_status(game: dict) -> dict:
    try:
        resp = get_game_lineup(game["s_no"])
    except Exception as e:
        return {"ready": False, "error": str(e)}

    home_players = resp.get(str(game["home_team"]), [])
    away_players = resp.get(str(game["away_team"]), [])
    home_ready, home_meta = _team_lineup_ready(home_players if isinstance(home_players, list) else [])
    away_ready, away_meta = _team_lineup_ready(away_players if isinstance(away_players, list) else [])
    return {
        "ready": bool(home_ready and away_ready),
        "home": home_meta,
        "away": away_meta,
    }


def _has_pending_prior_game(game: dict, all_games: list[dict]) -> bool:
    current_teams = {game["home_team"], game["away_team"]}
    for other in all_games:
        if other["s_no"] == game["s_no"]:
            continue
        if other["_datetime"] >= game["_datetime"]:
            continue
        if not current_teams.intersection({other["home_team"], other["away_team"]}):
            continue
        if other.get("home_score") is None or other.get("away_score") is None:
            return True
    return False


def _prepare_runtime(target_dt: datetime):
    elo = _replay_elo_until(target_dt)
    _reset_season_state()
    _init_season_state(target_dt.year, before_dt=target_dt)
    return elo


def _load_v8_config() -> dict | None:
    """v8 config 로드. 없으면 None."""
    config_path = MODEL_DIR / "v8_config.json"
    if not config_path.exists():
        return None
    try:
        return json.loads(config_path.read_text())
    except Exception:
        logger.warning("v8_config.json 파싱 실패")
        return None


def _predict_due_games(games: list[dict], target_dt: datetime, use_confirmed_lineup: bool) -> tuple[list[dict], str]:
    elo = _prepare_runtime(target_dt)
    game_inputs = []
    for game in games:
        game_inputs.append(
            build_game_features_v65_live(game, elo, target_dt.year, use_confirmed_lineup=use_confirmed_lineup)
        )

    v65_names = feature_names_v6(include_bench=False, include_sp_war=True)
    v65plus_count = len(v65_names) + len(COMPARISON_FEATURES)

    # Try v8 first (229 features, same as v65+ but with optimized inference)
    v8_config = _load_v8_config()
    v8_model_path = MODEL_DIR / "xgb_v8.json"
    if v8_config and v8_model_path.exists():
        nf = _get_model_num_features(v8_model_path)
        if nf == v65plus_count:
            model_v8 = _load_model_by_path(v8_model_path)
            for gi in game_inputs:
                gi["features"] = add_comparison_features_live(gi["features"], v65_names)
            threshold = v8_config.get("threshold", 0.5)
            deadzone_push = v8_config.get("deadzone_push", 0.0)
            logger.info("v8 모델 사용: threshold=%.3f, deadzone_push=%.3f", threshold, deadzone_push)
            return batch_predict(elo, model_v8, game_inputs,
                                 threshold=threshold, deadzone_push=deadzone_push), "xgb_v8"

    # Fallback: v65+ (229 features)
    model_plus, name_plus = _load_live_model(v65plus_count)
    if model_plus is not None:
        for gi in game_inputs:
            gi["features"] = add_comparison_features_live(gi["features"], v65_names)
        return batch_predict(elo, model_plus, game_inputs), name_plus

    # Fallback: v65 (219 features)
    feature_count = len(game_inputs[0]["features"]) if game_inputs else 0
    model, model_name = _load_live_model(feature_count)
    return batch_predict(elo, model, game_inputs), model_name


def _record_submission(state: dict, s_no: int, stage: str, pred: dict, model_name: str, dry_run: bool):
    game_state = state["games"].setdefault(str(s_no), {})
    entry = {
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
        "percent": pred["percent"],
        "model_name": model_name,
        "dry_run": dry_run,
    }
    if dry_run:
        game_state[f"{stage}_preview"] = entry
    else:
        game_state[stage] = entry


def _all_games_closed_for_sync(games: list[dict], current_dt: datetime) -> bool:
    if not games:
        return False
    for game in games:
        game_dt = game["_datetime"]
        if current_dt < game_dt:
            return False
        if game.get("state") in {3, 4, 5}:
            continue
        if game.get("home_score") is not None and game.get("away_score") is not None:
            continue
        return False
    return True


def _read_cached_games(date_str: str) -> list[dict]:
    """로컬 캐시에서 당일 경기 정보를 읽는다. API 호출 없음."""
    y, m, d = map(int, date_str.split("-"))
    sched_file = DATA_DIR / f"schedules_{y}.json"
    if not sched_file.exists():
        return []
    try:
        rows = json.loads(sched_file.read_text())
    except Exception:
        return []
    games = []
    for row in rows:
        if row.get("month") != m or row.get("day") != d:
            continue
        if row.get("leagueType") != 10100:
            continue
        hm = row.get("hm", "14:00:00")
        parts = hm.split(":")
        game_dt = datetime(y, m, d, int(parts[0]), int(parts[1]))
        games.append({
            "s_no": row["s_no"],
            "home_team": row["homeTeam"],
            "away_team": row["awayTeam"],
            "home_score": row.get("homeScore"),
            "away_score": row.get("awayScore"),
            "state": row.get("state"),
            "_datetime": game_dt,
        })
    return sorted(games, key=lambda g: (g["_datetime"], g["s_no"]))


def _maybe_run_postgame_sync(date_str: str, current_dt: datetime):
    state = _load_state(date_str)
    state.setdefault("postgame_sync", {})
    state.setdefault("postgame_retrain", {})
    sync_done = bool(state["postgame_sync"].get("completed_at"))
    retrain_done = bool(state["postgame_retrain"].get("completed_at"))

    if sync_done and retrain_done:
        return

    if not sync_done:
        # 먼저 로컬 캐시로 경기 존재 여부 확인 (API 호출 없음)
        games = _read_cached_games(date_str)
        if not games:
            return
        if not _all_games_closed_for_sync(games, current_dt):
            return

        result = sync_live_day(date_str)
        state["postgame_sync"] = {
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            **result,
        }
        _save_state(date_str, state)
        logger.info(
            "postgame live sync 완료: %s games=%s batters=%s pitchers=%s",
            date_str, result.get("games"), result.get("batters"), result.get("pitchers"),
        )

    if not retrain_done:
        train_years = [2023, 2024, 2025, int(date_str[:4])]
        retrain = retrain_live_v65(train_years)
        # v65+ retrain alongside v65
        try:
            retrain_plus = retrain_live_v65plus(train_years)
            logger.info("postgame v65+ retrain: rows=%s acc=%s", retrain_plus.get("rows"), retrain_plus.get("mean_accuracy"))
        except Exception:
            logger.exception("postgame v65+ retrain 실패 — v65만 유지")
        # v8 retrain (v65+ fallback 유지)
        try:
            from pipeline.live_retrain_v8 import retrain_live_v8
            retrain_v8 = retrain_live_v8(train_years)
            logger.info("postgame v8 retrain: rows=%s acc=%s tuned=%s",
                        retrain_v8.get("rows"), retrain_v8.get("mean_accuracy"),
                        retrain_v8.get("mean_accuracy_tuned"))
        except Exception:
            logger.exception("postgame v8 retrain 실패 — v65+ fallback 유지")
        state["postgame_retrain"] = {
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            **retrain,
        }
        _save_state(date_str, state)
        logger.info(
            "postgame live retrain 완료: %s rows=%s acc=%s",
            date_str, retrain.get("rows"), retrain.get("mean_accuracy"),
        )


def tick(target_date: str | None = None, now: str | None = None, dry_run: bool = False, force: bool = False) -> dict:
    setup_logging()

    if now:
        current_dt = datetime.strptime(now, "%Y-%m-%d %H:%M")
    elif target_date:
        current_dt = datetime.strptime(target_date, "%Y-%m-%d")
    else:
        current_dt = datetime.now()

    date_str = current_dt.strftime("%Y-%m-%d")

    # ── 게이트: API 호출 없이 지금 실행할 필요가 있는지 판단 ──
    if not force:
        should_run, reason = _should_tick(current_dt, date_str)
        if not should_run:
            logger.debug("tick skip: %s (%s)", date_str, reason)
            return {"date": date_str, "submitted": [], "pending": [], "skipped": reason}
        logger.info("tick 진입: %s (%s)", date_str, reason)

    # 전일 미동기화분이 있으면 먼저 정리한다.
    prev_date_str = (current_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    _maybe_run_postgame_sync(prev_date_str, current_dt)

    state = _load_state(date_str)
    state.setdefault("schedule_refresh", {})
    state.setdefault("roster_refresh", {})
    state.setdefault("postgame_sync", {})
    state.setdefault("postgame_retrain", {})
    state.setdefault("games", {})

    # schedule 캐시 갱신 (API — 실패해도 로컬 캐시로 진행)
    try:
        collect_daily_schedule(date_str)
    except Exception:
        logger.warning("schedule API 갱신 실패 — 로컬 캐시 사용")

    if not state.get("schedule_refresh", {}).get("completed_at"):
        state["schedule_refresh"] = {
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        }

    # 로컬 캐시에서 경기 목록 로드 (API 호출 없음)
    games = _read_cached_games(date_str)
    if not games:
        _save_state(date_str, state)
        logger.info("당일 경기 없음: %s", date_str)
        return {"date": date_str, "submitted": [], "pending": []}

    if not state["postgame_sync"].get("completed_at") and _all_games_closed_for_sync(games, current_dt):
        _maybe_run_postgame_sync(date_str, current_dt)
        state = _load_state(date_str)

    due_stage1 = []
    due_stage2 = []
    pending = []

    for game in games:
        game_key = str(game["s_no"])
        gstate = state["games"].setdefault(game_key, {})
        game_dt = game["_datetime"]

        if _has_pending_prior_game(game, games):
            pending.append({"s_no": game["s_no"], "reason": "prior_game_pending"})
            continue

        if current_dt >= game_dt:
            if not gstate.get("t20"):
                pending.append({"s_no": game["s_no"], "reason": "game_started"})
            continue

        if game_dt - timedelta(minutes=50) <= current_dt < game_dt - timedelta(minutes=20):
            if not gstate.get("t50"):
                due_stage1.append(game)
            continue

        if current_dt >= game_dt - timedelta(minutes=20):
            lineup_status = _get_lineup_status(game)
            gstate["lineup_status"] = {
                "checked_at": datetime.now().isoformat(timespec="seconds"),
                **lineup_status,
            }
            if not gstate.get("t20"):
                if lineup_status.get("ready"):
                    due_stage2.append(game)
                else:
                    pending.append({"s_no": game["s_no"], "reason": "lineup_not_ready"})

    submitted = []

    if due_stage1:
        collect_daily_roster(date_str)
        state["roster_refresh"] = {
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        }
        preds, model_name = _predict_due_games(due_stage1, current_dt, use_confirmed_lineup=True)
        if not dry_run:
            submit_batch(preds)
        for pred in preds:
            _record_submission(state, pred["s_no"], "t50", pred, model_name, dry_run)
            submitted.append({"stage": "t50", **pred, "model_name": model_name})

    if due_stage2:
        preds, model_name = _predict_due_games(due_stage2, current_dt, use_confirmed_lineup=True)
        if not dry_run:
            submit_batch(preds)
        for pred in preds:
            _record_submission(state, pred["s_no"], "t20", pred, model_name, dry_run)
            submitted.append({"stage": "t20", **pred, "model_name": model_name})

    _save_state(date_str, state)
    logger.info(
        "tick 완료: date=%s, submitted=%d, pending=%d, dry_run=%s",
        date_str, len(submitted), len(pending), dry_run,
    )
    return {"date": date_str, "submitted": submitted, "pending": pending}


# ── 동적 game-runner plist 관리 ──

GAME_RUNNER_PLIST = Path.home() / "Library/LaunchAgents/com.croinda.kbo-game-runner.plist"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _build_game_day_schedule(game_times: list[datetime]) -> list[tuple[int, int]]:
    """경기 시각으로부터 T-50, T-20 실행 시각을 계산. postgame은 새벽 2시 별도 처리."""
    run_times: set[tuple[int, int]] = set()
    for gt in game_times:
        t50 = gt - timedelta(minutes=50)
        t20 = gt - timedelta(minutes=20)
        run_times.add((t50.hour, t50.minute))
        run_times.add((t20.hour, t20.minute))
    return sorted(run_times)


def _write_game_runner_plist(run_times: list[tuple[int, int]], day: int):
    """game-runner plist를 동적 생성하고 launchd에 등록."""
    import os
    import plistlib
    import subprocess

    # 기존 해제
    if GAME_RUNNER_PLIST.exists():
        subprocess.run(
            ["launchctl", "bootout", f"gui/{os.getuid()}", str(GAME_RUNNER_PLIST)],
            capture_output=True,
        )

    calendar_intervals = [{"Hour": h, "Minute": m, "Day": day} for h, m in run_times]

    plist = {
        "Label": "com.croinda.kbo-game-runner",
        "ProgramArguments": [
            "/bin/bash",
            str(PROJECT_ROOT / "scripts/auto_submit_tick.sh"),
        ],
        "WorkingDirectory": str(PROJECT_ROOT),
        "StartCalendarInterval": calendar_intervals,
        "StandardOutPath": str(PROJECT_ROOT / "logs/launchd_game_runner.out.log"),
        "StandardErrorPath": str(PROJECT_ROOT / "logs/launchd_game_runner.err.log"),
    }

    with open(GAME_RUNNER_PLIST, "wb") as f:
        plistlib.dump(plist, f)

    subprocess.run(
        ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(GAME_RUNNER_PLIST)],
        capture_output=True,
    )
    logger.info("game-runner plist 등록: %s", [f"{h:02d}:{m:02d}" for h, m in run_times])


def _unload_game_runner_plist():
    """game-runner plist 해제 및 삭제."""
    import os
    import subprocess

    if GAME_RUNNER_PLIST.exists():
        subprocess.run(
            ["launchctl", "bootout", f"gui/{os.getuid()}", str(GAME_RUNNER_PLIST)],
            capture_output=True,
        )
        GAME_RUNNER_PLIST.unlink(missing_ok=True)
        logger.info("game-runner plist 해제")


def nightly_retrain():
    """새벽 2시: 전날 경기 결과 수집 + 모델 재학습."""
    setup_logging()
    current_dt = datetime.now()
    prev_date_str = (current_dt - timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info("nightly retrain 시작: 대상 %s", prev_date_str)

    try:
        # 전날 스케줄 캐시 갱신
        try:
            collect_daily_schedule(prev_date_str)
        except Exception:
            logger.warning("전날 schedule API 실패 — 로컬 캐시 사용")

        # postgame sync + retrain
        _maybe_run_postgame_sync(prev_date_str, current_dt)
        logger.info("nightly retrain 완료: %s", prev_date_str)
    except Exception:
        logger.exception("nightly retrain 실패: %s", prev_date_str)


def morning_check():
    """오전 10시 스케줄 확인 + game-runner 동적 스케줄링."""
    setup_logging()
    current_dt = datetime.now()
    date_str = current_dt.strftime("%Y-%m-%d")

    # 전일 game-runner 해제
    _unload_game_runner_plist()

    # 전일 postgame 미처리분 확인
    prev_date_str = (current_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        _maybe_run_postgame_sync(prev_date_str, current_dt)
    except Exception:
        logger.exception("전일 postgame sync 실패")

    # 오늘 스케줄 fetch
    try:
        collect_daily_schedule(date_str)
    except Exception:
        logger.warning("Schedule API 실패 — 로컬 캐시 사용")

    # 상태 기록
    state = _load_state(date_str)
    state.setdefault("schedule_refresh", {})
    state["schedule_refresh"] = {
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    _save_state(date_str, state)

    # 오늘 경기 시각 확인
    game_times = _read_cached_game_times(date_str)
    if not game_times:
        logger.info("오늘 경기 없음 (%s) — game-runner 미설정", date_str)
        return

    # game-runner plist 동적 생성
    run_schedule = _build_game_day_schedule(game_times)
    _write_game_runner_plist(run_schedule, current_dt.day)

    game_strs = [f"{gt:%H:%M}" for gt in game_times]
    run_strs = [f"{h:02d}:{m:02d}" for h, m in run_schedule]
    logger.info("경기 %d개 (%s) → 실행 시각: %s", len(game_times), game_strs, run_strs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="KBO 자동 제출 오케스트레이터")
    parser.add_argument("--date", type=str, default=None, help="대상 날짜 YYYY-MM-DD")
    parser.add_argument("--now", type=str, default=None, help="테스트용 현재시각 YYYY-MM-DD HH:MM")
    parser.add_argument("--dry-run", action="store_true", help="제출 없이 상태만 기록")
    parser.add_argument("--force", action="store_true", help="게이트 무시하고 강제 실행")
    parser.add_argument("--morning-check", action="store_true", help="오전 스케줄 확인 + game-runner 설정")
    parser.add_argument("--nightly-retrain", action="store_true", help="새벽 전날 경기 결과 수집 + 재학습")
    args = parser.parse_args()

    if args.nightly_retrain:
        nightly_retrain()
    elif args.morning_check:
        morning_check()
    else:
        tick(target_date=args.date, now=args.now, dry_run=args.dry_run, force=args.force)
