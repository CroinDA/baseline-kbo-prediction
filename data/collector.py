"""
스탯티즈 API 데이터 수집 모듈.

인증: HMAC-SHA256 서명 (X-API-KEY + X-TIMESTAMP + X-SIGNATURE)
22개 엔드포인트에 대한 래퍼 함수 제공.
Rate limiting (1초 딜레이) 및 재시도 로직 포함.
"""
import time
import logging
from typing import Any, Optional

import requests

from config.api_config import (
    BASE_URL,
    ENDPOINTS,
    REQUEST_DELAY_SEC,
    MAX_RETRIES,
    RETRY_BACKOFF_SEC,
    get_api_key,
    generate_signature,
)
from config.constants import LEAGUE_REGULAR

logger = logging.getLogger(__name__)

# ── 마지막 요청 시각 (rate limiting) ──
_last_request_time: float = 0.0


def _request(
    method: str,
    endpoint_key: str,
    params: Optional[dict] = None,
    json_body: Optional[dict] = None,
) -> dict[str, Any]:
    """공통 API 요청 함수.

    HMAC-SHA256 서명 + Rate limiting + 재시도 + 에러 핸들링.
    """
    global _last_request_time

    endpoint_path = ENDPOINTS[endpoint_key]
    url = BASE_URL + endpoint_path
    api_key = get_api_key()

    # 서명 생성 (GET은 params, POST는 json_body 기반)
    sign_params = params if method == "GET" else json_body
    timestamp, signature = generate_signature(method, endpoint_path, sign_params)

    headers = {
        "X-API-KEY": api_key,
        "X-TIMESTAMP": timestamp,
        "X-SIGNATURE": signature,
        "Content-Type": "application/json",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        # Rate limiting
        elapsed = time.time() - _last_request_time
        if elapsed < REQUEST_DELAY_SEC:
            time.sleep(REQUEST_DELAY_SEC - elapsed)

        try:
            _last_request_time = time.time()

            # 재시도 시 타임스탬프/서명 갱신
            if attempt > 1:
                timestamp, signature = generate_signature(
                    method, endpoint_path, sign_params
                )
                headers["X-TIMESTAMP"] = timestamp
                headers["X-SIGNATURE"] = signature

            if method == "GET":
                resp = requests.get(url, headers=headers, params=params, timeout=30)
            else:
                resp = requests.post(url, headers=headers, json=json_body, timeout=30)

            resp.raise_for_status()
            data = resp.json()

            # API 자체 에러 코드 확인 (100=성공)
            result_cd = data.get("result_cd")
            if result_cd and result_cd != 100:
                logger.warning(
                    "API 응답 코드: %s — %s", result_cd, data.get("result_msg")
                )

            return data

        except requests.exceptions.RequestException as e:
            logger.warning("요청 실패 (시도 %d/%d): %s", attempt, MAX_RETRIES, e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC * attempt)
            else:
                raise

    return {}  # unreachable


# ══════════════════════════════════════════════════════════════
# GET 엔드포인트
# ══════════════════════════════════════════════════════════════


def get_player_roster(date: str) -> dict:
    """날짜 기준 로스터 선수 명단.

    Args:
        date: "2026-03-28" 형식
    """
    return _request("GET", "player_roster", params={"date": date})


def get_player_season(
    p_no: int,
    m2: str = "batting",
    year: Optional[int] = None,
    league_type: int = LEAGUE_REGULAR,
) -> dict:
    """선수 년도별 기본 기록.

    Args:
        p_no: 선수번호
        m2: "batting" 또는 "pitching"
        year: 년도 (None이면 전체)
        league_type: 리그 타입 (10100=정규시즌)
    """
    params = {"p_no": p_no, "m2": m2, "leagueType": league_type}
    if year:
        params["year"] = year
    return _request("GET", "player_season", params=params)


def get_player_situation(
    p_no: int,
    m2: str = "batting",
    m3: str = "time",
    year: Optional[int] = None,
    league_type: int = LEAGUE_REGULAR,
) -> dict:
    """선수 상황별 기록.

    Args:
        p_no: 선수번호
        m2: "batting" 또는 "pitching"
        m3: "time", "stadium", "situation", "ballcount", "type"
        year: 년도
        league_type: 리그 타입
    """
    params = {"p_no": p_no, "m2": m2, "m3": m3, "leagueType": league_type}
    if year:
        params["year"] = year
    return _request("GET", "player_situation", params=params)


def get_team_record(
    m2: str = "batting",
    year: Optional[int] = None,
    t_code: Optional[int] = None,
    league_type: int = LEAGUE_REGULAR,
) -> dict:
    """시즌 팀 기록실.

    Args:
        m2: "batting" 또는 "pitching"
        year: 년도
        t_code: 팀 코드 (None이면 전체)
        league_type: 리그 타입
    """
    params = {"m2": m2, "leagueType": league_type}
    if year:
        params["year"] = year
    if t_code:
        params["t_code"] = t_code
    return _request("GET", "team_record", params=params)


def get_player_day(
    p_no: int,
    m2: str = "batting",
    year: Optional[int] = None,
    month: Optional[int] = None,
) -> dict:
    """날짜별 선수 기록.

    Args:
        p_no: 선수번호
        m2: "batting" 또는 "pitching"
        year: 년도
        month: 월
    """
    params = {"p_no": p_no, "m2": m2}
    if year:
        params["year"] = year
    if month:
        params["month"] = month
    return _request("GET", "player_day", params=params)


def get_game_boxscore(s_no: int) -> dict:
    """박스스코어 및 경기 정보.

    Args:
        s_no: 경기번호
    """
    return _request("GET", "game_boxscore", params={"s_no": s_no})


def get_game_schedule(
    year: Optional[int] = None,
    month: Optional[int] = None,
    day: Optional[int] = None,
) -> dict:
    """경기일정 조회.

    Args:
        year: 년도
        month: 월
        day: 일
    """
    params = {}
    if year:
        params["year"] = year
    if month:
        params["month"] = month
    if day:
        params["day"] = day
    return _request("GET", "game_schedule", params=params)


def get_game_lineup(s_no: int) -> dict:
    """경기 라인업 조회.

    Args:
        s_no: 경기번호
    """
    return _request("GET", "game_lineup", params={"s_no": s_no})


# ══════════════════════════════════════════════════════════════
# POST 엔드포인트
# ══════════════════════════════════════════════════════════════


def submit_prediction(s_no: int, percent: float) -> dict:
    """승부예측 결과 제출.

    Args:
        s_no: 경기번호
        percent: 홈팀 승리확률 (소수점 둘째 자리, 예: 65.43)

    Returns:
        API 응답 (result_cd, result_msg)
    """
    # 소수점 둘째 자리 반올림
    percent = round(percent, 2)

    # 50.00% 방어
    if percent == 50.00:
        raise ValueError("50.00%는 대회 규정상 실패 처리됩니다.")

    return _request(
        "POST",
        "save_prediction",
        json_body={"s_no": s_no, "percent": percent},
    )
