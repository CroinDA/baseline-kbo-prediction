"""
스탯티즈 승부예측 API 설정.

인증: HMAC-SHA256 서명 방식
- X-API-KEY: API 키
- X-TIMESTAMP: Unix epoch (초)
- X-SIGNATURE: HMAC-SHA256(secret, "METHOD|PATH|QUERY|TIMESTAMP")
"""
import os
import hmac
import hashlib
import time
from pathlib import Path
from urllib.parse import quote

# ── Base URL ──
BASE_URL = "https://api.statiz.co.kr/baseballApi"

# ── API Key / Secret ──
_KEY_FILE = Path(__file__).parent / "api_key.txt"
_SECRET_FILE = Path(__file__).parent / "api_secret.txt"


def get_api_key() -> str:
    """API 키를 파일 또는 환경변수에서 로드."""
    if _KEY_FILE.exists():
        return _KEY_FILE.read_text().strip()
    key = os.environ.get("STATIZ_API_KEY", "")
    if not key:
        raise RuntimeError(
            "API 키 없음. config/api_key.txt 파일을 생성하거나 "
            "STATIZ_API_KEY 환경변수를 설정하세요."
        )
    return key


def get_api_secret() -> str:
    """API Secret을 파일 또는 환경변수에서 로드."""
    if _SECRET_FILE.exists():
        return _SECRET_FILE.read_text().strip()
    secret = os.environ.get("STATIZ_API_SECRET", "")
    if not secret:
        raise RuntimeError(
            "API Secret 없음. config/api_secret.txt 파일을 생성하거나 "
            "STATIZ_API_SECRET 환경변수를 설정하세요."
        )
    return secret


def generate_signature(
    method: str,
    path: str,
    params: dict = None,
) -> tuple[str, str]:
    """HMAC-SHA256 서명 생성.

    Args:
        method: HTTP 메서드 (GET/POST)
        path: 엔드포인트 경로 (/prediction/... 에서 /baseballApi 제외)
        params: 쿼리 파라미터 또는 POST body

    Returns:
        (timestamp, signature)
    """
    secret = get_api_secret()
    timestamp = str(int(time.time()))

    # PATH: /baseballApi 접두사 제거
    clean_path = path.lstrip("/")
    if clean_path.startswith("baseballApi/"):
        clean_path = clean_path[len("baseballApi/"):]

    # 쿼리 정규화: 키 알파벳순 정렬, URL 인코딩
    if params:
        normalized_query = "&".join(
            f"{quote(str(k), safe='')}={quote(str(v), safe='')}"
            for k, v in sorted(params.items())
        )
    else:
        normalized_query = ""

    # 페이로드: METHOD|PATH|QUERY|TIMESTAMP
    payload = f"{method}|{clean_path}|{normalized_query}|{timestamp}"

    signature = hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return timestamp, signature


# ── 엔드포인트 ──
ENDPOINTS = {
    # GET 엔드포인트
    "player_roster":    "/prediction/playerRoster",
    "player_season":    "/prediction/playerSeason",
    "player_situation": "/prediction/playerSituation",
    "team_record":      "/prediction/teamRecord",
    "player_day":       "/prediction/playerDay",
    "game_boxscore":    "/prediction/gameBoxscore",
    "game_schedule":    "/prediction/gameSchedule",
    "game_lineup":      "/prediction/gameLineup",
    # POST 엔드포인트
    "save_prediction":  "/prediction/savePrediction",
}

# ── Rate Limit (대회 규정: 1분 내 과도 요청 시 3분 차단) ──
REQUEST_DELAY_SEC = 1.0  # 요청 간 최소 대기 (초)
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 5.0
