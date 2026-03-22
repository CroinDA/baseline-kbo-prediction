"""
스탯티즈 승부예측 API 설정.
API 키는 config/api_key.txt 파일에 저장 (.gitignore 대상).
"""
import os
from pathlib import Path

# ── Base URL ──
BASE_URL = "https://api.statiz.co.kr/baseballApi"

# ── API Key ──
_KEY_FILE = Path(__file__).parent / "api_key.txt"


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
