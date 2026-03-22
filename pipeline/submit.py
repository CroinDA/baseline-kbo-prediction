"""
예측 결과 제출 모듈.

제출 전 검증 + 로깅 + 결과 확인.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

from data.collector import submit_prediction

logger = logging.getLogger(__name__)

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def _log_submission(s_no: int, percent: float, response: dict):
    """제출 이력을 JSONL로 기록."""
    log_file = LOG_DIR / f"submissions_{datetime.now():%Y%m}.jsonl"
    entry = {
        "timestamp": datetime.now().isoformat(),
        "s_no": s_no,
        "percent": percent,
        "result_cd": response.get("result_cd") or response.get("cdoe"),
        "result_msg": response.get("result_msg", ""),
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def submit_single(s_no: int, percent: float) -> dict:
    """단일 경기 제출.

    Args:
        s_no: 경기번호
        percent: 홈팀 승리확률 (소수점 둘째 자리)

    Returns:
        API 응답
    """
    logger.info("제출: 경기 %d → %.2f%%", s_no, percent)

    response = submit_prediction(s_no, percent)
    _log_submission(s_no, percent, response)

    result_cd = response.get("result_cd") or response.get("cdoe")
    if result_cd == 200:
        logger.info("제출 성공: 경기 %d", s_no)
    else:
        logger.error("제출 실패: 경기 %d — %s", s_no, response.get("result_msg"))

    return response


def submit_batch(predictions: list[dict]) -> list[dict]:
    """여러 경기 일괄 제출.

    Args:
        predictions: [{"s_no": int, "percent": float}, ...]

    Returns:
        각 경기의 API 응답 리스트
    """
    results = []
    success = 0
    fail = 0

    for pred in predictions:
        try:
            resp = submit_single(pred["s_no"], pred["percent"])
            results.append({"s_no": pred["s_no"], "response": resp})

            result_cd = resp.get("result_cd") or resp.get("cdoe")
            if result_cd == 200:
                success += 1
            else:
                fail += 1
        except Exception as e:
            logger.error("제출 예외: 경기 %d — %s", pred["s_no"], e)
            results.append({"s_no": pred["s_no"], "error": str(e)})
            fail += 1

    logger.info("제출 완료: 성공 %d / 실패 %d / 총 %d", success, fail, len(predictions))
    return results
