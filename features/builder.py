"""
피처 빌더 — 10개 핵심 피처를 Differential 형태로 산출.

모든 피처는 (홈 - 원정) 차이로 계산하여 모델 입력으로 사용.
라인업 미확정 시 팀 시즌 평균으로 폴백.
"""
import logging
from dataclasses import dataclass
from typing import Optional

from config.constants import (
    PYTHAGOREAN_EXPONENT,
    ROLLING_WINDOW_RECENT,
    BULLPEN_LOAD_DAYS,
    NIGHT_GAME_HOUR,
)

logger = logging.getLogger(__name__)


@dataclass
class GameFeatures:
    """단일 경기의 10개 피처."""

    elo_diff: float               # 1. 홈-원정 Elo 차이 (SP 보정 포함)
    home_pyth_diff: float         # 2. 피타고리안 기대승률 차이
    sp_fip_diff: float            # 3. 선발 투수 FIP 차이
    sp_k_bb_diff: float           # 4. 선발 투수 K/BB 차이
    lineup_wrc_plus_diff: float   # 5. 라인업 wRC+ 차이
    recent_10g_winpct_diff: float # 6. 최근 10경기 승률 차이
    bullpen_load_diff: float      # 7. 불펜 피로도 차이 (최근 3일 이닝)
    rest_days_diff: float         # 8. 휴식일 차이
    temperature: float            # 9. 기온 (°C)
    is_night: float               # 10. 야간 경기 여부 (0 or 1)

    def to_list(self) -> list[float]:
        return [
            self.elo_diff,
            self.home_pyth_diff,
            self.sp_fip_diff,
            self.sp_k_bb_diff,
            self.lineup_wrc_plus_diff,
            self.recent_10g_winpct_diff,
            self.bullpen_load_diff,
            self.rest_days_diff,
            self.temperature,
            self.is_night,
        ]

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "elo_diff",
            "home_pyth_diff",
            "sp_fip_diff",
            "sp_k_bb_diff",
            "lineup_wrc_plus_diff",
            "recent_10g_winpct_diff",
            "bullpen_load_diff",
            "rest_days_diff",
            "temperature",
            "is_night",
        ]


class FeatureBuilder:
    """경기 피처를 산출하는 빌더.

    외부에서 데이터를 주입받아 피처를 계산합니다.
    API 호출은 하지 않음 — 파이프라인에서 데이터를 전달.
    """

    # ── 피타고리안 기대승률 ──

    @staticmethod
    def pythagorean_expectation(runs_scored: float, runs_allowed: float) -> float:
        """피타고리안 기대승률.

        RS^e / (RS^e + RA^e), e=1.83
        """
        if runs_scored <= 0 and runs_allowed <= 0:
            return 0.5
        rs_e = runs_scored ** PYTHAGOREAN_EXPONENT
        ra_e = runs_allowed ** PYTHAGOREAN_EXPONENT
        denom = rs_e + ra_e
        if denom == 0:
            return 0.5
        return rs_e / denom

    # ── 라인업 wRC+ 평균 ──

    @staticmethod
    def lineup_avg_wrc_plus(
        lineup_players: Optional[list[dict]],
        team_avg_wrc_plus: float = 100.0,
    ) -> float:
        """라인업 선수들의 가중 평균 wRC+.

        lineup_players: [{"p_no": int, "wrc_plus": float, "batting_order": int}, ...]
        라인업 미확정 시 team_avg_wrc_plus 사용 (폴백).
        """
        if not lineup_players:
            return team_avg_wrc_plus

        # 타순 가중치: 1~4번(클린업) 비중 높게
        ORDER_WEIGHTS = {1: 1.2, 2: 1.1, 3: 1.3, 4: 1.3, 5: 1.1,
                         6: 1.0, 7: 0.9, 8: 0.8, 9: 0.7}

        total_weight = 0.0
        weighted_sum = 0.0
        for p in lineup_players:
            order = p.get("batting_order", 5)
            w = ORDER_WEIGHTS.get(order, 1.0)
            wrc = p.get("wrc_plus", team_avg_wrc_plus)
            weighted_sum += w * wrc
            total_weight += w

        if total_weight == 0:
            return team_avg_wrc_plus
        return weighted_sum / total_weight

    # ── 최근 N경기 승률 ──

    @staticmethod
    def recent_win_pct(
        recent_results: list[bool],
        window: int = ROLLING_WINDOW_RECENT,
    ) -> float:
        """최근 N경기 승률.

        recent_results: [True, False, True, ...] (최신이 마지막)
        """
        if not recent_results:
            return 0.5
        tail = recent_results[-window:]
        return sum(tail) / len(tail)

    # ── 불펜 피로도 ──

    @staticmethod
    def bullpen_load(recent_bullpen_ip: list[float]) -> float:
        """최근 3일 불펜 총 이닝.

        recent_bullpen_ip: [1.2, 3.0, 2.1] (최근 3일)
        """
        return sum(recent_bullpen_ip[-BULLPEN_LOAD_DAYS:])

    # ── 통합 피처 산출 ──

    def build(
        self,
        # Elo
        elo_diff: float,
        # 팀 득실점
        home_rs: float, home_ra: float,
        away_rs: float, away_ra: float,
        # 선발 투수
        home_sp_fip: float, away_sp_fip: float,
        home_sp_k_bb: float, away_sp_k_bb: float,
        # 라인업
        home_lineup: Optional[list[dict]] = None,
        away_lineup: Optional[list[dict]] = None,
        home_team_wrc: float = 100.0,
        away_team_wrc: float = 100.0,
        # 최근 폼
        home_recent: Optional[list[bool]] = None,
        away_recent: Optional[list[bool]] = None,
        # 불펜
        home_bp_ip: Optional[list[float]] = None,
        away_bp_ip: Optional[list[float]] = None,
        # 휴식
        home_rest: int = 1,
        away_rest: int = 1,
        # 환경
        temperature: float = 15.0,
        game_hour: int = 18,
    ) -> GameFeatures:
        """10개 피처를 한번에 산출."""

        # 1. Elo 차이 (이미 SP 보정 포함된 값을 받음)
        f_elo_diff = elo_diff

        # 2. 피타고리안 기대승률 차이
        home_pyth = self.pythagorean_expectation(home_rs, home_ra)
        away_pyth = self.pythagorean_expectation(away_rs, away_ra)
        f_pyth_diff = home_pyth - away_pyth

        # 3. 선발 투수 FIP 차이 (낮을수록 좋으므로 away - home)
        f_sp_fip_diff = away_sp_fip - home_sp_fip

        # 4. 선발 투수 K/BB 차이 (높을수록 좋으므로 home - away)
        f_sp_k_bb_diff = home_sp_k_bb - away_sp_k_bb

        # 5. 라인업 wRC+ 차이
        home_wrc = self.lineup_avg_wrc_plus(home_lineup, home_team_wrc)
        away_wrc = self.lineup_avg_wrc_plus(away_lineup, away_team_wrc)
        f_lineup_diff = home_wrc - away_wrc

        # 6. 최근 10경기 승률 차이
        home_rpct = self.recent_win_pct(home_recent or [])
        away_rpct = self.recent_win_pct(away_recent or [])
        f_recent_diff = home_rpct - away_rpct

        # 7. 불펜 피로도 차이 (높을수록 나쁨 → away - home)
        home_bp = self.bullpen_load(home_bp_ip or [])
        away_bp = self.bullpen_load(away_bp_ip or [])
        f_bp_diff = away_bp - home_bp  # 상대 불펜이 지쳐있으면 홈팀 유리

        # 8. 휴식일 차이
        f_rest_diff = float(home_rest - away_rest)

        # 9. 기온
        f_temp = temperature

        # 10. 야간 경기
        f_night = 1.0 if game_hour >= NIGHT_GAME_HOUR else 0.0

        return GameFeatures(
            elo_diff=f_elo_diff,
            home_pyth_diff=f_pyth_diff,
            sp_fip_diff=f_sp_fip_diff,
            sp_k_bb_diff=f_sp_k_bb_diff,
            lineup_wrc_plus_diff=f_lineup_diff,
            recent_10g_winpct_diff=f_recent_diff,
            bullpen_load_diff=f_bp_diff,
            rest_days_diff=f_rest_diff,
            temperature=f_temp,
            is_night=f_night,
        )
