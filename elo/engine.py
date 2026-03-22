"""
Elo Rating Engine for KBO.

FiveThirtyEight MLB Elo 방법론을 KBO에 맞게 조정.
- 팀 Elo 갱신 (승차 반영, 비선형 스케일링)
- 선발 투수 보정
- 홈 이점 반영
- 시즌 초기화 (1/3 평균 회귀)
"""
import json
import math
from pathlib import Path
from typing import Optional

from config.constants import (
    ELO_INITIAL,
    ELO_K_EARLY,
    ELO_K_NORMAL,
    ELO_K_TRANSITION_GAME,
    ELO_HOME_ADVANTAGE,
    ELO_REVERT_FACTOR,
    ELO_MOV_EXPONENT,
    ELO_MOV_MULTIPLIER,
    SP_ADJUSTMENT_ALPHA,
    TEAM_CODES,
)


class EloEngine:
    """KBO Elo Rating 엔진."""

    def __init__(self, ratings_path: Optional[str] = None):
        self.ratings: dict[int, float] = {}
        self.sp_ratings: dict[int, float] = {}  # 선수번호 → 투수 Elo
        self.team_sp_avg: dict[int, float] = {}  # 팀코드 → 팀 투수 평균
        self.games_played: int = 0
        self._ratings_path = ratings_path or str(
            Path(__file__).parent / "ratings.json"
        )
        self._init_ratings()

    def _init_ratings(self):
        """모든 팀을 초기 Elo로 설정."""
        for code in TEAM_CODES:
            self.ratings[code] = ELO_INITIAL

    def load(self) -> bool:
        """저장된 레이팅 로드. 성공 시 True."""
        path = Path(self._ratings_path)
        if not path.exists():
            return False
        data = json.loads(path.read_text())
        self.ratings = {int(k): v for k, v in data.get("ratings", {}).items()}
        self.sp_ratings = {int(k): v for k, v in data.get("sp_ratings", {}).items()}
        self.team_sp_avg = {int(k): v for k, v in data.get("team_sp_avg", {}).items()}
        self.games_played = data.get("games_played", 0)
        return True

    def save(self):
        """현재 레이팅 저장."""
        data = {
            "ratings": self.ratings,
            "sp_ratings": self.sp_ratings,
            "team_sp_avg": self.team_sp_avg,
            "games_played": self.games_played,
        }
        Path(self._ratings_path).write_text(
            json.dumps(data, indent=2, ensure_ascii=False)
        )

    def new_season(self):
        """시즌 초기화: 전 시즌 최종 Elo에서 1500 방향으로 1/3 회귀."""
        for code in self.ratings:
            prev = self.ratings[code]
            self.ratings[code] = prev + ELO_REVERT_FACTOR * (ELO_INITIAL - prev)
        self.games_played = 0
        self.save()

    # ── 확률 계산 ──

    @staticmethod
    def elo_to_prob(elo_diff: float) -> float:
        """Elo 차이를 승리 확률로 변환."""
        return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

    def get_k_factor(self) -> float:
        """시즌 진행도에 따른 K-factor."""
        if self.games_played < ELO_K_TRANSITION_GAME:
            # 선형 보간: K_EARLY → K_NORMAL
            t = self.games_played / ELO_K_TRANSITION_GAME
            return ELO_K_EARLY + t * (ELO_K_NORMAL - ELO_K_EARLY)
        return ELO_K_NORMAL

    def get_sp_adjustment(
        self, sp_no: int, team_code: int
    ) -> float:
        """선발 투수 Elo 보정값 계산.

        pitcher_adj = α × (SP_rating - team_avg_pitching_rating)
        """
        sp_rating = self.sp_ratings.get(sp_no, 0.0)
        team_avg = self.team_sp_avg.get(team_code, 0.0)
        return SP_ADJUSTMENT_ALPHA * (sp_rating - team_avg)

    def predict(
        self,
        home_team: int,
        away_team: int,
        home_sp: Optional[int] = None,
        away_sp: Optional[int] = None,
    ) -> float:
        """홈팀 승리 확률 예측.

        Returns:
            홈팀 승리 확률 (0.0 ~ 1.0)
        """
        home_elo = self.ratings.get(home_team, ELO_INITIAL)
        away_elo = self.ratings.get(away_team, ELO_INITIAL)

        # 홈 이점
        elo_diff = home_elo - away_elo + ELO_HOME_ADVANTAGE

        # 선발 투수 보정
        if home_sp is not None:
            elo_diff += self.get_sp_adjustment(home_sp, home_team)
        if away_sp is not None:
            elo_diff -= self.get_sp_adjustment(away_sp, away_team)

        return self.elo_to_prob(elo_diff)

    # ── Elo 갱신 ──

    @staticmethod
    def _adjusted_margin(score_diff: int) -> float:
        """승차를 비선형 스케일링. 대승 과대반영 방지.

        adj_margin = ((|diff| + 1)^0.7) × 1.41
        """
        return (abs(score_diff) + 1) ** ELO_MOV_EXPONENT * ELO_MOV_MULTIPLIER

    def update(
        self,
        home_team: int,
        away_team: int,
        home_score: int,
        away_score: int,
        home_sp: Optional[int] = None,
        away_sp: Optional[int] = None,
    ):
        """경기 결과로 Elo 갱신.

        shift = K × (outcome - win_prob) × (adj_margin / expected_margin)
        """
        # 예측 확률
        win_prob = self.predict(home_team, away_team, home_sp, away_sp)

        # 실제 결과 (홈 기준)
        if home_score > away_score:
            outcome = 1.0
        elif home_score < away_score:
            outcome = 0.0
        else:
            outcome = 0.5  # 무승부 (KBO에서는 거의 없지만 방어적 처리)

        # 승차 스케일링
        score_diff = home_score - away_score
        adj_margin = self._adjusted_margin(score_diff)
        # 기대 승차: Elo 차이에 비례 (0 차이 → 기대 승차 ~1.41)
        elo_diff = abs(
            self.ratings.get(home_team, ELO_INITIAL)
            - self.ratings.get(away_team, ELO_INITIAL)
        )
        expected_margin = self._adjusted_margin(
            round(elo_diff / 25)  # ~25 Elo points per run (경험적)
        )
        if expected_margin == 0:
            expected_margin = 1.0

        mov_multiplier = adj_margin / expected_margin

        # K-factor
        k = self.get_k_factor()

        # Elo 변동
        shift = k * (outcome - win_prob) * mov_multiplier

        self.ratings[home_team] = self.ratings.get(home_team, ELO_INITIAL) + shift
        self.ratings[away_team] = self.ratings.get(away_team, ELO_INITIAL) - shift

        self.games_played += 1

    # ── 선발 투수 레이팅 ──

    def update_sp_rating(
        self,
        p_no: int,
        team_code: int,
        fip: float,
        league_avg_fip: float = 4.20,
    ):
        """선발 투수의 개인 레이팅을 FIP 기반으로 산출.

        FIP가 낮을수록 좋으므로 역수 관계.
        sp_rating = (league_avg_fip - fip) × 10
        → FIP 3.20 (리그 평균 4.20) → rating = +10
        → FIP 5.20 → rating = -10
        """
        self.sp_ratings[p_no] = (league_avg_fip - fip) * 10.0

    def update_team_sp_avg(self, team_code: int, team_avg_fip: float,
                           league_avg_fip: float = 4.20):
        """팀 투수진 평균 레이팅 갱신."""
        self.team_sp_avg[team_code] = (league_avg_fip - team_avg_fip) * 10.0

    # ── 유틸리티 ──

    def get_all_ratings(self) -> dict[int, float]:
        """전체 팀 레이팅 반환 (정렬)."""
        return dict(sorted(self.ratings.items(), key=lambda x: -x[1]))

    def get_rating(self, team_code: int) -> float:
        return self.ratings.get(team_code, ELO_INITIAL)
