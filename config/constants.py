"""
고정 상수 및 팀/구장 코드 매핑.
출처가 명시된 상수는 MLB 공개 야구 분석학 표준 공식에서 참조.
"""

# ── Elo 파라미터 ──
ELO_INITIAL = 1500
ELO_K_EARLY = 6          # 시즌 초반 (~30경기) 빠른 적응
ELO_K_NORMAL = 4          # 시즌 중반 이후 안정화 (FiveThirtyEight MLB 기준)
ELO_K_TRANSITION_GAME = 30  # K-factor 전환 기준 경기 수
ELO_HOME_ADVANTAGE = 30   # KBO 홈 이점 (≈55.2%). MLB 24pt 대비 높게 설정
ELO_REVERT_FACTOR = 1 / 3  # 시즌 초기화 시 평균 회귀 비율
ELO_MOV_EXPONENT = 0.7    # 승차(Margin of Victory) 스케일링 지수
ELO_MOV_MULTIPLIER = 1.41  # 승차 스케일링 승수

# ── 선발 투수 보정 ──
SP_ADJUSTMENT_ALPHA = 4.7  # FiveThirtyEight 선발 투수 보정 계수

# ── 피타고리안 기대승률 ──
PYTHAGOREAN_EXPONENT = 1.83  # Bill James 공식. KBO 1.80~1.85 범위

# ── FIP 상수 (시즌 리그 평균으로 재산출 필요) ──
FIP_HR_WEIGHT = 13
FIP_BB_WEIGHT = 3
FIP_K_WEIGHT = 2
FIP_CONSTANT_DEFAULT = 3.10  # 초기값. 리그 평균으로 보정 예정

# ── Rolling 윈도우 ──
ROLLING_WINDOW_RECENT = 10  # 최근 폼 계산용 경기 수
BULLPEN_LOAD_DAYS = 3       # 불펜 피로도 측정 기간 (일)

# ── 블렌딩 가중치 (시즌 진행도별) ──
# (경기 수 기준, Elo 가중치)
BLEND_SCHEDULE = [
    (30, 0.80),   # 개막~30경기: Elo 80%
    (80, 0.60),   # 31~80경기: Elo 60%
    (120, 0.45),  # 81~120경기: Elo 45%
    (999, 0.40),  # 121경기~: Elo 40%
]

# ── 콜드 스타트: Prior 블렌딩 ──
# (경기 수 기준, 전시즌 Prior 비중)
PRIOR_SCHEDULE = [
    (20, 0.70),
    (50, 0.50),
    (80, 0.30),
    (999, 0.20),
]

# ── 제출 규칙 ──
SUBMIT_MIN_PROB = 0.01     # 최소 제출 확률
SUBMIT_MAX_PROB = 99.99    # 최대 제출 확률
SUBMIT_FORBIDDEN = 50.00   # 금지 확률값
SUBMIT_DECIMAL_PLACES = 2  # 소수점 둘째 자리

# ── KBO 팀 코드 (statiz API) ──
TEAM_CODES = {
    1001: '삼성',
    2002: 'KIA',
    3001: '롯데',
    5002: 'LG',
    6002: '두산',
    7002: '한화',
    9002: 'SSG',
    10001: '키움',
    11001: 'NC',
    12001: 'KT',
}

TEAM_CODE_BY_NAME = {v: k for k, v in TEAM_CODES.items()}

# ── 리그 타입 ──
LEAGUE_REGULAR = 10100
LEAGUE_POSTSEASON = 10200

# ── 야간 경기 기준 ──
NIGHT_GAME_HOUR = 18  # 18시 이후 = 야간
