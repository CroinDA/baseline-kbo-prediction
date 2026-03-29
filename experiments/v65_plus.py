"""
v65+ 실험 — v65 기반 개선 3가지 전략 비교.

전략 A: v65 219피처 + 비교 피처 10개 추가 (229피처)
전략 B: v65 + 데드존 캘리브레이션 (48-52% 구간 Elo 대체)
전략 C: A + B 결합
전략 D: v65 + isotonic calibration
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PARAMS_V65 = {
    "max_depth": 9, "learning_rate": 0.0093, "n_estimators": 588,
    "subsample": 0.54, "colsample_bytree": 0.33, "min_child_weight": 11,
    "reg_lambda": 1.26, "reg_alpha": 0.28, "gamma": 0.77,
    "objective": "binary:logistic", "eval_metric": "logloss",
    "random_state": 42, "verbosity": 0,
}

# v65 + 비교 피처 전용 파라미터 (피처 수 증가 반영)
PARAMS_V65_PLUS = {
    **PARAMS_V65,
    "colsample_bytree": 0.30,  # 피처 늘어난 만큼 약간 낮춤
}


def _add_comparison_features(df: pd.DataFrame) -> pd.DataFrame:
    """v65 DataFrame에 비교/집계 피처 추가."""
    df = df.copy()

    # 1. 라인업 wRC+ 평균 (홈/원정)
    batter_wrc_cols_home = [f"home_b{i}_wrcplus" for i in range(1, 10)]
    batter_wrc_cols_away = [f"away_b{i}_wrcplus" for i in range(1, 10)]
    df["home_lineup_mean_wrc"] = df[batter_wrc_cols_home].mean(axis=1)
    df["away_lineup_mean_wrc"] = df[batter_wrc_cols_away].mean(axis=1)
    df["lineup_wrc_diff"] = df["home_lineup_mean_wrc"] - df["away_lineup_mean_wrc"]

    # 2. 라인업 Top3 wRC+
    home_wrc = df[batter_wrc_cols_home].values
    away_wrc = df[batter_wrc_cols_away].values
    df["home_lineup_top3_wrc"] = np.sort(home_wrc, axis=1)[:, -3:].mean(axis=1)
    df["away_lineup_top3_wrc"] = np.sort(away_wrc, axis=1)[:, -3:].mean(axis=1)

    # 3. SP FIP diff
    df["sp_fip_diff"] = df["away_sp_fip"] - df["home_sp_fip"]

    # 4. Elo diff (홈Elo - 원정Elo)
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    # 5. Log5
    home_wpct = df["home_wpct"].clip(0.01, 0.99)
    away_wpct = df["away_wpct"].clip(0.01, 0.99)
    df["log5_home"] = (home_wpct * (1 - away_wpct)) / (home_wpct * (1 - away_wpct) + away_wpct * (1 - home_wpct))

    # 6. 팀 강도 차이
    df["pyth_diff"] = df["home_pyth_exp"] - df["away_pyth_exp"]
    df["wpct_diff"] = df["home_wpct"] - df["away_wpct"]

    return df


def _cv_evaluate(X, y, params, n_splits=5, label="", deadzone_fix=False, calibrate=False):
    """CV 평가. deadzone_fix=True면 48-52% 구간 반전, calibrate=True면 isotonic."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        prob = model.predict_proba(X_val)[:, 1]

        if calibrate:
            # Isotonic calibration on train set
            train_prob = model.predict_proba(X_train)[:, 1]
            iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            iso.fit(train_prob, y_train)
            prob = iso.predict(prob)

        if deadzone_fix:
            # 48-52% 구간: 반전 (50기준으로 mirror)
            mask = (prob >= 0.48) & (prob < 0.52)
            prob[mask] = 1.0 - prob[mask]

        pred = (prob > 0.5).astype(int)
        acc = accuracy_score(y_val, pred)

        # threshold tuning
        best_t, best_acc = 0.5, acc
        for t in np.arange(0.45, 0.551, 0.005):
            a = accuracy_score(y_val, (prob >= t).astype(int))
            if a > best_acc:
                best_acc = a
                best_t = t

        results.append({
            "fold": fold + 1,
            "accuracy": round(acc, 4),
            "accuracy_tuned": round(best_acc, 4),
            "threshold": round(best_t, 3),
            "logloss": round(log_loss(y_val, np.clip(prob, 1e-7, 1 - 1e-7)), 4),
        })

    mean_acc = np.mean([r["accuracy"] for r in results])
    mean_tuned = np.mean([r["accuracy_tuned"] for r in results])
    mean_ll = np.mean([r["logloss"] for r in results])
    avg_t = np.mean([r["threshold"] for r in results])

    logger.info("[%s] acc=%.4f, tuned=%.4f (t=%.3f), logloss=%.4f", label, mean_acc, mean_tuned, avg_t, mean_ll)
    return {
        "label": label,
        "mean_accuracy": round(mean_acc, 4),
        "mean_accuracy_tuned": round(mean_tuned, 4),
        "mean_threshold": round(avg_t, 3),
        "mean_logloss": round(mean_ll, 4),
        "folds": results,
    }


def run():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    df = pd.read_parquet(DATA_DIR / "training_data_v65_live.parquet")
    feat_cols_v65 = [c for c in df.columns if c != "label"]
    X_v65 = df[feat_cols_v65].values
    y = df["label"].values

    all_results = []

    # ── Control: v65 baseline ──
    r = _cv_evaluate(X_v65, y, PARAMS_V65, label="v65-baseline")
    r["features"] = X_v65.shape[1]
    all_results.append(r)

    # ── Strategy A: v65 + comparison features ──
    df_plus = _add_comparison_features(df)
    feat_cols_plus = [c for c in df_plus.columns if c != "label"]
    X_plus = df_plus[feat_cols_plus].values

    r = _cv_evaluate(X_plus, y, PARAMS_V65_PLUS, label="A: v65+compare")
    r["features"] = X_plus.shape[1]
    all_results.append(r)

    # ── Strategy B: v65 + deadzone fix ──
    r = _cv_evaluate(X_v65, y, PARAMS_V65, label="B: v65+deadzone", deadzone_fix=True)
    r["features"] = X_v65.shape[1]
    all_results.append(r)

    # ── Strategy C: A + B combined ──
    r = _cv_evaluate(X_plus, y, PARAMS_V65_PLUS, label="C: v65+compare+dz", deadzone_fix=True)
    r["features"] = X_plus.shape[1]
    all_results.append(r)

    # ── Strategy D: v65 + isotonic calibration ──
    r = _cv_evaluate(X_v65, y, PARAMS_V65, label="D: v65+isotonic", calibrate=True)
    r["features"] = X_v65.shape[1]
    all_results.append(r)

    # ── Strategy E: v65 + isotonic + deadzone ──
    r = _cv_evaluate(X_v65, y, PARAMS_V65, label="E: v65+iso+dz", calibrate=True, deadzone_fix=True)
    r["features"] = X_v65.shape[1]
    all_results.append(r)

    # ── Strategy F: v65+compare + isotonic ──
    r = _cv_evaluate(X_plus, y, PARAMS_V65_PLUS, label="F: v65+cmp+iso", calibrate=True)
    r["features"] = X_plus.shape[1]
    all_results.append(r)

    # ── Strategy G: 하이퍼파라미터 탐색 (v65 피처, 다른 세팅) ──
    alt_params = [
        ("G1: depth6", {**PARAMS_V65, "max_depth": 6, "n_estimators": 800, "learning_rate": 0.01}),
        ("G2: depth4", {**PARAMS_V65, "max_depth": 4, "n_estimators": 800, "learning_rate": 0.015}),
        ("G3: high-reg", {**PARAMS_V65, "reg_lambda": 3.0, "reg_alpha": 1.0, "gamma": 2.0, "min_child_weight": 20}),
        ("G4: wide-col", {**PARAMS_V65, "colsample_bytree": 0.5, "subsample": 0.7}),
    ]
    for label, params in alt_params:
        r = _cv_evaluate(X_v65, y, params, label=label)
        r["features"] = X_v65.shape[1]
        all_results.append(r)

    # ── 결과 출력 ──
    summary = []
    for r in all_results:
        summary.append({
            "label": r["label"],
            "features": r.get("features"),
            "accuracy": r["mean_accuracy"],
            "tuned": r.get("mean_accuracy_tuned"),
            "threshold": r.get("mean_threshold"),
            "logloss": r.get("mean_logloss"),
        })

    out_path = RESULTS_DIR / "v65_plus_results.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n" + "=" * 85)
    print(f"{'Strategy':<25} {'Feat':>5} {'Acc':>7} {'Tuned':>7} {'Thr':>6} {'LL':>7}")
    print("-" * 85)
    for s in summary:
        print(
            f"{s['label']:<25} {s.get('features', '?'):>5} "
            f"{s['accuracy']:>7.4f} {s.get('tuned', 0):>7.4f} "
            f"{s.get('threshold', '-'):>6} {s.get('logloss', '-'):>7}"
        )
    print("=" * 85)

    return summary


if __name__ == "__main__":
    run()
