"""
v8 Sequential Experiment — v65+ 기반 순차 개선 측정.

Stage 0: Control (v65+ baseline at t=0.5)
Stage 1: Threshold optimization
Stage 2: Sample weight (year-based)
Stage 3: Dead zone Elo push
Stage 4: Hyperparameter re-tuning (Optuna 100 trials)
Stage 5: Final combined
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

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "saved"

# v65+ 파라미터 (현재 프로덕션)
PARAMS_V65_PLUS = {
    "max_depth": 9, "learning_rate": 0.0093, "n_estimators": 588,
    "subsample": 0.54, "colsample_bytree": 0.30, "min_child_weight": 11,
    "reg_lambda": 1.26, "reg_alpha": 0.28, "gamma": 0.77,
    "objective": "binary:logistic", "eval_metric": "logloss",
    "random_state": 42, "verbosity": 0,
}

WEIGHT_SCHEMES = {
    "A": {2023: 0.5, 2024: 0.8, 2025: 1.0, 2026: 1.5},
    "B": {2023: 0.7, 2024: 0.9, 2025: 1.0, 2026: 1.2},
    "C": {2023: 1.0, 2024: 1.0, 2025: 1.0, 2026: 2.0},
}

DEADZONE_PUSHES = [0.01, 0.02, 0.03]


def _infer_year_labels(n_rows: int) -> np.ndarray:
    """스케줄 파일 기반 연도 레이블 추정."""
    year_counts = []
    for yr in [2023, 2024, 2025, 2026]:
        sched_file = DATA_DIR / f"schedules_{yr}.json"
        if not sched_file.exists():
            continue
        rows = json.loads(sched_file.read_text())
        completed = [
            g for g in rows
            if g.get("leagueType") == 10100
            and g.get("homeScore") is not None
            and g.get("awayScore") is not None
        ]
        if completed:
            year_counts.append((yr, len(completed)))

    total = sum(c for _, c in year_counts)
    labels = []
    for i, (yr, count) in enumerate(year_counts):
        if i < len(year_counts) - 1:
            n = round(n_rows * count / total)
        else:
            n = n_rows - len(labels)
        labels.extend([yr] * max(n, 0))

    # 길이 보정
    while len(labels) < n_rows:
        labels.append(labels[-1] if labels else 2025)
    return np.array(labels[:n_rows])


def _find_optimal_threshold(y_true, y_prob):
    """Validation fold에서 최적 threshold 탐색 (0.45~0.55, step=0.005)."""
    best_t, best_acc = 0.5, accuracy_score(y_true, (y_prob >= 0.5).astype(int))
    for t in np.arange(0.45, 0.551, 0.005):
        acc = accuracy_score(y_true, (y_prob >= t).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_t = round(float(t), 3)
    return best_t, best_acc


def _apply_deadzone_push(prob, elo_diff_vals, push_strength):
    """Dead zone (0.48~0.52) 구간에 Elo 방향 push 적용."""
    prob = prob.copy()
    mask = (prob >= 0.48) & (prob <= 0.52)
    prob[mask] += push_strength * np.sign(elo_diff_vals[mask])
    return np.clip(prob, 0.001, 0.999)


def _cv_evaluate(X, y, params, n_splits=5, sample_weights=None,
                 deadzone_push=0.0, elo_diff_idx=None):
    """범용 CV 평가. threshold tuning 포함."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        fit_kwargs = {"eval_set": [(X_val, y_val)], "verbose": False}
        if sample_weights is not None:
            fit_kwargs["sample_weight"] = sample_weights[train_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, **fit_kwargs)
        prob = model.predict_proba(X_val)[:, 1]

        # Dead zone push
        if deadzone_push > 0 and elo_diff_idx is not None:
            prob = _apply_deadzone_push(prob, X_val[:, elo_diff_idx], deadzone_push)

        # Default accuracy (t=0.5)
        acc_default = accuracy_score(y_val, (prob >= 0.5).astype(int))

        # Threshold tuning
        best_t, acc_tuned = _find_optimal_threshold(y_val, prob)

        ll = log_loss(y_val, np.clip(prob, 1e-7, 1 - 1e-7))

        # Dead zone stats
        dz_mask = (prob >= 0.48) & (prob <= 0.52)
        dz_count = int(dz_mask.sum())
        dz_acc = float(accuracy_score(y_val[dz_mask], (prob[dz_mask] >= 0.5).astype(int))) if dz_count > 0 else 0.0

        results.append({
            "fold": fold + 1,
            "accuracy": round(acc_default, 4),
            "accuracy_tuned": round(acc_tuned, 4),
            "threshold": best_t,
            "logloss": round(ll, 4),
            "deadzone_count": dz_count,
            "deadzone_accuracy": round(dz_acc, 4),
        })

    return {
        "mean_accuracy": round(float(np.mean([r["accuracy"] for r in results])), 4),
        "mean_accuracy_tuned": round(float(np.mean([r["accuracy_tuned"] for r in results])), 4),
        "mean_threshold": round(float(np.mean([r["threshold"] for r in results])), 3),
        "mean_logloss": round(float(np.mean([r["logloss"] for r in results])), 4),
        "mean_dz_count": round(float(np.mean([r["deadzone_count"] for r in results])), 1),
        "mean_dz_accuracy": round(float(np.mean([r["deadzone_accuracy"] for r in results])), 4),
        "folds": results,
    }


# ── Stage Implementations ──

def stage0_control(X, y):
    """Stage 0: v65+ baseline (현재 프로덕션 파라미터)."""
    logger.info("=" * 60)
    logger.info("Stage 0: Control (v65+ baseline)")
    logger.info("=" * 60)
    result = _cv_evaluate(X, y, PARAMS_V65_PLUS)
    logger.info("  acc=%.4f  tuned=%.4f  t=%.3f  logloss=%.4f  dz_acc=%.4f (%d games)",
                result["mean_accuracy"], result["mean_accuracy_tuned"],
                result["mean_threshold"], result["mean_logloss"],
                result["mean_dz_accuracy"], result["mean_dz_count"])
    return result


def stage1_threshold(result_s0):
    """Stage 1: Threshold 최적화 (Stage 0 결과에서 threshold 추출)."""
    logger.info("=" * 60)
    logger.info("Stage 1: Threshold Optimization")
    logger.info("=" * 60)
    # Stage 1 = Stage 0과 동일 모델, threshold tuning이 핵심 산출물
    optimal_t = result_s0["mean_threshold"]
    delta = (result_s0["mean_accuracy_tuned"] - result_s0["mean_accuracy"]) * 100
    logger.info("  Optimal threshold: %.3f", optimal_t)
    logger.info("  Threshold improvement: +%.2f%%p (%.4f → %.4f)",
                delta, result_s0["mean_accuracy"], result_s0["mean_accuracy_tuned"])
    return result_s0, optimal_t


def stage2_sample_weight(X, y, year_labels, baseline_tuned):
    """Stage 2: Sample weight (3개 scheme 비교)."""
    logger.info("=" * 60)
    logger.info("Stage 2: Sample Weight")
    logger.info("=" * 60)

    best_result = None
    best_scheme = None
    best_tuned = baseline_tuned

    for name, weights in WEIGHT_SCHEMES.items():
        sw = np.array([weights.get(int(yr), 1.0) for yr in year_labels], dtype=np.float64)
        result = _cv_evaluate(X, y, PARAMS_V65_PLUS, sample_weights=sw)
        delta = (result["mean_accuracy_tuned"] - baseline_tuned) * 100
        logger.info("  Scheme %s: acc=%.4f  tuned=%.4f  t=%.3f  delta=%+.2f%%p",
                    name, result["mean_accuracy"], result["mean_accuracy_tuned"],
                    result["mean_threshold"], delta)
        if result["mean_accuracy_tuned"] > best_tuned:
            best_tuned = result["mean_accuracy_tuned"]
            best_result = result
            best_scheme = name

    if best_scheme is None:
        logger.info("  → Sample weight 미적용 (개선 없음)")
        return None, None, None
    else:
        logger.info("  → Best scheme: %s (tuned=%.4f, delta=+%.2f%%p)",
                    best_scheme, best_tuned, (best_tuned - baseline_tuned) * 100)
        return best_result, best_scheme, WEIGHT_SCHEMES[best_scheme]


def stage3_deadzone(X, y, sample_weights, elo_diff_idx, baseline_tuned):
    """Stage 3: Dead zone Elo-directed push."""
    logger.info("=" * 60)
    logger.info("Stage 3: Dead Zone Push")
    logger.info("=" * 60)

    best_result = None
    best_push = 0.0
    best_tuned = baseline_tuned

    for push in DEADZONE_PUSHES:
        result = _cv_evaluate(X, y, PARAMS_V65_PLUS, sample_weights=sample_weights,
                              deadzone_push=push, elo_diff_idx=elo_diff_idx)
        delta = (result["mean_accuracy_tuned"] - baseline_tuned) * 100
        logger.info("  Push %.0f%%p: acc=%.4f  tuned=%.4f  t=%.3f  dz_acc=%.4f  delta=%+.2f%%p",
                    push * 100, result["mean_accuracy"], result["mean_accuracy_tuned"],
                    result["mean_threshold"], result["mean_dz_accuracy"], delta)
        if result["mean_accuracy_tuned"] > best_tuned:
            best_tuned = result["mean_accuracy_tuned"]
            best_result = result
            best_push = push

    if best_push == 0.0:
        logger.info("  → Dead zone push 미적용 (개선 없음)")
        return None, 0.0
    else:
        logger.info("  → Best push: %.0f%%p (tuned=%.4f, delta=+%.2f%%p)",
                    best_push * 100, best_tuned, (best_tuned - baseline_tuned) * 100)
        return best_result, best_push


def stage4_hypertune(X, y, sample_weights, deadzone_push, elo_diff_idx,
                     baseline_tuned, n_trials=100):
    """Stage 4: Optuna hyperparameter re-tuning."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    logger.info("=" * 60)
    logger.info("Stage 4: Hyperparameter Tuning (Optuna %d trials)", n_trials)
    logger.info("=" * 60)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
            "subsample": trial.suggest_float("subsample", 0.5, 0.8),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.5),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 20),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 1.0),
            "gamma": trial.suggest_float("gamma", 0.3, 2.0),
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
            "verbosity": 0,
        }
        result = _cv_evaluate(X, y, params, sample_weights=sample_weights,
                              deadzone_push=deadzone_push, elo_diff_idx=elo_diff_idx)
        return result["mean_accuracy_tuned"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = {
        **study.best_params,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "verbosity": 0,
    }

    # 최적 파라미터로 전체 결과 재평가
    result = _cv_evaluate(X, y, best_params, sample_weights=sample_weights,
                          deadzone_push=deadzone_push, elo_diff_idx=elo_diff_idx)

    delta = (result["mean_accuracy_tuned"] - baseline_tuned) * 100
    tunable = {k: v for k, v in best_params.items()
               if k not in ("objective", "eval_metric", "random_state", "verbosity")}
    logger.info("  Best params: %s", tunable)
    logger.info("  tuned=%.4f  delta=%+.2f%%p", result["mean_accuracy_tuned"], delta)

    return result, best_params


def run():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # ── 데이터 로드 ──
    for name in ["training_data_v65plus_live", "training_data_v65plus"]:
        path = DATA_DIR / f"{name}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            logger.info("데이터 로드: %s (%d행)", path.name, len(df))
            break
    else:
        logger.error("v65+ 학습 데이터 없음")
        return

    feat_cols = [c for c in df.columns if c != "label"]
    X = df[feat_cols].values
    y = df["label"].values

    # elo_diff 컬럼 인덱스 탐색
    elo_diff_idx = None
    for i, name in enumerate(feat_cols):
        if name == "elo_diff":
            elo_diff_idx = i
            break
    if elo_diff_idx is None:
        # v65+ 비교 피처 7번째 = index 225 (v65 219feat + 6)
        elo_diff_idx = min(225, X.shape[1] - 1)
        logger.warning("elo_diff 컬럼명 없음 — 인덱스 %d 사용", elo_diff_idx)

    logger.info("데이터: %d행 × %d피처  elo_diff_idx=%d", X.shape[0], X.shape[1], elo_diff_idx)

    # 연도 레이블
    year_labels = _infer_year_labels(X.shape[0])
    unique, counts = np.unique(year_labels, return_counts=True)
    logger.info("연도별 행 수: %s", dict(zip(unique.tolist(), counts.tolist())))

    all_results = {}

    # ── Stage 0: Control ──
    r0 = stage0_control(X, y)
    all_results["stage0"] = r0
    baseline_acc = r0["mean_accuracy"]
    baseline_tuned = r0["mean_accuracy_tuned"]

    # ── Stage 1: Threshold ──
    r1, optimal_threshold = stage1_threshold(r0)
    all_results["stage1"] = {"result": r1, "optimal_threshold": optimal_threshold}

    # 이후 stage의 기준선 = Stage 0의 tuned (threshold 포함)
    running_best = baseline_tuned

    # ── Stage 2: Sample Weight ──
    r2, best_scheme, best_weights = stage2_sample_weight(X, y, year_labels, running_best)
    if r2 is not None:
        running_best = r2["mean_accuracy_tuned"]
        sw = np.array([best_weights.get(int(yr), 1.0) for yr in year_labels], dtype=np.float64)
        all_results["stage2"] = {**r2, "scheme": best_scheme, "weights": best_weights}
    else:
        sw = None
        r2 = r1  # 개선 없으면 이전 단계 결과 유지
        all_results["stage2"] = {**r1, "scheme": None, "weights": None, "note": "no improvement"}

    # ── Stage 3: Dead Zone ──
    r3, best_push = stage3_deadzone(X, y, sw, elo_diff_idx, running_best)
    if r3 is not None:
        running_best = r3["mean_accuracy_tuned"]
        all_results["stage3"] = {**r3, "push": best_push}
    else:
        best_push = 0.0
        r3_display = all_results.get("stage2", r1)
        all_results["stage3"] = {**r3_display, "push": 0.0, "note": "no improvement"}

    # ── Stage 4: Hyperparameter Tuning ──
    r4, best_params = stage4_hypertune(X, y, sw, best_push, elo_diff_idx, running_best)
    if r4["mean_accuracy_tuned"] > running_best:
        running_best = r4["mean_accuracy_tuned"]
        use_new_params = True
    else:
        use_new_params = False
    all_results["stage4"] = {**r4, "best_params": best_params, "adopted": use_new_params}

    # ── Stage 5: Final Combined ──
    logger.info("=" * 60)
    logger.info("Stage 5: Final Combined")
    logger.info("=" * 60)

    final_params = best_params if use_new_params else PARAMS_V65_PLUS
    r5 = _cv_evaluate(X, y, final_params, sample_weights=sw,
                      deadzone_push=best_push, elo_diff_idx=elo_diff_idx)
    all_results["stage5"] = r5

    # ── 결과 출력 ──
    stages = [
        ("Stage 0 (Control)", r0, baseline_acc),
        ("Stage 1 (Threshold)", r1, baseline_acc),
        ("Stage 2 (SampleWt)", all_results["stage2"], baseline_tuned),
        ("Stage 3 (DeadZone)", all_results["stage3"], baseline_tuned),
        ("Stage 4 (HyperTune)", r4, baseline_tuned),
        ("Stage 5 (Combined)", r5, baseline_tuned),
    ]

    print("\n" + "=" * 80)
    print("v8 Sequential Experiment Results")
    print("=" * 80)
    print(f"{'Stage':<25} {'acc':>7} {'tuned':>7} {'t':>6} {'logloss':>8} {'delta':>8}")
    print("-" * 80)

    for label, r, ref in stages:
        acc = r.get("mean_accuracy", r.get("result", {}).get("mean_accuracy", 0))
        tuned = r.get("mean_accuracy_tuned", r.get("result", {}).get("mean_accuracy_tuned", 0))
        t = r.get("mean_threshold", r.get("result", {}).get("mean_threshold", 0))
        ll = r.get("mean_logloss", r.get("result", {}).get("mean_logloss", 0))
        delta = (tuned - ref) * 100
        sign = "+" if delta >= 0 else ""
        print(f"{label:<25} {acc:>7.4f} {tuned:>7.4f} {t:>6.3f} {ll:>8.4f} {sign}{delta:>6.2f}%p")

    print("=" * 80)

    # ── 승격 판단 ──
    final_tuned = r5["mean_accuracy_tuned"]
    total_delta = (final_tuned - baseline_tuned) * 100
    total_delta_from_default = (final_tuned - baseline_acc) * 100

    print(f"\nBaseline (t=0.5):  {baseline_acc:.4f}")
    print(f"Baseline (tuned):  {baseline_tuned:.4f}")
    print(f"Final v8 (tuned):  {final_tuned:.4f}")
    print(f"Delta vs tuned:    {'+' if total_delta >= 0 else ''}{total_delta:.2f}%p")
    print(f"Delta vs default:  {'+' if total_delta_from_default >= 0 else ''}{total_delta_from_default:.2f}%p")

    if total_delta_from_default >= 1.5:
        print(f"\nv8 승격 기준 충족: +{total_delta_from_default:.2f}%p (>= 1.5%p from default)")
    elif total_delta >= 1.5:
        print(f"\nv8 승격 기준 충족: +{total_delta:.2f}%p (>= 1.5%p from tuned baseline)")
    else:
        print(f"\nv8 승격 기준 미달: +{total_delta:.2f}%p (< 1.5%p)")

    # ── v8 config 저장 ──
    v8_config = {
        "params": final_params,
        "threshold": float(r5["mean_threshold"]),
        "sample_weights": best_weights if sw is not None else None,
        "deadzone_push": float(best_push),
        "elo_diff_feature_idx": int(elo_diff_idx),
        "feature_count": int(X.shape[1]),
        "baseline_accuracy": float(baseline_acc),
        "baseline_tuned": float(baseline_tuned),
        "v8_accuracy": float(r5["mean_accuracy"]),
        "v8_tuned": float(final_tuned),
        "delta_from_default": round(float(total_delta_from_default), 2),
        "delta_from_tuned": round(float(total_delta), 2),
        "new_params_adopted": use_new_params,
    }

    config_path = RESULTS_DIR / "v8_config.json"
    config_path.write_text(json.dumps(v8_config, indent=2, ensure_ascii=False, default=_json_convert))
    logger.info("v8 config 저장: %s", config_path)

    # 프로덕션 배포용 config 복사
    prod_config_path = MODEL_DIR / "v8_config.json"
    prod_config_path.write_text(json.dumps(v8_config, indent=2, ensure_ascii=False, default=_json_convert))
    logger.info("프로덕션 config 저장: %s", prod_config_path)

    # 전체 결과 저장
    out_path = RESULTS_DIR / "v8_experiment_results.json"
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False, default=_json_convert))
    logger.info("전체 결과 저장: %s", out_path)

    return all_results


def _json_convert(obj):
    """numpy 타입 JSON 직렬화."""
    if isinstance(obj, (np.int64, np.int32, np.int_)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32, np.floating)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == "__main__":
    run()
