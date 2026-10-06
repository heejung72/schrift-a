"""
심화 분석 (5·6장 수상 포인트):
  [1] 확률 보정 (Calibration) — OOF 확률의 신뢰도 곡선 + isotonic 보정 전후 Brier
  [2] 불확실성 추정 (Uncertainty) — 앙상블 멤버(RF vs LGB) 예측 불일치 → 불확실성
       → 불확실성 기반 3단계 판정 재정당화
  [3] 분포 드리프트 (Drift) — labeled vs unlabeled PSI → 확장·운영 모니터링 근거

출력: outputs/13_calibration.png, 14_uncertainty.png, 15_drift.png,
      outputs/advanced_metrics.json
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

from src.features import load_labeled, load_unlabeled, make_features, SEED

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

N_SPLITS, N_REPEATS = 5, 3


def reliability(y, p, bins=8):
    edges = np.linspace(0, max(p.max(), 1e-6), bins + 1)
    xs, ys, ns = [], [], []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum() > 0:
            xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def psi(expected, actual, bins=10):
    edges = np.percentile(expected, np.linspace(0, 100, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def rf_lgb():
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                min_samples_leaf=2, n_jobs=-1, random_state=SEED)
    lg = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.03, num_leaves=31,
                            min_child_samples=3, scale_pos_weight=55,
                            reg_alpha=0.1, reg_lambda=1.0, feature_fraction=0.8,
                            bagging_fraction=0.8, bagging_freq=5,
                            n_jobs=-1, random_state=SEED, verbose=-1)
    return rf, lg


if __name__ == "__main__":
    X, y_s, g = make_features(load_labeled(), is_train=True)
    y = y_s.to_numpy()
    out = {}

    # ════════════════ [2] 불확실성 + OOF 멤버 확률 수집 (CV) ════════════════
    print("CV로 멤버별 OOF 확률 수집 중 (불확실성·보정용)...")
    oof_rf = np.zeros(len(y)); oof_lg = np.zeros(len(y)); cnt = np.zeros(len(y))
    for rep in range(N_REPEATS):
        sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + rep)
        for tr, va in sgkf.split(X, y, groups=g):
            rf, lg = rf_lgb()
            rf.fit(X.iloc[tr], y[tr]); lg.fit(X.iloc[tr], y[tr])
            oof_rf[va] += rf.predict_proba(X.iloc[va])[:, 1]
            oof_lg[va] += lg.predict_proba(X.iloc[va])[:, 1]
            cnt[va] += 1
    oof_rf /= np.maximum(cnt, 1); oof_lg /= np.maximum(cnt, 1)
    p_ens = (oof_rf + oof_lg) / 2
    uncert = np.abs(oof_rf - oof_lg)   # 멤버 불일치 = 불확실성

    # 불확실성 사분위별 불량률
    q = np.quantile(uncert, [0.25, 0.5, 0.75])
    bands = np.digitize(uncert, q)
    unc_rows = []
    for b in range(4):
        m = bands == b
        unc_rows.append({"band": ["Q1(낮음)", "Q2", "Q3", "Q4(높음)"][b],
                         "n": int(m.sum()), "defect_rate": round(float(y[m].mean()), 4),
                         "defects": int(y[m].sum())})
    out["uncertainty"] = {"quartiles": unc_rows,
                          "corr_defect": round(float(np.corrcoef(uncert, y)[0, 1]), 4)}
    print("  불확실성 사분위별 불량률:",
          " / ".join(f"{r['band']} {r['defect_rate']*100:.1f}%" for r in unc_rows))

    # 불확실성 기반 3단계: 高확률→즉시, (中확률 or 高불확실)→재검사, else 통과
    hp = np.quantile(p_ens, 0.95); lp = np.quantile(p_ens, 0.80); hu = np.quantile(uncert, 0.80)
    tier = np.zeros(len(y), int)
    tier[(p_ens >= lp) | (uncert >= hu)] = 1
    tier[p_ens >= hp] = 2
    tier_rows = []
    for t, name in [(2, "즉시검사"), (1, "재검사"), (0, "통과")]:
        m = tier == t
        tier_rows.append({"tier": t, "name": name, "n": int(m.sum()),
                          "defects": int(y[m].sum()),
                          "recall": round(float(y[m].sum() / y.sum()), 4)})
    insp = tier >= 1
    out["uncertainty_tier"] = {"tiers": tier_rows,
                               "inspect_frac": round(float(insp.mean()), 4),
                               "inspect_recall": round(float(y[insp].sum() / y.sum()), 4)}
    print(f"  불확실성 3단계: 검사량 {insp.mean()*100:.0f}% → 불량 {y[insp].sum()/y.sum()*100:.0f}% 포착")

    # ════════════════ [1] 확률 보정 (isotonic, 2-fold 교차적용으로 정직하게) ════════════════
    print("확률 보정(isotonic) 평가 중...")
    brier_raw = brier_score_loss(y, p_ens)
    cal = np.zeros(len(y))
    sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=SEED)
    for tr, va in sgkf.split(p_ens, y, groups=g):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(p_ens[tr], y[tr])
        cal[va] = ir.predict(p_ens[va])
    brier_cal = brier_score_loss(y, cal)
    out["calibration"] = {"brier_raw": round(float(brier_raw), 5),
                          "brier_isotonic": round(float(brier_cal), 5),
                          "improve_pct": round(float((brier_raw - brier_cal) / brier_raw * 100), 1)}
    print(f"  Brier: 원본 {brier_raw:.4f} → isotonic {brier_cal:.4f} "
          f"({out['calibration']['improve_pct']:+.0f}%)")

    # ════════════════ [3] 분포 드리프트 (labeled vs unlabeled PSI) ════════════════
    print("분포 드리프트(PSI) 계산 중...")
    Xu, _, _ = make_features(load_unlabeled(), is_train=False)
    feats = ["ewma_resid_norm", "Plasticizing_Time_delta", "Clamp_Close_Time",
             "barrel_temp_range", "Cycle_Time", "Max_Injection_Pressure",
             "Barrel_Temperature_5", "Cushion_Position"]
    psi_rows = [{"feat": f, "psi": round(psi(X[f].to_numpy(), Xu[f].to_numpy()), 3)}
                for f in feats if f in Xu.columns]
    psi_rows.sort(key=lambda r: -r["psi"])
    out["drift"] = {"features": psi_rows,
                    "note": "PSI<0.1 안정, 0.1~0.25 보통, >0.25 유의한 드리프트"}
    print("  PSI 상위:", " / ".join(f"{r['feat']} {r['psi']}" for r in psi_rows[:3]))

    # ════════════════ 그래프 ════════════════
    # 13 보정
    fig, ax = plt.subplots(figsize=(6, 5))
    xs, ys, ns = reliability(y, p_ens)
    xc, yc, _ = reliability(y, cal)
    ax.plot([0, max(xs.max(), 0.3)], [0, max(xs.max(), 0.3)], "--", color="gray", label="완벽 보정")
    ax.plot(xs, ys, "o-", color="#1976D2", label=f"원본 (Brier {brier_raw:.4f})")
    ax.plot(xc, yc, "s-", color="#E53935", label=f"isotonic (Brier {brier_cal:.4f})")
    ax.set_xlabel("예측 확률"); ax.set_ylabel("실제 불량률"); ax.set_title("확률 신뢰도 곡선 (Reliability)")
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("outputs/13_calibration.png", dpi=150, bbox_inches="tight"); plt.close()

    # 14 불확실성
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = [r["band"] for r in unc_rows]; rates = [r["defect_rate"] * 100 for r in unc_rows]
    ax.bar(labels, rates, color=["#4CAF50", "#FFC107", "#FF9800", "#E53935"])
    ax.axhline(y.mean() * 100, ls="--", color="gray", label=f"기준 불량률 {y.mean()*100:.1f}%")
    ax.set_ylabel("불량률 (%)"); ax.set_title("앙상블 불확실성(RF↔LGB 불일치) 사분위별 불량률")
    ax.legend(); ax.grid(alpha=0.3, axis="y"); plt.tight_layout()
    plt.savefig("outputs/14_uncertainty.png", dpi=150, bbox_inches="tight"); plt.close()

    # 15 드리프트
    fig, ax = plt.subplots(figsize=(7, 5))
    fs = [r["feat"] for r in psi_rows][::-1]; ps = [r["psi"] for r in psi_rows][::-1]
    colors = ["#E53935" if v > 0.25 else "#FF9800" if v > 0.1 else "#4CAF50" for v in ps]
    ax.barh(fs, ps, color=colors)
    ax.axvline(0.1, ls="--", color="gray", lw=0.8); ax.axvline(0.25, ls="--", color="#E53935", lw=0.8)
    ax.set_xlabel("PSI (labeled→unlabeled)"); ax.set_title("분포 드리프트 (운영 모니터링 지표)")
    ax.grid(alpha=0.3, axis="x"); plt.tight_layout()
    plt.savefig("outputs/15_drift.png", dpi=150, bbox_inches="tight"); plt.close()

    with open("outputs/advanced_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n저장: outputs/advanced_metrics.json + 13·14·15 그래프")
