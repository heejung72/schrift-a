"""
최종 앙상블 모델(RF+LGB)로 unlabeled 데이터 예측 및 검사 우선순위 곡선 생성
- 검사 우선순위 곡선: CV holdout 기반 (train.py에서 생성된 cv_predictions.csv 사용)
- 3단계 판정: 상위 5% = 즉시검사, 6~20% = 재검사, 나머지 = 통과 (percentile 기반)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.features import load_labeled, load_unlabeled, make_features, SEED

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def load_model(path="outputs/final_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def make_inspection_curve(y_true: np.ndarray, y_prob: np.ndarray,
                          title: str, save_path: str):
    """위험도 상위 k% 검사 시 불량 포착률 곡선 (CV holdout 기반)."""
    sorted_idx = np.argsort(y_prob)[::-1]
    n_total = len(y_true)
    n_fail = y_true.sum()

    fracs = np.arange(1, n_total + 1) / n_total
    cum_fails = y_true[sorted_idx].cumsum()
    recall_at_k = cum_fails / n_fail

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fracs * 100, recall_at_k * 100, color="#1976D2", lw=2, label="모델 우선순위 (CV holdout)")
    ax.plot([0, 100], [0, 100], "--", color="gray", label="무작위 검사 (기준선)")
    ax.fill_between(fracs * 100, recall_at_k * 100, fracs * 100, alpha=0.2, color="#1976D2")

    for k in [10, 20, 30]:
        idx = int(n_total * k / 100) - 1
        cap = recall_at_k[idx] * 100
        ax.scatter(k, cap, zorder=5, s=60, color="red")
        ax.annotate(f"상위 {k}%→불량 {cap:.0f}%", xy=(k, cap),
                    xytext=(k + 2, cap - 8), fontsize=9)

    ax.set_xlabel("검사 비율 (%)")
    ax.set_ylabel("불량 포착률 (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"그래프 저장: {save_path}")

    stats = {}
    for k in [10, 20, 30]:
        idx = int(n_total * k / 100) - 1
        cap = recall_at_k[idx] * 100
        stats[k] = cap
        print(f"  상위 {k}% 검사 → 불량 {cap:.0f}% 포착 (무작위 대비 {cap/k:.1f}배)")
    return stats, recall_at_k, fracs


def three_tier_decision_percentile(y_prob: np.ndarray) -> np.ndarray:
    """Percentile 기반 3단계 판정.
    - 상위 5%  → 즉시검사 (tier=2)
    - 6~20%   → 재검사   (tier=1)
    - 나머지   → 통과     (tier=0)
    """
    hi_thresh = np.percentile(y_prob, 95)
    lo_thresh = np.percentile(y_prob, 80)
    tier = np.zeros(len(y_prob), dtype=int)
    tier[y_prob >= lo_thresh] = 1
    tier[y_prob >= hi_thresh] = 2
    return tier


if __name__ == "__main__":
    print("데이터 로드 중...")
    lab = load_labeled()
    unlab = load_unlabeled()

    _, y_lab, _ = make_features(lab, is_train=True)
    X_unlab, _, _ = make_features(unlab, is_train=False)

    model = load_model()

    # ── 검사 우선순위 곡선: CV holdout 사용 (과적합 없는 평가)
    print("\n[CV holdout 기반 검사 우선순위 곡선]")
    df_cv = pd.read_csv("outputs/cv_predictions.csv")
    stats, recall_at_k, fracs = make_inspection_curve(
        df_cv["y_true"].values, df_cv["y_prob"].values,
        title="검사 우선순위 효과 곡선 (앙상블 모델, CV holdout 기반)",
        save_path="outputs/05_inspection_priority_curve.png",
    )

    # ── 3단계 판정 (CV holdout 기반)
    print("\n[3단계 판정 성능]")
    tier_cv = three_tier_decision_percentile(df_cv["y_prob"].values)
    df_cv["tier"] = tier_cv
    for t, label in [(2, "즉시검사"), (1, "재검사"), (0, "통과")]:
        sub = df_cv[df_cv["tier"] == t]
        n_fail = sub["y_true"].sum()
        pct = n_fail / len(sub) * 100 if len(sub) > 0 else 0
        print(f"  {label}: {len(sub)}건 중 실제불량={int(n_fail)}개 ({pct:.1f}%)")
    df_cv.to_csv("outputs/cv_predictions.csv", index=False)

    # ── unlabeled 예측 결과 저장
    print("\nunlabeled 예측 중...")
    y_prob_unlab = model.predict_proba(X_unlab)[:, 1]
    tier_unlab = three_tier_decision_percentile(y_prob_unlab)
    y_pred_unlab = (y_prob_unlab >= np.percentile(y_prob_unlab, 95)).astype(int)

    result_df = pd.DataFrame({
        "Unnamed: 0": unlab["Unnamed: 0"],
        "mold":        unlab["mold"],
        "fail_prob":   y_prob_unlab.round(4),
        "PassOrFail":  y_pred_unlab,
        "tier":        tier_unlab,
    })
    result_df.to_csv("outputs/predictions_unlabeled.csv", index=False)
    print(f"예측 결과 저장: outputs/predictions_unlabeled.csv")
    print(f"  즉시검사: {(tier_unlab==2).sum()}, 재검사: {(tier_unlab==1).sum()}, 통과: {(tier_unlab==0).sum()}")
