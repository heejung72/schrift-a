"""
최종 모델로 unlabeled 데이터 예측 및 검사 우선순위 곡선 생성
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


def load_model(path="outputs/final_lgbm.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def make_inspection_curve(y_true: np.ndarray, y_prob: np.ndarray,
                          title: str, save_path: str):
    """위험도 상위 k% 검사 시 불량 포착률 곡선."""
    sorted_idx = np.argsort(y_prob)[::-1]
    n_total = len(y_true)
    n_fail = y_true.sum()

    fracs = np.arange(1, n_total + 1) / n_total
    cum_fails = y_true[sorted_idx].cumsum()
    recall_at_k = cum_fails / n_fail

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fracs * 100, recall_at_k * 100, color="#1976D2", lw=2, label="모델 우선순위")
    ax.plot([0, 100], [0, 100], "--", color="gray", label="무작위 검사 (기준선)")
    ax.fill_between(fracs * 100, recall_at_k * 100, fracs * 100, alpha=0.2, color="#1976D2")

    for k in [10, 20, 30]:
        idx = int(n_total * k / 100) - 1
        cap = recall_at_k[idx] * 100
        ax.scatter(k, cap, zorder=5, s=60, color="red")
        ax.annotate(f"상위 {k}%→불량 {cap:.0f}% 포착", xy=(k, cap),
                    xytext=(k + 5, cap - 8), fontsize=9)

    ax.set_xlabel("검사 비율 (%)")
    ax.set_ylabel("불량 포착률 (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"그래프 저장: {save_path}")

    for k in [10, 20, 30]:
        idx = int(n_total * k / 100) - 1
        cap = recall_at_k[idx] * 100
        print(f"  상위 {k}% 검사 → 불량 {cap:.0f}% 포착 (전수 대비 검사량 {100-k}% 감소)")

    return recall_at_k, fracs


def three_tier_decision(y_prob: np.ndarray,
                        lo_thresh: float = 0.10,
                        hi_thresh: float = 0.35) -> np.ndarray:
    """불확실성 기반 3단계 판정.

    Returns
    -------
    tier : 0=통과, 1=샘플·재검사, 2=즉시 검사
    """
    tier = np.zeros(len(y_prob), dtype=int)
    tier[y_prob >= lo_thresh] = 1
    tier[y_prob >= hi_thresh] = 2
    return tier


if __name__ == "__main__":
    print("데이터 로드 중...")
    lab = load_labeled()
    unlab = load_unlabeled()

    X_lab, y_lab, g_lab = make_features(lab, is_train=True)
    X_unlab, _, g_unlab = make_features(unlab, is_train=False)

    model = load_model()

    # ── 학습 데이터 내 검사 우선순위 곡선 (CV 대신 전체 재예측으로 방향 확인용)
    y_prob_lab = model.predict_proba(X_lab)[:, 1]
    print("\n[학습 데이터 내 검사 우선순위 곡선]")
    make_inspection_curve(
        y_lab.values, y_prob_lab,
        title="검사 우선순위 효과 곡선 (학습 데이터 기준)",
        save_path="outputs/05_inspection_priority_curve.png"
    )

    # ── 3단계 판정
    tier = three_tier_decision(y_prob_lab)
    tier_df = pd.DataFrame({"y_true": y_lab, "y_prob": y_prob_lab, "tier": tier})
    for t, label in [(2, "즉시검사"), (1, "재검사"), (0, "통과")]:
        sub = tier_df[tier_df["tier"] == t]
        n_fail = sub["y_true"].sum()
        print(f"  {label}: {len(sub)}건 중 실제불량={n_fail}개 ({n_fail/len(sub)*100:.1f}%)")

    # ── unlabeled 예측 결과 저장
    y_prob_unlab = model.predict_proba(X_unlab)[:, 1]
    tier_unlab = three_tier_decision(y_prob_unlab)
    y_pred_unlab = (y_prob_unlab >= 0.15).astype(int)

    result_df = pd.DataFrame({
        "Unnamed: 0": unlab["Unnamed: 0"],
        "mold":        unlab["mold"],
        "fail_prob":   y_prob_unlab.round(4),
        "PassOrFail":  y_pred_unlab,
        "tier":        tier_unlab,
    })
    result_df.to_csv("outputs/predictions_unlabeled.csv", index=False)
    print(f"\n예측 결과 저장: outputs/predictions_unlabeled.csv")
    print(f"  예측 불량 수: {y_pred_unlab.sum()}개 / {len(y_pred_unlab)}개 ({y_pred_unlab.mean()*100:.1f}%)")
    print(f"  즉시검사: {(tier_unlab==2).sum()}, 재검사: {(tier_unlab==1).sum()}, 통과: {(tier_unlab==0).sum()}")
