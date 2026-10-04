"""
3장: 영향요인 및 오류분석
- SHAP 변수 중요도 및 beeswarm
- SHAP 의존도 플롯 (상호작용)
- FN/FP 오류분석 (워밍업 vs 안정 구간, 금형별)
- 불량 유형 군집화 (계층적 군집, k=3)
- 3단계 판정 시각화
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from src.features import load_labeled, make_features, SEED

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
os.makedirs("outputs", exist_ok=True)


# ═══════════════════════════════════════════════════════
# 0. 데이터 로드
# ═══════════════════════════════════════════════════════
def load_data():
    lab = load_labeled()
    X, y, g = make_features(lab, is_train=True)
    df_cv = pd.read_csv("outputs/cv_predictions.csv")
    with open("outputs/final_lgbm.pkl", "rb") as f:
        lgbm = pickle.load(f)
    return lab, X, y, lgbm, df_cv


# ═══════════════════════════════════════════════════════
# 1. SHAP 변수 중요도
# ═══════════════════════════════════════════════════════
def run_shap(X: pd.DataFrame, lgbm, n_sample: int = 500):
    import shap
    sampler = shap.sample(X, n_sample, random_state=SEED)
    explainer = shap.TreeExplainer(lgbm)
    shap_vals = explainer.shap_values(sampler)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    # 평균 |SHAP| 기준 Top 15
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:15]
    top_feat = X.columns[top_idx]
    top_vals = mean_abs[top_idx]

    # Bar plot
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#D32F2F" if "ewma" in f or "spc" in f or "delta" in f or "range" in f
              else "#1976D2" for f in top_feat]
    ax.barh(range(len(top_feat))[::-1], top_vals, color=colors)
    ax.set_yticks(range(len(top_feat)))
    ax.set_yticklabels(top_feat[::-1], fontsize=9)
    ax.set_xlabel("평균 |SHAP| 값")
    ax.set_title("SHAP 변수 중요도 Top 15\n(빨간색: 제조지식·Rolling·SPC 기반 피처)")
    plt.tight_layout()
    plt.savefig("outputs/06_shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("저장: outputs/06_shap_importance.png")

    # Beeswarm
    shap_df = pd.DataFrame(shap_vals[:, top_idx], columns=top_feat)
    X_sample_top = sampler[top_feat].reset_index(drop=True)
    fig, axes = plt.subplots(1, 1, figsize=(10, 7))
    for i, feat in enumerate(top_feat[:10]):
        ax = plt.gca()
    try:
        shap.summary_plot(shap_vals[:, top_idx], sampler[top_feat],
                          feature_names=list(top_feat), show=False, max_display=15)
        plt.tight_layout()
        plt.savefig("outputs/07_shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("저장: outputs/07_shap_beeswarm.png")
    except Exception as e:
        print(f"  beeswarm 저장 실패: {e}")

    # 의존도 플롯: Top1 × Top2
    feat1, feat2 = top_feat[0], top_feat[1]
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(sampler[feat1], shap_vals[:, top_idx[0]],
                    c=sampler[feat2], cmap="RdBu_r", alpha=0.6, s=15)
    plt.colorbar(sc, ax=ax, label=feat2)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel(feat1)
    ax.set_ylabel(f"SHAP({feat1})")
    ax.set_title(f"SHAP 의존도: {feat1}\n(색상={feat2})")
    plt.tight_layout()
    plt.savefig("outputs/09_shap_dependence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("저장: outputs/09_shap_dependence.png")

    return mean_abs, top_feat, top_vals


# ═══════════════════════════════════════════════════════
# 2. 불량 유형 군집화
# ═══════════════════════════════════════════════════════
def cluster_defects(X: pd.DataFrame, y: pd.Series, top_feat: np.ndarray):
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

    X_fail = X[y == 1].reset_index(drop=True)
    feats = [f for f in top_feat[:10] if f in X_fail.columns]
    X_clus = X_fail[feats].fillna(0)

    Z = linkage(X_clus, method="ward")
    n_clusters = 3
    labels = fcluster(Z, n_clusters, criterion="maxclust")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 덴드로그램
    ax = axes[0]
    dendrogram(Z, ax=ax, leaf_rotation=90, leaf_font_size=7,
               color_threshold=Z[-n_clusters+1, 2])
    ax.set_title(f"불량 유형 덴드로그램 (n={len(X_fail)}개)")
    ax.set_ylabel("거리 (Ward 연결)")
    ax.axhline(Z[-(n_clusters-1), 2], color="red", ls="--", lw=1.5, label=f"k={n_clusters} 기준선")
    ax.legend()

    # 군집별 주요 변수 평균
    ax = axes[1]
    X_fail["cluster"] = labels
    cluster_means = X_fail.groupby("cluster")[feats[:6]].mean()
    cluster_means.T.plot(kind="bar", ax=ax, color=["#1976D2","#D32F2F","#388E3C"])
    ax.set_title("불량 유형별 주요 변수 평균 (Z-score 정규화)")
    ax.set_ylabel("평균값 (Z-score)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.legend(title="불량 유형", labels=["유형A","유형B","유형C"])

    plt.tight_layout()
    plt.savefig("outputs/12_defect_clustering.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("저장: outputs/12_defect_clustering.png")

    # 군집별 요약
    print("\n[불량 유형 군집화 결과]")
    for c in sorted(X_fail["cluster"].unique()):
        sub = X_fail[X_fail["cluster"] == c]
        print(f"  유형 {chr(64+c)} (n={len(sub)}): ", end="")
        top3 = sub[feats[:6]].mean().abs().nlargest(3)
        print(", ".join([f"{k}={v:.2f}" for k, v in top3.items()]))

    return labels, X_fail


# ═══════════════════════════════════════════════════════
# 3. FN/FP 오류 분석
# ═══════════════════════════════════════════════════════
def analyze_errors(df_cv: pd.DataFrame):
    tp = df_cv[df_cv["error_type"] == "TP"]
    fn = df_cv[df_cv["error_type"] == "FN"]
    fp = df_cv[df_cv["error_type"] == "FP"]
    tn = df_cv[df_cv["error_type"] == "TN"]

    print(f"\n[혼동 행렬] TP={len(tp)}, FN={len(fn)}, FP={len(fp)}, TN={len(tn)}")
    print(f"  Recall={len(tp)/(len(tp)+len(fn)):.1%}, Precision={len(tp)/(len(tp)+len(fp)):.1%}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (1) 금형별 FN/TP
    ax = axes[0]
    molds = ["cn7", "rg3"]
    fn_by_mold = [fn[fn["mold"]==m].shape[0] for m in molds]
    tp_by_mold = [tp[tp["mold"]==m].shape[0] for m in molds]
    total_by_mold = [
        df_cv[(df_cv["mold"]==m) & (df_cv["y_true"]==1)].shape[0] for m in molds
    ]
    x = np.arange(len(molds))
    ax.bar(x - 0.2, tp_by_mold, 0.35, label="TP(정탐)", color="#1976D2")
    ax.bar(x + 0.2, fn_by_mold, 0.35, label="FN(미탐)", color="#D32F2F")
    for i, (tp_n, tot) in enumerate(zip(tp_by_mold, total_by_mold)):
        ax.text(i-0.2, tp_n+0.1, f"Recall\n{tp_n/tot:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(molds)
    ax.set_title("금형별 TP/FN")
    ax.set_ylabel("샷 수")
    ax.legend()

    # (2) 워밍업 vs 안정 구간
    ax = axes[1]
    for col_idx, (subset, label) in enumerate([(tp, "TP"), (fn, "FN")]):
        wu_n = subset["warmup"].sum()
        st_n = (~subset["warmup"]).sum()
        ax.bar(col_idx - 0.2, wu_n, 0.35, color="#FFA000", label="워밍업" if col_idx==0 else "")
        ax.bar(col_idx + 0.2, st_n, 0.35, color="#455A64", label="안정" if col_idx==0 else "")
        ax.text(col_idx-0.2, wu_n+0.1, f"{wu_n}", ha="center", fontsize=8)
        ax.text(col_idx+0.2, st_n+0.1, f"{st_n}", ha="center", fontsize=8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["TP(정탐)", "FN(미탐)"])
    ax.set_title("워밍업 vs 안정 구간별 오류 분포")
    ax.set_ylabel("샷 수")
    ax.legend()

    # (3) y_prob 분포
    ax = axes[2]
    ax.hist(fn["y_prob"], bins=20, alpha=0.7, color="#D32F2F", label=f"FN(미탐) n={len(fn)}")
    ax.hist(tp["y_prob"], bins=20, alpha=0.7, color="#1976D2", label=f"TP(정탐) n={len(tp)}")
    ax.set_xlabel("예측 불량 확률")
    ax.set_ylabel("빈도")
    ax.set_title("FN·TP의 예측 확률 분포")
    ax.legend()

    plt.tight_layout()
    plt.savefig("outputs/11_fn_fp_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("저장: outputs/11_fn_fp_analysis.png")

    # 통계 출력
    print(f"\n  [워밍업 구간 분석]")
    wu_tp = tp["warmup"].mean()
    wu_fn = fn["warmup"].mean()
    print(f"    TP 중 워밍업 비율: {wu_tp:.0%}")
    print(f"    FN 중 워밍업 비율: {wu_fn:.0%}")
    wu_defects = df_cv[(df_cv["y_true"]==1) & (df_cv["warmup"]==True)]
    wu_recall = tp[tp["warmup"]].shape[0] / len(wu_defects) if len(wu_defects) > 0 else 0
    st_defects = df_cv[(df_cv["y_true"]==1) & (df_cv["warmup"]==False)]
    st_recall = tp[~tp["warmup"]].shape[0] / len(st_defects) if len(st_defects) > 0 else 0
    print(f"    워밍업 구간 Recall: {wu_recall:.0%}")
    print(f"    안정 구간 Recall:   {st_recall:.0%}")

    return tp, fn, fp


# ═══════════════════════════════════════════════════════
# 4. 3단계 판정 시각화
# ═══════════════════════════════════════════════════════
def plot_3tier(df_cv: pd.DataFrame):
    from src.predict import three_tier_decision_percentile

    y_prob = df_cv["y_prob"].values
    y_true = df_cv["y_true"].values
    tier = three_tier_decision_percentile(y_prob)

    hi = np.percentile(y_prob, 95)
    lo = np.percentile(y_prob, 80)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.hist(y_prob[y_true == 0], bins=50, alpha=0.6, color="#455A64", label="양품")
    ax.hist(y_prob[y_true == 1], bins=15, alpha=0.9, color="#D32F2F", label="불량")
    ax.axvline(lo, color="#FFA000", lw=2, ls="--", label=f"재검사 임계값 ({lo:.3f})")
    ax.axvline(hi, color="#D32F2F", lw=2, ls="--", label=f"즉시검사 임계값 ({hi:.3f})")
    ax.set_xlabel("예측 불량 확률")
    ax.set_ylabel("빈도")
    ax.set_title("3단계 판정 구간 (CV holdout)")
    ax.legend(fontsize=8)

    ax = axes[1]
    labels = ["통과\n(tier=0)", "재검사\n(tier=1)", "즉시검사\n(tier=2)"]
    for t, label, color in [(0, labels[0], "#455A64"), (1, labels[1], "#FFA000"), (2, labels[2], "#D32F2F")]:
        sub_y = y_true[tier == t]
        n = len(sub_y)
        n_fail = sub_y.sum()
        ax.bar(t, n_fail, color=color, alpha=0.9, label=f"{label} 불량")
        ax.bar(t, n - n_fail, bottom=n_fail, color=color, alpha=0.3)
        ax.text(t, n + 5, f"불량 {int(n_fail)}개\n({n_fail/n*100:.1f}%)", ha="center", fontsize=8)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("샷 수")
    ax.set_title("3단계 판정별 불량 분포")

    plt.tight_layout()
    plt.savefig("outputs/10_3tier_decision.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("저장: outputs/10_3tier_decision.png")

    # 수치 요약
    print("\n[3단계 판정 요약]")
    for t, label in [(2, "즉시검사"), (1, "재검사"), (0, "통과")]:
        sub = y_true[tier == t]
        n_fail = int(sub.sum())
        print(f"  {label} ({len(sub)}건): 실제불량 {n_fail}개 ({n_fail/len(sub)*100:.1f}%)")
    tier12 = y_true[tier >= 1]
    print(f"  1+2단계 합산: 검사량 {(tier>=1).sum()/len(tier)*100:.0f}%, 불량 포착 {int(y_true[tier>=1].sum())}/{int(y_true.sum())}개 ({y_true[tier>=1].sum()/y_true.sum()*100:.0f}%)")


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    print("═" * 60)
    print("분석 시작: SHAP + 불량 군집화 + FN/FP + 3단계 판정")
    print("═" * 60)

    lab, X, y, lgbm, df_cv = load_data()

    print("\n[1] SHAP 변수 중요도 계산...")
    mean_abs, top_feat, top_vals = run_shap(X, lgbm)

    print("\n[2] 불량 유형 군집화...")
    labels, X_fail = cluster_defects(X, y, top_feat)

    print("\n[3] FN/FP 오류 분석...")
    tp, fn, fp = analyze_errors(df_cv)

    print("\n[4] 3단계 판정 시각화...")
    plot_3tier(df_cv)

    print("\n✓ 모든 분석 완료")
    print("  생성된 파일: 06, 07, 09, 10, 11, 12번 그래프")
