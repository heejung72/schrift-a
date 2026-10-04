"""
2장: AI 예측모델 개발
- 4개 모델을 동일 조건(StratifiedGroupKFold 5fold × 10반복)에서 비교
- 주 지표: F1-score (불량=1). 보조: Recall, Precision, PR-AUC, Brier score
- 검증 전략 1: 연속 중복 구간을 같은 fold에 배정 (누수 방지)
- 검증 전략 2: 금형 교차검증 (cn7 학습→rg3 평가, 반대)
- 임계값: validation fold에서 F1 최대화 임계값을 각 fold마다 탐색
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (f1_score, recall_score, precision_score,
                             average_precision_score, brier_score_loss)
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

from src.features import load_labeled, make_features, SEED

N_SPLITS = 5
N_REPEATS = 10
np.random.seed(SEED)


def make_models():
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED, C=0.1)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 min_samples_leaf=2, n_jobs=-1, random_state=SEED)
    lgbm = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                               min_child_samples=5,
                               scale_pos_weight=55,  # ≈ 양품/불량 비율
                               n_jobs=-1, random_state=SEED, verbose=-1)
    ensemble = VotingClassifier(
        estimators=[
            ("lr",   LogisticRegression(class_weight="balanced", max_iter=1000, C=0.1, random_state=SEED)),
            ("rf",   RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                            min_samples_leaf=2, n_jobs=-1, random_state=SEED)),
            ("lgbm", lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                         scale_pos_weight=55, n_jobs=-1,
                                         random_state=SEED, verbose=-1)),
        ],
        voting="soft"
    )
    return {
        "로지스틱회귀(베이스라인)": lr,
        "RandomForest":             rf,
        "LightGBM":                 lgbm,
        "앙상블(LR+RF+LGB)":         ensemble,
    }


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """F1을 최대화하는 임계값을 반환한다."""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def evaluate_cv(X: pd.DataFrame, y: pd.Series, g: pd.Series) -> dict:
    """모델별 CV 성능을 딕셔너리로 반환한다."""
    models = make_models()
    results = {}

    for name, model in models.items():
        f1s, recs, precs, ap_scores, briers, thresholds = [], [], [], [], [], []
        print(f"  [{name}]", end="", flush=True)

        for repeat in range(N_REPEATS):
            sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                        random_state=SEED + repeat)
            for train_idx, val_idx in sgkf.split(X, y, groups=g):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model.fit(X_tr, y_tr)
                y_prob = model.predict_proba(X_val)[:, 1]

                t = best_threshold(y_val.values, y_prob)
                y_pred = (y_prob >= t).astype(int)

                f1s.append(f1_score(y_val, y_pred, zero_division=0))
                recs.append(recall_score(y_val, y_pred, zero_division=0))
                precs.append(precision_score(y_val, y_pred, zero_division=0))
                ap_scores.append(average_precision_score(y_val, y_prob))
                briers.append(brier_score_loss(y_val, y_prob))
                thresholds.append(t)

        results[name] = {
            "F1":        (np.mean(f1s),    np.std(f1s)),
            "Recall":    (np.mean(recs),   np.std(recs)),
            "Precision": (np.mean(precs),  np.std(precs)),
            "PR-AUC":    (np.mean(ap_scores), np.std(ap_scores)),
            "Brier":     (np.mean(briers), np.std(briers)),
            "Threshold": (np.mean(thresholds), np.std(thresholds)),
        }
        print(f" F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}  PR-AUC={np.mean(ap_scores):.3f}")

    return results


def evaluate_cross_mold(X: pd.DataFrame, y: pd.Series, mold_enc: pd.Series):
    """금형 교차검증: cn7→rg3, rg3→cn7. LightGBM만 사용."""
    print("\n  [금형 교차검증 - LightGBM]")
    lgbm = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                               scale_pos_weight=55, n_jobs=-1,
                               random_state=SEED, verbose=-1)
    cross_results = {}
    for train_mold, test_mold, label in [(0, 1, "cn7→rg3"), (1, 0, "rg3→cn7")]:
        tr_idx = mold_enc == train_mold
        te_idx = mold_enc == test_mold
        lgbm.fit(X[tr_idx], y[tr_idx])
        y_prob = lgbm.predict_proba(X[te_idx])[:, 1]
        t = best_threshold(y[te_idx].values, y_prob)
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y[te_idx], y_pred, zero_division=0)
        ap = average_precision_score(y[te_idx], y_prob)
        cross_results[label] = {"F1": f1, "PR-AUC": ap, "Threshold": t,
                                 "n_train": tr_idx.sum(), "n_test": te_idx.sum()}
        print(f"    {label}: F1={f1:.3f}  PR-AUC={ap:.3f}  임계값={t:.2f}")
    return cross_results


def print_results_table(results: dict):
    print("\n" + "=" * 90)
    print("【모델 비교 결과 (5-fold × 10회 반복 CV, mean ± std, 임계값 최적화 포함)】")
    print(f"{'모델':<22} {'F1':>12} {'Recall':>12} {'Precision':>12} {'PR-AUC':>12} {'Brier':>10} {'임계값':>8}")
    print("-" * 90)
    for name, m in results.items():
        row = f"{name:<22}"
        for key in ["F1", "Recall", "Precision", "PR-AUC", "Brier", "Threshold"]:
            v, s = m[key]
            row += f"  {v:.3f}±{s:.3f}"
        print(row)
    print("=" * 90)


if __name__ == "__main__":
    print("데이터 로드 중...")
    lab = load_labeled()
    X, y, g = make_features(lab, is_train=True)
    mold_enc = X["mold_enc"]

    print(f"학습 데이터: {X.shape}, 불량 비율: {y.mean()*100:.1f}%")
    print()

    print("교차검증 시작 (5-fold × 10회 반복)...")
    results = evaluate_cv(X, y, g)
    print_results_table(results)

    cross_results = evaluate_cross_mold(X, y, mold_enc)

    # 최종 LightGBM 전체 데이터로 학습 후 저장
    print("\n최종 LightGBM 전체 학습 중...")
    final_lgbm = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                     scale_pos_weight=55, n_jobs=-1,
                                     random_state=SEED, verbose=-1)
    final_lgbm.fit(X, y)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/final_lgbm.pkl", "wb") as f:
        pickle.dump(final_lgbm, f)
    print("모델 저장: outputs/final_lgbm.pkl")
