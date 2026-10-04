"""
데이터 로드 및 피처 엔지니어링
- 정규화 방식: labeled+unlabeled가 금형별(cn7, rg3)로 쌍 정규화됨
  → cn7와 rg3의 수치 절대값 비교 불가, mold 변수 필수
- Clamp_Open_Position: labeled에서 전부 0 (상수) → 제거
- 연속 중복 그룹: max 크기 2, StratifiedGroupKFold용 group_id 포함
"""

import pandas as pd
import numpy as np

BASE = "data/1. 사출성형기 AI 데이터셋"

DROP_COLS = ["Unnamed: 0", "PassOrFail", "Clamp_Open_Position"]
MOLD_MAP = {"cn7": 0, "rg3": 1}

SEED = 42


def load_labeled() -> pd.DataFrame:
    lc = pd.read_csv(f"{BASE}/moldset_labeled_cn7.csv")
    lr = pd.read_csv(f"{BASE}/moldset_labeled_rg3.csv")
    lab = pd.concat([lc.assign(mold="cn7"), lr.assign(mold="rg3")], ignore_index=True)
    return lab


def load_unlabeled() -> pd.DataFrame:
    uc = pd.read_csv(f"{BASE}/moldset_unlabeled_cn7.csv")
    ur = pd.read_csv(f"{BASE}/moldset_unlabeled_rg3.csv")
    unlab = pd.concat([uc.assign(mold="cn7"), ur.assign(mold="rg3")], ignore_index=True)
    return unlab


def assign_groups(df: pd.DataFrame, feat_cols: list) -> pd.Series:
    """연속으로 동일한 피처값이 이어지는 구간을 하나의 그룹으로 묶는다.
    같은 그룹은 StratifiedGroupKFold에서 같은 fold에 배정된다."""
    changed = (df[feat_cols] != df[feat_cols].shift()).any(axis=1)
    return changed.cumsum().rename("group_id")


def make_features(df: pd.DataFrame, is_train: bool = True):
    """피처 행렬 X, 타깃 y, 그룹 g를 반환한다.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series or None  (unlabeled이면 None)
    g : pd.Series  (group_id)
    """
    feat_cols = [c for c in df.columns if c not in DROP_COLS + ["mold", "group_id"]]

    g = assign_groups(df, feat_cols)

    df = df.copy()

    # 제조지식 기반 피처
    # 1) 사이클타임 편차: 이전 샷 대비 변화량 (워밍업·불안정 구간 감지)
    for c in ["Cycle_Time", "Injection_Time", "Plasticizing_Time"]:
        df[f"{c}_delta"] = df[c].diff().fillna(0)

    # 2) 사출 속도-압력 비 (공정 안정성 지표)
    df["speed_pressure_ratio"] = df["Max_Injection_Speed"] / (df["Max_Injection_Pressure"].abs() + 1e-6)

    # 3) 배럴 온도 편차 (1~6구간 최대-최소)
    barrel_cols = [f"Barrel_Temperature_{i}" for i in range(1, 7)]
    df["barrel_temp_range"] = df[barrel_cols].max(axis=1) - df[barrel_cols].min(axis=1)

    # 4) mold 인코딩
    df["mold_enc"] = df["mold"].map(MOLD_MAP)

    X_cols = [c for c in df.columns if c not in DROP_COLS + ["mold", "group_id"]]
    X = df[X_cols].copy()
    y = df["PassOrFail"] if is_train else None

    return X, y, g
