"""
데이터 로드 및 피처 엔지니어링
- 정규화 방식: labeled+unlabeled가 금형별(cn7, rg3)로 쌍 정규화됨
  → cn7와 rg3의 수치 절대값 비교 불가, mold 변수 필수
- Clamp_Open_Position: labeled에서 전부 0 (상수) → 제거
- 연속 중복 그룹: max 크기 2, StratifiedGroupKFold용 group_id 포함
- Rolling 피처: EWMA 잔차 (공정 드리프트 감지)
- SPC 피처: 3-sigma 관리 한계 위반 횟수
"""

import pandas as pd
import numpy as np

BASE = "data/1. 사출성형기 AI 데이터셋"

DROP_COLS = ["Unnamed: 0", "PassOrFail", "Clamp_Open_Position"]
MOLD_MAP = {"cn7": 0, "rg3": 1}

SEED = 42
ROLLING_WINDOW = 10  # SPC/EWMA 계산 기준 window

# EWMA·SPC 피처를 적용할 핵심 공정 변수
KEY_PROCESS_VARS = [
    "Cycle_Time", "Plasticizing_Time", "Clamp_Close_Time",
    "Max_Injection_Speed", "Max_Injection_Pressure", "Cushion_Position",
]


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
    """연속으로 동일한 피처값이 이어지는 구간을 하나의 그룹으로 묶는다."""
    changed = (df[feat_cols] != df[feat_cols].shift()).any(axis=1)
    return changed.cumsum().rename("group_id")


def add_rolling_spc_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    금형별 독립 계산:
      - {var}_ewma_resid : 현재값 - EWMA(span=ROLLING_WINDOW) → 최근 추세 이탈도
      - ewma_resid_norm  : 6개 변수 EWMA 잔차의 L2 크기 (복합 이탈도)
      - spc_violation_count : 몇 개 변수가 rolling 3-sigma 한계를 초과했는가
    """
    df = df.copy()
    for mold in df["mold"].unique():
        mask = df["mold"] == mold
        sub = df.loc[mask]

        residuals = []
        spc_viol = pd.Series(0.0, index=sub.index)

        for var in KEY_PROCESS_VARS:
            if var not in sub.columns:
                continue
            col = sub[var]

            # EWMA 잔차
            ewma = col.ewm(span=ROLLING_WINDOW, min_periods=1).mean()
            resid = col - ewma
            df.loc[mask, f"{var}_ewma_resid"] = resid
            residuals.append(resid)

            # Rolling 3-sigma SPC 위반 여부
            rm = col.rolling(ROLLING_WINDOW, min_periods=3).mean()
            rs = col.rolling(ROLLING_WINDOW, min_periods=3).std()
            # 초기 구간: expanding으로 대체
            rm = rm.where(rm.notna(), col.expanding().mean())
            rs = rs.where(rs.notna(), col.expanding().std()).fillna(1e-6)
            spc_viol += ((col > rm + 3 * rs) | (col < rm - 3 * rs)).astype(float)

        df.loc[mask, "spc_violation_count"] = spc_viol

        if residuals:
            stacked = pd.concat(residuals, axis=1)
            df.loc[mask, "ewma_resid_norm"] = np.sqrt((stacked ** 2).sum(axis=1))

    return df


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

    # 1) 사이클타임 delta 피처 — 금형별로 독립 계산 (경계 오염 방지)
    for c in ["Cycle_Time", "Injection_Time", "Plasticizing_Time"]:
        df[f"{c}_delta"] = 0.0
        for mold in df["mold"].unique():
            mask = df["mold"] == mold
            df.loc[mask, f"{c}_delta"] = df.loc[mask, c].diff().fillna(0)

    # 2) 사출 속도-압력 비 (공정 안정성 지표)
    df["speed_pressure_ratio"] = (
        df["Max_Injection_Speed"] / (df["Max_Injection_Pressure"].abs() + 1e-6)
    )

    # 3) 배럴 온도 편차 (1~6구간 최대-최소)
    barrel_cols = [f"Barrel_Temperature_{i}" for i in range(1, 7)]
    df["barrel_temp_range"] = df[barrel_cols].max(axis=1) - df[barrel_cols].min(axis=1)

    # 4) mold 인코딩
    df["mold_enc"] = df["mold"].map(MOLD_MAP)

    # 5) Rolling EWMA 잔차 + SPC 위반 횟수 (금형별 독립)
    df = add_rolling_spc_features(df)

    X_cols = [c for c in df.columns if c not in DROP_COLS + ["mold", "group_id"]]
    X = df[X_cols].copy()
    y = df["PassOrFail"] if is_train else None

    return X, y, g
