"""
FastAPI 앱 — 사출성형 불량 예측 대시보드
- POST /predict        : 공정 데이터 입력 → 불량 확률 + 3단계 등급
- GET  /inspection-queue : 위험도 순 검사 대기열
- GET  /stats          : 요약 통계 (대시보드용)
- GET  /               : HTML 대시보드
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pickle, json, sqlite3, numpy as np, pandas as pd
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.features import make_features, ROLLING_WINDOW, MOLD_MAP

# ═══════════════════════════════════════════════════════
# 전역 상태
# ═══════════════════════════════════════════════════════
MODEL = None
FEATURE_COLS = None
MODEL_INFO = None
DB_PATH = "outputs/shots.db"

RAW_COLS = [
    "Injection_Time", "Filling_Time", "Plasticizing_Time", "Cycle_Time",
    "Clamp_Close_Time", "Cushion_Position", "Plasticizing_Position",
    "Max_Injection_Speed", "Max_Screw_RPM", "Average_Screw_RPM",
    "Max_Injection_Pressure", "Max_Switch_Over_Pressure",
    "Max_Back_Pressure", "Average_Back_Pressure",
    "Barrel_Temperature_1", "Barrel_Temperature_2", "Barrel_Temperature_3",
    "Barrel_Temperature_4", "Barrel_Temperature_5", "Barrel_Temperature_6",
    "Hopper_Temperature", "Mold_Temperature_3", "Mold_Temperature_4",
]


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs("outputs", exist_ok=True)
    conn = get_db()
    raw_cols_sql = ", ".join(f"{c} REAL" for c in RAW_COLS)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS shots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            shot_no INTEGER,
            mold TEXT,
            fail_prob REAL,
            tier INTEGER,
            inspected INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            {raw_cols_sql}
        )
    """)
    conn.commit()
    conn.close()


# ═══════════════════════════════════════════════════════
# 모델 성능/우선순위 정보 (대시보드 2·3페이지용)
# cv_predictions.csv에서 라이브 계산 + 실제 피처 중요도 추출
# ═══════════════════════════════════════════════════════
FEATURE_LABELS = {
    "ewma_resid_norm": "복합 EWMA 이탈도",
    "Clamp_Close_Time_ewma_resid": "금형닫힘 EWMA 잔차",
    "Plasticizing_Time_ewma_resid": "가소화 EWMA 잔차",
    "Plasticizing_Time_delta": "가소화시간 변화량",
    "Cushion_Position_ewma_resid": "쿠션위치 EWMA 잔차",
    "Max_Injection_Speed_ewma_resid": "사출속도 EWMA 잔차",
    "Cycle_Time_ewma_resid": "사이클타임 EWMA 잔차",
    "Max_Injection_Pressure_ewma_resid": "사출압력 EWMA 잔차",
    "barrel_temp_range": "배럴 온도 편차",
    "Clamp_Close_Time": "금형 닫힘 시간",
    "spc_violation_count": "SPC 위반 횟수",
    "Mold_Temperature_3": "금형 온도 3",
    "Cycle_Time_delta": "사이클타임 변화량",
    "Injection_Time_delta": "사출시간 변화량",
    "speed_pressure_ratio": "속도·압력 비",
    "Barrel_Temperature_5": "배럴 온도 5",
    "Max_Switch_Over_Pressure": "전환 압력",
    "Hopper_Temperature": "호퍼 온도",
    "Plasticizing_Time": "가소화 시간",
    "Cycle_Time": "사이클 타임",
    "Injection_Time": "사출 시간",
    "Filling_Time": "충전 시간",
    "Cushion_Position": "쿠션 위치",
    "Plasticizing_Position": "가소화 위치",
    "Max_Injection_Speed": "최대 사출속도",
    "Max_Injection_Pressure": "최대 사출압력",
    "Max_Screw_RPM": "최대 스크류 RPM",
    "Average_Screw_RPM": "평균 스크류 RPM",
    "Max_Back_Pressure": "최대 배압",
    "Average_Back_Pressure": "평균 배압",
    "Mold_Temperature_4": "금형 온도 4",
    "Barrel_Temperature_1": "배럴 온도 1",
    "Barrel_Temperature_2": "배럴 온도 2",
    "Barrel_Temperature_3": "배럴 온도 3",
    "Barrel_Temperature_4": "배럴 온도 4",
    "Barrel_Temperature_6": "배럴 온도 6",
    "Clamp_Close_Time_delta": "금형닫힘 변화량",
    "mold_enc": "금형 구분",
}


def _feature_group(name: str) -> str:
    """피처를 3그룹으로 분류: 신규(Rolling·SPC) / 제조지식 / 원시 공정값."""
    if "ewma_resid" in name or "spc_" in name or name in ("ewma_resid_norm", "spc_violation_count"):
        return "rolling"
    if name.endswith("_delta") or name in ("barrel_temp_range", "speed_pressure_ratio"):
        return "domain"
    return "base"


def compute_model_info() -> dict:
    from sklearn.metrics import average_precision_score

    info: dict = {}

    # cv_predictions.csv 기반 라이브 계산 (재현 가능)
    df = pd.read_csv("outputs/cv_predictions.csv")
    yt = df["y_true"].to_numpy()
    yp = df["y_prob"].to_numpy()
    n, nd = len(df), int(yt.sum())
    info["dataset"] = {"n_shots": n, "n_defect": nd,
                       "defect_rate": round(nd / n, 4)}

    order = np.argsort(-yp)
    cum = np.cumsum(yt[order])
    info["priority_curve"] = [
        {"pct": p, "capture": round(float(cum[int(round(n * p / 100)) - 1]) / nd, 4)
         if p > 0 else 0.0}
        for p in range(0, 101, 2)
    ]
    info["priority_points"] = {
        str(p): round(float(cum[int(round(n * p / 100)) - 1]) / nd, 4)
        for p in (10, 20, 30)
    }
    try:
        info["pr_auc_live"] = round(float(average_precision_score(yt, yp)), 4)
    except Exception:
        info["pr_auc_live"] = None

    # 실제 피처 중요도 (앙상블 rf+lgbm 평균, 정규화)
    try:
        rf = MODEL.named_estimators_["rf"].feature_importances_.astype(float)
        lg = MODEL.named_estimators_["lgbm"].feature_importances_.astype(float)
        rf, lg = rf / rf.sum(), lg / lg.sum()
        imp = (rf + lg) / 2
        pairs = sorted(zip(FEATURE_COLS, imp), key=lambda x: -x[1])[:12]
        mx = pairs[0][1]
        info["feature_importance"] = [
            {"name": c, "label": FEATURE_LABELS.get(c, c),
             "value": round(float(v), 4), "rel": round(float(v / mx), 4),
             "group": _feature_group(c)}
            for c, v in pairs
        ]
    except Exception:
        info["feature_importance"] = []

    # 보고서(report_notes ch2/ch3) 확정 수치
    info["metrics"] = {"f1": 0.343, "pr_auc": 0.258, "recall": 0.284,
                       "precision": 0.650, "brier": 0.015, "f1_baseline": 0.240}
    info["models"] = [
        {"name": "로지스틱 회귀", "role": "베이스라인", "f1": 0.240, "pr_auc": 0.161},
        {"name": "RandomForest", "role": "배깅", "f1": 0.336, "pr_auc": 0.260},
        {"name": "LightGBM", "role": "부스팅", "f1": 0.303, "pr_auc": 0.235},
        {"name": "앙상블 (RF+LGB)", "role": "최종 선정", "f1": 0.343, "pr_auc": 0.258},
    ]
    info["feature_steps"] = [
        {"label": "기본 (class_weight)", "f1": 0.222, "pr_auc": 0.221},
        {"label": "+ 임계값 최적화", "f1": 0.266, "pr_auc": 0.230},
        {"label": "+ Rolling·SPC 피처", "f1": 0.303, "pr_auc": 0.235},
    ]
    info["tiers"] = [
        {"name": "즉시 검사", "range": "상위 5%", "count": 121, "defects": 13,
         "precision": 0.107, "level": 2},
        {"name": "재검사", "range": "상위 6~20%", "count": 359, "defects": 13,
         "precision": 0.036, "level": 1},
        {"name": "통과", "range": "하위 80%", "count": 1913, "defects": 16,
         "precision": 0.008, "level": 0},
    ]
    info["clusters"] = [
        {"type": "A", "n": 4, "label": "공정 급변동형",
         "feature": "ewma_resid_norm 7.01", "action": "즉시 공정 정지 검토"},
        {"type": "B", "n": 17, "label": "온도편차 · 금형닫힘 복합형",
         "feature": "barrel_temp_range 2.68", "action": "금형 온도 · 압력 점검"},
        {"type": "C", "n": 21, "label": "배럴 온도 편차형",
         "feature": "barrel_temp_range 2.54", "action": "배럴 온도 제어 확인"},
    ]
    info["confusion"] = {"tp": 15, "fn": 27, "fp": 177, "tn": 2174, "recall": 0.36}
    info["warmup"] = [
        {"seg": "워밍업 구간", "recall": 0.50},
        {"seg": "안정 구간", "recall": 0.23},
    ]
    return info


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, FEATURE_COLS, MODEL_INFO
    with open("outputs/final_model.pkl", "rb") as f:
        MODEL = pickle.load(f)
    with open("outputs/feature_columns.json") as f:
        FEATURE_COLS = json.load(f)
    init_db()
    try:
        MODEL_INFO = compute_model_info()
        print("모델 성능 정보 계산 완료.")
    except Exception as e:
        print(f"모델 정보 계산 실패(대시보드 2·3페이지 제한): {e}")
        MODEL_INFO = {}
    print("모델 로드 완료. 서버 준비.")
    yield


app = FastAPI(title="사출성형 불량 예측 시스템", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


# ═══════════════════════════════════════════════════════
# Pydantic 스키마
# ═══════════════════════════════════════════════════════
class ShotInput(BaseModel):
    mold: str          # "cn7" or "rg3"
    shot_no: int
    Injection_Time: float
    Filling_Time: float
    Plasticizing_Time: float
    Cycle_Time: float
    Clamp_Close_Time: float
    Cushion_Position: float
    Plasticizing_Position: float
    Max_Injection_Speed: float
    Max_Screw_RPM: float
    Average_Screw_RPM: float
    Max_Injection_Pressure: float
    Max_Switch_Over_Pressure: float
    Max_Back_Pressure: float
    Average_Back_Pressure: float
    Barrel_Temperature_1: float
    Barrel_Temperature_2: float
    Barrel_Temperature_3: float
    Barrel_Temperature_4: float
    Barrel_Temperature_5: float
    Barrel_Temperature_6: float
    Hopper_Temperature: float
    Mold_Temperature_3: float
    Mold_Temperature_4: float


# ═══════════════════════════════════════════════════════
# 피처 계산
# ═══════════════════════════════════════════════════════
def compute_features(shot: ShotInput) -> pd.DataFrame:
    """DB에서 최근 ROLLING_WINDOW 샷을 가져와 rolling 피처까지 계산."""
    conn = get_db()
    recent_rows = pd.read_sql(
        f"SELECT {', '.join(RAW_COLS)}, mold FROM shots WHERE mold=? ORDER BY id DESC LIMIT {ROLLING_WINDOW - 1}",
        conn, params=[shot.mold]
    )
    conn.close()
    recent_rows = recent_rows.iloc[::-1].reset_index(drop=True)  # 시간순

    new_row = pd.DataFrame([{c: getattr(shot, c) for c in RAW_COLS}])
    new_row["mold"] = shot.mold
    combined = pd.concat([recent_rows, new_row], ignore_index=True)

    X, _, _ = make_features(combined, is_train=False)

    # 마지막 행(새 샷)의 피처만 반환
    X_last = X.iloc[[-1]].copy()

    # FEATURE_COLS 순서에 맞추고 없는 컬럼은 0으로 채움
    for col in FEATURE_COLS:
        if col not in X_last.columns:
            X_last[col] = 0.0
    return X_last[FEATURE_COLS]


def get_tier(prob: float, hi_pct: float = 0.95, lo_pct: float = 0.80) -> int:
    """앱 레벨에서는 최근 100개 샷 분위수 기준으로 tier 결정."""
    # 간단한 고정 임계값 사용 (배포 환경에서 calibration 필요)
    if prob >= 0.12:
        return 2  # 즉시검사
    elif prob >= 0.03:
        return 1  # 재검사
    else:
        return 0  # 통과


# ═══════════════════════════════════════════════════════
# 엔드포인트
# ═══════════════════════════════════════════════════════
@app.post("/predict")
async def predict(shot: ShotInput):
    """공정 데이터 → 불량 확률 + 3단계 등급 반환."""
    if shot.mold not in MOLD_MAP:
        raise HTTPException(400, f"mold는 'cn7' 또는 'rg3'이어야 합니다.")

    X = compute_features(shot)
    prob = float(MODEL.predict_proba(X)[0, 1])
    tier = get_tier(prob)
    tier_labels = {0: "통과", 1: "재검사", 2: "즉시검사"}

    # DB 저장
    conn = get_db()
    raw_vals = {c: getattr(shot, c) for c in RAW_COLS}
    cols = ["shot_no", "mold", "fail_prob", "tier"] + RAW_COLS
    vals = [shot.shot_no, shot.mold, round(prob, 4), tier] + [raw_vals[c] for c in RAW_COLS]
    placeholders = ", ".join(["?"] * len(cols))
    conn.execute(f"INSERT INTO shots ({', '.join(cols)}) VALUES ({placeholders})", vals)
    conn.commit()
    conn.close()

    return {
        "shot_no":    shot.shot_no,
        "mold":       shot.mold,
        "fail_prob":  round(prob, 4),
        "tier":       tier,
        "tier_label": tier_labels[tier],
        "action": {
            2: "즉시 검사 대기열에 추가",
            1: "샘플 재검사 예약",
            0: "통과 — 정상 생산 계속",
        }[tier],
    }


@app.get("/inspection-queue")
async def inspection_queue(limit: int = 50):
    """위험도 순 검사 대기열 반환 (미처리 건 우선)."""
    conn = get_db()
    rows = pd.read_sql(
        """SELECT id, shot_no, mold, fail_prob, tier, inspected, created_at
           FROM shots
           WHERE tier >= 1
           ORDER BY fail_prob DESC, id DESC
           LIMIT ?""",
        conn, params=[limit]
    )
    conn.close()
    tier_labels = {0: "통과", 1: "재검사", 2: "즉시검사"}
    result = rows.to_dict("records")
    for r in result:
        r["tier_label"] = tier_labels.get(r["tier"], "?")
    return {"total": len(result), "queue": result}


@app.post("/inspection-queue/{shot_id}/complete")
async def mark_inspected(shot_id: int):
    """검사 완료 처리."""
    conn = get_db()
    conn.execute("UPDATE shots SET inspected=1 WHERE id=?", [shot_id])
    conn.commit()
    conn.close()
    return {"status": "ok", "id": shot_id}


@app.get("/stats")
async def stats():
    """대시보드 요약 통계."""
    conn = get_db()
    df = pd.read_sql("SELECT fail_prob, tier, mold FROM shots", conn)
    conn.close()
    if len(df) == 0:
        return {"total": 0, "immediate": 0, "review": 0, "pass_": 0,
                "avg_prob": 0, "recent_probs": []}
    recent = pd.read_sql(
        "SELECT fail_prob, tier, mold FROM shots ORDER BY id DESC LIMIT 200",
        sqlite3.connect(DB_PATH)
    )
    return {
        "total":       int(len(df)),
        "immediate":   int((df["tier"] == 2).sum()),
        "review":      int((df["tier"] == 1).sum()),
        "pass_":       int((df["tier"] == 0).sum()),
        "avg_prob":    float(df["fail_prob"].mean().round(4)),
        "recent_probs": recent["fail_prob"].tolist()[::-1],
        "cn7_total":   int((df["mold"] == "cn7").sum()),
        "rg3_total":   int((df["mold"] == "rg3").sum()),
    }


@app.delete("/reset")
async def reset_db():
    """시뮬레이션 초기화 (DB 초기화)."""
    conn = get_db()
    conn.execute("DELETE FROM shots")
    conn.commit()
    conn.close()
    return {"status": "reset complete"}


@app.get("/model-info")
async def model_info():
    """모델 성능 · 피처 중요도 · 검사 우선순위 (대시보드 2·3페이지용)."""
    return MODEL_INFO or {}


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    with open("app/static/index.html", encoding="utf-8") as f:
        return f.read()
