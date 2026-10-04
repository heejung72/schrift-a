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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, FEATURE_COLS
    with open("outputs/final_model.pkl", "rb") as f:
        MODEL = pickle.load(f)
    with open("outputs/feature_columns.json") as f:
        FEATURE_COLS = json.load(f)
    init_db()
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


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    with open("app/static/index.html", encoding="utf-8") as f:
        return f.read()
