"""
시뮬레이터 — unlabeled 데이터를 실시간처럼 FastAPI 서버로 흘려보냄
사용법: python app/simulator.py [--rate 2] [--limit 300] [--mold cn7]
  --rate  : 초당 전송 건수 (기본 2)
  --limit : 전송할 최대 샷 수 (기본 300)
  --mold  : cn7 | rg3 | all (기본 all)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse, time, requests, pandas as pd, numpy as np
from src.features import load_unlabeled

SERVER = "http://localhost:8000"

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


def run(rate: float, limit: int, mold_filter: str):
    df = load_unlabeled()
    if mold_filter != "all":
        df = df[df["mold"] == mold_filter].reset_index(drop=True)
    df = df.head(limit)

    # DB 초기화
    resp = requests.delete(f"{SERVER}/reset")
    print(f"DB 초기화: {resp.json()}")

    interval = 1.0 / rate
    n_immediate = n_review = n_pass = 0
    sent = 0

    print(f"\n시뮬레이션 시작: {len(df)}개 샷, {rate}건/초")
    print("-" * 60)

    for _, row in df.iterrows():
        payload = {
            "mold":    row["mold"],
            "shot_no": int(row["Unnamed: 0"]),
        }
        for c in RAW_COLS:
            payload[c] = float(row.get(c, 0.0))

        try:
            resp = requests.post(f"{SERVER}/predict", json=payload, timeout=5)
            result = resp.json()
            tier = result.get("tier", 0)
            prob = result.get("fail_prob", 0)

            if tier == 2:
                n_immediate += 1
                marker = "🔴 즉시검사"
            elif tier == 1:
                n_review += 1
                marker = "🟡 재검사  "
            else:
                n_pass += 1
                marker = "   통과    "

            sent += 1
            if sent % 10 == 0 or tier >= 1:
                print(f"  [{sent:4d}] Shot {payload['shot_no']:5d} ({row['mold']}) "
                      f"prob={prob:.3f} {marker}")
        except Exception as e:
            print(f"  오류: {e}")

        time.sleep(interval)

    print("-" * 60)
    print(f"\n완료: 총 {sent}건 전송")
    print(f"  즉시검사: {n_immediate}건  재검사: {n_review}건  통과: {n_pass}건")
    print(f"  검사 비율: {(n_immediate+n_review)/sent*100:.1f}%")

    # 최종 대기열 확인
    queue = requests.get(f"{SERVER}/inspection-queue?limit=20").json()
    print(f"\n검사 대기열 상위 20건 (현재 {queue['total']}건):")
    for item in queue["queue"][:10]:
        print(f"  Shot {item['shot_no']:5d} ({item['mold']}) prob={item['fail_prob']:.3f} [{item['tier_label']}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate",  type=float, default=2.0,  help="건수/초")
    parser.add_argument("--limit", type=int,   default=300,  help="최대 전송 건수")
    parser.add_argument("--mold",  type=str,   default="all", help="cn7 | rg3 | all")
    args = parser.parse_args()
    run(args.rate, args.limit, args.mold)
