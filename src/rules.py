"""
공정 조건 규칙 도출 + '구분 불가 불량' 정량 증명 (3·4·5장 수상 포인트)

- 기존 labeled 데이터 + 피처(src.features)에서 사람이 읽는 IF-THEN 규칙을 통계로 도출.
  각 규칙: coverage(적용 건수), defect_rate(규칙 내 불량률=정밀도), lift(기준 대비 향상도),
  recall(전체 불량 중 포착 비율). 실제 SHAP 상위 피처(변화량·추세)로만 구성.
- '구분 불가' 증명: 라벨 충돌(동일 측정값에 양품·불량 공존) + 측정 정상 구간의 불량.
  → 3단계 '재검사'의 근거.
- 주의: spc_violation_count는 labeled 전 구간에서 0(3σ 미초과)이라 규칙에서 제외.
  연속형 EWMA 잔차(ewma_resid_norm)가 SPC/추세 신호를 대신 수행.

출력:
  outputs/process_rules.json  — 앱이 소비할 규칙(임계값·조치 포함)
  콘솔 요약                   — 보고서(report_notes)용 수치
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.features import load_labeled, make_features

RAW_MEASURED = [
    "Injection_Time", "Filling_Time", "Plasticizing_Time", "Cycle_Time",
    "Clamp_Close_Time", "Cushion_Position", "Plasticizing_Position",
    "Max_Injection_Speed", "Max_Screw_RPM", "Average_Screw_RPM",
    "Max_Injection_Pressure", "Max_Switch_Over_Pressure",
    "Max_Back_Pressure", "Average_Back_Pressure",
    "Barrel_Temperature_1", "Barrel_Temperature_2", "Barrel_Temperature_3",
    "Barrel_Temperature_4", "Barrel_Temperature_5", "Barrel_Temperature_6",
    "Hopper_Temperature", "Mold_Temperature_3", "Mold_Temperature_4",
]

# 불량 유형 군집(3장 계층적 군집) — 원인 진단·대응용 (예측 규칙과 별개)
CLUSTERS = [
    {"type": "A", "n": 4, "label": "공정 급변동형",
     "signature": "ewma_resid_norm 극단값(7.0)",
     "cause": "원료 투입 변동·스크류 마모·유압 불안정 등 급격한 공정 교란",
     "action_short": "즉시 공정 정지 검토, 직전 10샷 추세·설비 상태 점검",
     "action_long": "급변동 트리거 원인 설비 보전 이력화, 자동 정지 임계 연동"},
    {"type": "B", "n": 17, "label": "온도편차·금형닫힘 복합형",
     "signature": "barrel_temp_range 2.68 + Clamp_Close_Time 1.05",
     "cause": "금형 체결 불량·형체압력 이상 + 배럴 온도 산포 복합",
     "action_short": "금형 체결·형체압력 점검, 금형 온도 분포 확인 후 재검사",
     "action_long": "금형 보전주기 단축, 형체압력 상시 모니터링, 금형별 온도 관리도 운영"},
    {"type": "C", "n": 21, "label": "배럴 온도 편차형",
     "signature": "barrel_temp_range 2.54 + Barrel_Temperature_5",
     "cause": "배럴 히터·열전대 이상으로 인한 구간별 온도 편차",
     "action_short": "배럴 히터·열전대 상태 점검, 온도 안정 후 샘플 재검사",
     "action_long": "배럴 구간별 히터 PID 재튜닝, 열전대 교정주기 관리, 편차 경보 자동화"},
]


def stats(mask: np.ndarray, y: np.ndarray) -> dict:
    n = int(mask.sum())
    d = int(y[mask].sum())
    rate = d / n if n else 0.0
    return {"coverage": n, "defects": d, "defect_rate": round(rate, 4),
            "recall": round(d / y.sum(), 4),
            "lift": round(rate / y.mean(), 2) if y.mean() > 0 else 0.0}


if __name__ == "__main__":
    X, y_s, g = make_features(load_labeled(), is_train=True)
    lab = load_labeled()
    y = y_s.to_numpy()
    base = float(y.mean())
    n_def = int(y.sum())

    print("=" * 72)
    print(f"기준선: {len(y)}샷, 불량 {n_def}개, 불량률 {base*100:.2f}%")
    print("=" * 72)

    def P(col, q):
        return float(np.percentile(X[col], q))

    e = X["ewma_resid_norm"].to_numpy()
    pdelta = X["Plasticizing_Time_delta"].to_numpy()
    clamp = X["Clamp_Close_Time"].to_numpy()
    btr = X["barrel_temp_range"].to_numpy()

    thr = {
        "ewma_p90": P("ewma_resid_norm", 90),
        "ewma_p50": P("ewma_resid_norm", 50),
        "pdelta_p95": P("Plasticizing_Time_delta", 95),
        "pdelta_p05": P("Plasticizing_Time_delta", 5),
        "btr_p80": P("barrel_temp_range", 80),
        "clamp_p80": P("Clamp_Close_Time", 80),
    }

    # ── 예측 규칙 (실제 SHAP 상위 '변화·추세' 피처 기반)
    rules = []
    m1 = e >= thr["ewma_p90"]
    rules.append({"id": "R1", "cluster": "A", "name": "복합 공정 이탈 (급변동)",
                  "cond_text": "복합 EWMA 이탈도가 상위 10% 수준 (여러 공정 변수가 동시에 최근 추세에서 벗어남)",
                  "machine": [{"feat": "ewma_resid_norm", "op": ">=", "val": round(thr["ewma_p90"], 4)}],
                  "tier": 2, "action_short": CLUSTERS[0]["action_short"],
                  "action_long": CLUSTERS[0]["action_long"], **stats(m1, y)})

    m2 = (pdelta >= thr["pdelta_p95"]) | (pdelta <= thr["pdelta_p05"])
    rules.append({"id": "R2", "cluster": "A", "name": "가소화시간 급변",
                  "cond_text": "직전 shot 대비 가소화 시간이 급등 또는 급락 (상·하위 5%)",
                  "machine": [{"feat": "Plasticizing_Time_delta", "op": "abs>=",
                               "val": round(min(thr["pdelta_p95"], -thr["pdelta_p05"]), 4)}],
                  "tier": 2, "action_short": "가소화 공정(스크류 회전·백압·원료 건조) 점검 후 즉시검사",
                  "action_long": "원료 수분·계량 안정화, 스크류 마모 점검 주기화", **stats(m2, y)})

    m3 = (btr >= thr["btr_p80"]) & (clamp >= thr["clamp_p80"])
    rules.append({"id": "R3", "cluster": "B", "name": "온도편차·금형닫힘 복합",
                  "cond_text": "배럴 온도 편차 상위 20% + 금형 닫힘 시간 상위 20% 동시",
                  "machine": [{"feat": "barrel_temp_range", "op": ">=", "val": round(thr["btr_p80"], 4)},
                              {"feat": "Clamp_Close_Time", "op": ">=", "val": round(thr["clamp_p80"], 4)}],
                  "tier": 1, "action_short": CLUSTERS[1]["action_short"],
                  "action_long": CLUSTERS[1]["action_long"], **stats(m3, y)})

    print("\n[예측 규칙]")
    for r in rules:
        print(f"  {r['id']} {r['name']:<22} 적용 {r['coverage']:4d}건 | "
              f"불량률 {r['defect_rate']*100:5.1f}% | lift {r['lift']:4.1f}배 | "
              f"포착 {r['defects']}/{n_def} ({r['recall']*100:.0f}%)")

    union = m1 | m2 | m3
    s_union = stats(union, y)
    print(f"  {'규칙 합집합':<25} 적용 {s_union['coverage']:4d}건({s_union['coverage']/len(y)*100:.0f}%) | "
          f"포착 {s_union['defects']}/{n_def} ({s_union['recall']*100:.0f}%) | lift {s_union['lift']}배")

    # ── 라벨 충돌: 동일 측정값(raw)에 양품·불량 공존
    key = lab[RAW_MEASURED].round(3).astype(str).agg("|".join, axis=1)
    lab2 = lab.assign(_key=key)
    grp = lab2.groupby(["mold", "_key"])["PassOrFail"].agg(["nunique", "size", "sum"])
    coll = grp[grp["nunique"] > 1]
    n_coll_rows, n_coll_def = int(coll["size"].sum()), int(coll["sum"].sum())

    # ── 구분 불가 구간: 변화·추세 신호가 정상인데 불량
    indist = (e <= thr["ewma_p50"]) & (pdelta >= thr["pdelta_p05"]) & (pdelta <= thr["pdelta_p95"])
    s_ind = stats(indist, y)

    print("\n[구분 불가 증명]")
    print(f"  (1) 라벨 충돌: 동일 측정값에 양품·불량 공존 그룹 {len(coll)}개 / 불량 {n_coll_def}개")
    print(f"      → 불량의 {n_coll_def/n_def*100:.0f}%가 양품과 '완전히 동일한 측정값'에서 발생 = 측정값만으론 예측 불가")
    print(f"  (2) 변화·추세 정상 구간(EWMA 이탈 중앙값 이하 + 가소화 급변 아님): "
          f"{s_ind['coverage']}건 중 불량 {s_ind['defects']}개 (불량률 {s_ind['defect_rate']*100:.2f}%, lift {s_ind['lift']})")
    print(f"      → 전체 불량의 {s_ind['recall']*100:.0f}%가 여기 존재 → 3단계 '재검사'로 흡수")
    print(f"\n  cf) spc_violation_count 고유값={X['spc_violation_count'].nunique()} (전부 0, 3σ 미초과) → 규칙 제외")

    out = {
        "baseline_defect_rate": round(base, 4), "n_shots": len(y), "n_defects": n_def,
        "rules": rules, "rules_union": s_union, "clusters": CLUSTERS,
        "label_collision": {"groups": int(len(coll)), "rows": n_coll_rows,
                            "defects": n_coll_def, "defect_ratio": round(n_coll_def / n_def, 4)},
        "indistinguishable": {
            "cond_text": "EWMA 복합 이탈도 중앙값 이하 + 가소화시간 급변 없음 (변화·추세 신호 정상)",
            "machine": [{"feat": "ewma_resid_norm", "op": "<=", "val": round(thr["ewma_p50"], 4)}],
            "tier": 1, "action_short": "측정값만으로 판별 불가 → 샘플 재검사로 미탐지 위험 분산",
            "action_long": "비전검사·X-ray 등 측정 외 검사 축 추가 검토", **s_ind},
        "thresholds": {k: round(v, 4) for k, v in thr.items()},
    }
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/process_rules.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n저장: outputs/process_rules.json")
