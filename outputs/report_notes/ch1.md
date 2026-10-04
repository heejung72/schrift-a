# 1장 보고서 수치 — 데이터 이해 및 진단

## 1.1 데이터 구성

| 파일 | 행 수 | 불량 수 | 불량률 | 역할 |
|---|---|---|---|---|
| moldset_labeled_cn7.csv | 1,211 | 17 | 1.4% | 학습 |
| moldset_labeled_rg3.csv | 1,182 | 25 | 2.1% | 학습 |
| moldset_unlabeled_cn7.csv | 35,239 | - | - | 테스트(예측 대상) |
| moldset_unlabeled_rg3.csv | 35,941 | - | - | 테스트(예측 대상) |
| **합계** | **학습 2,393** | **42** | **1.75%** | |

## 1.2 생산단위
**1행 = 1 Shot (사출 1회 사이클)**. 근거: Injection_Time, Cycle_Time 등 1회 사이클 측정값.

## 1.3 변수 목록

| 그룹 | 변수명 | 의미 | 비고 |
|---|---|---|---|
| 사이클 시간 (5) | Injection_Time, Filling_Time, Plasticizing_Time, Cycle_Time, Clamp_Close_Time | 각 공정 단계 소요 시간 | |
| 위치 (3) | Cushion_Position, Plasticizing_Position, Clamp_Open_Position | 스크류/금형 위치 | Clamp_Open_Position: **labeled 전부 0 → 제거** |
| 속도/회전 (3) | Max_Injection_Speed, Max_Screw_RPM, Average_Screw_RPM | | |
| 압력 (4) | Max_Injection_Pressure, Max_Switch_Over_Pressure, Max_Back_Pressure, Average_Back_Pressure | | |
| 온도 (9) | Barrel_Temperature_1~6, Hopper_Temperature, Mold_Temperature_3~4 | | |
| 타깃 | **PassOrFail** | 0=양품, 1=불량 | labeled에만 존재 |

모든 변수 이미 Z-score 정규화 완료 (mean≈0, std≈1). 재정규화 불필요.

## 1.4 클래스 불균형
불량:양품 = 42:2351 = **1:56**. 정확도(Accuracy) 주 지표 부적합. F1-score 사용.

## 1.5 데이터 특이사항

**정규화 방식**: labeled+unlabeled를 금형별로 쌍 정규화 (cn7 기준, rg3 기준 별개).
→ cn7와 rg3의 수치 절대값 비교 불가. mold 변수(cn7=0, rg3=1) 필수.

**연속 중복 행**: labeled의 약 50%가 직전 행과 피처 동일 (공정 안정 구간 반복).
→ 유지. StratifiedGroupKFold로 같은 쌍이 같은 fold에 들어가도록 처리.

**라벨 충돌**: cn7 11건, rg3 25건 (동일 피처값에 양품·불량 혼재).
→ "측정 변수만으로 구분 불가 구간" — 5장 3단계 판정(재검사 구간)의 근거.

**이상치**: |z|>3인 행 0.3~1.4% (Plasticizing_Time 최대 z=20). 센서 오류 추정.

## 1.6 행 순서 = 시간 순서 (5가지 근거)

1. Unnamed:0 단조증가·중복없음 (Shot counter)
2. Lag-1 자기상관: Mold_Temperature_3=0.995, Clamp_Close_Time=0.975 (열관성)
3. 공정 변수의 점진적 드리프트 (워밍업→안정화)
4. cn7 불량 17개가 Shot 47~120(앞 10%)에 군집 → 기계 초기 불안정
5. unlabeled 최빈 간격=1 (42.7%) → 연속 shot 샘플링

## 1.7 cn7 불량 군집 — 중요 발견
cn7 불량 17개 모두 Shot 47~120에 집중. **워밍업 구간 불량** 현상.
→ 시간순 80/20 분할 시 cn7 불량이 전부 학습세트로 → StratifiedGroupKFold 채택 이유.
→ 3장 FN/FP 분석: "워밍업 구간 불량은 모델이 75% 탐지, 안정 구간 불량은 탐지 어려움"

## 1.8 데이터 누수
없음. 모든 변수가 최종 검사 전 공정 중 기록.

## 1.9 검증 전략

| 방법 | 설명 |
|---|---|
| **기본: StratifiedGroupKFold 5×10** | 연속 중복 쌍 → 같은 fold. 불량 fold별 균등 배분. fold당 불량 ~8개 |
| 보조: 금형 교차검증 | cn7→rg3, rg3→cn7 수행. F1=0. 원인: 금형별 독립 정규화. 한계로 기록. |
