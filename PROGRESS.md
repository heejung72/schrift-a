# PROGRESS — 새 세션 시작 시 이 파일부터 읽는다

## 현재 날짜: 10/05

## 완료된 단계

### 10/04 완료
- [x] 환경 세팅 (.venv, requirements.txt, 폴더 구조)
- [x] 1장 EDA (notebooks/01_eda.ipynb, outputs/01~08_*.png)
- [x] 2장 모델 비교 초안 (src/train.py — LR/RF/LGB/앙상블 5×10 CV)
- [x] 검사 우선순위 곡선 (outputs/05_inspection_priority_curve.png)
- [x] SHAP 변수 중요도 초안 (outputs/06_shap_importance.png)
- [x] 최종 모델 저장 (outputs/final_lgbm.pkl)
- [x] unlabeled 예측 파일 (outputs/predictions_unlabeled.csv)
- [x] 행 순서=시간 순서 검증 5가지 근거 (outputs/08_time_order_check.png)

## 10/05 오늘 할 일

### 오전 — 3장: 영향요인 및 오류분석
- [x] SHAP 상호작용값 (outputs/09_shap_dependence.png)
- [x] FN/FP 분석: 워밍업 vs 안정 구간, cn7 vs rg3 (outputs/11_fn_fp_analysis.png)
- [x] 3단계 임계값 최적화 → 상위 5%=즉시검사, 6~20%=재검사 (검사량 20%, 불량 52% 포착)
- [x] report_notes/ch1.md, ch2.md, ch3.md 작성
- [ ] 03_analysis.ipynb 노트북 작성 (선택 — 시간 여유 시)

### 오후 — 4·5장: FastAPI 앱
- [ ] app/main.py — FastAPI + SQLite
- [ ] POST /predict: 공정 데이터 → 불량 확률 + 3단계 등급
- [ ] GET /inspection-queue: 위험도 순 대기열
- [ ] app/simulator.py — unlabeled를 실시간처럼 흘려보내기
- [ ] app/static/index.html — 간단한 대시보드
- [ ] 대시보드 캡처

---

## 확정된 주요 결정사항

| 결정 | 내용 |
|---|---|
| 학습 데이터 | labeled_cn7 + labeled_rg3 합산 (mold 변수 추가) |
| 제거 변수 | Clamp_Open_Position (labeled에서 전부 0) |
| 정규화 | 이미 완료 (금형별 쌍 정규화). 재정규화 없음 |
| 검증 전략 | StratifiedGroupKFold 5×10 (연속 중복 쌍 = 같은 fold) |
| 최종 모델 | LightGBM (PR-AUC 0.230, 검사량 70%↓ 불량 74% 포착) |
| 제조지식 피처 | Plasticizing_Time_delta, Cycle_Time_delta, barrel_temp_range |

## 미결 사항

| 항목 | 상태 |
|---|---|
| 3단계 임계값 | lo=0.10, hi=0.35 → 통과 구간 불량 81%. 오늘 개선 |
| 금형 교차검증 F1=0 | 정규화 차이 때문. 보고서에 한계로 기록 |
| 모델 성능 분산 높음 | std 0.13~0.20 → 불량 42개 한계. 반복 CV로 보완 |
