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

### 10/05 완료
- [x] SHAP 상호작용값 (outputs/09_shap_dependence.png)
- [x] FN/FP 분석: 워밍업 vs 안정 구간, cn7 vs rg3 (outputs/11_fn_fp_analysis.png)
- [x] 3단계 임계값 최적화 → percentile 기반 (상위 5%=즉시, 6~20%=재검사)
- [x] report_notes/ch1.md, ch2.md, ch3.md 작성

#### 오후: 성능 향상 + 앱
- [x] Rolling EWMA 잔차 피처 추가 (src/features.py)
- [x] SPC 관리 한계 위반 횟수 피처 추가 (spc_violation_count)
- [x] 모델 재학습 — 앙상블(RF+LGB) 최종 선정: F1=0.343, PR-AUC=0.258
- [x] 검사 우선순위 20%→불량 62% 포착 (기존 52% 대비 +10%p)
- [x] 불량 유형 군집화 (3유형: 급변동형/복합형/온도형)
- [x] src/analysis.py (SHAP+FN/FP+군집화 통합 분석 스크립트)
- [x] FastAPI 앱 (app/main.py, simulator.py, static/index.html)
- [x] run_all.py 정리, requirements.txt 갱신

## 10/06 내일 할 일

### 오전 — 보고서 초안
- [ ] report_notes/ch4.md (현장 활용방안 — 3단계 판정, 불량 유형별 대응)
- [ ] report_notes/ch5.md (창의성 — SPC+ML 결합, Rolling 피처, 군집화 연결)
- [ ] report_notes/ch6.md (코드 재현성 — run_all.py 실행 설명)
- [ ] 보고서 본문 초안 (hwpx 붙여넣기용 텍스트)
- [ ] 대시보드 스크린샷 (사용자 직접: 브라우저 http://localhost:8000 → uvicorn 실행 후)

### 오후 — 마무리
- [ ] README.md 작성 (실행 환경, 실행 순서, 소요 시간)
- [ ] 코드 정리 및 재현성 확인 (run_all.py 처음부터 실행)
- [ ] PROGRESS.md 최종 업데이트

---

## 확정된 주요 결정사항

| 결정 | 내용 |
|---|---|
| 학습 데이터 | labeled_cn7 + labeled_rg3 합산 (mold 변수 추가) |
| 제거 변수 | Clamp_Open_Position (labeled에서 전부 0) |
| 정규화 | 이미 완료 (금형별 쌍 정규화). 재정규화 없음 |
| 검증 전략 | StratifiedGroupKFold 5×10 (연속 중복 쌍 = 같은 fold) |
| 최종 모델 | 앙상블(RF+LGB) — F1=0.343, PR-AUC=0.258 |
| 피처 | delta(3개) + SPC/Rolling(8개) + 기타(3개) = 총 37개 |
| 3단계 판정 | 상위 5%=즉시검사, 6~20%=재검사, 나머지=통과 |
| 불량 유형 | A:급변동형(4개), B:복합형(17개), C:온도형(21개) |

## 미결 사항

| 항목 | 상태 |
|---|---|
| 금형 교차검증 F1=0 | 정규화 차이 때문. 보고서에 한계로 기록 |
| 대시보드 스크린샷 | 사용자 직접 촬영 필요 (http://localhost:8000) |
| 모델 성능 분산 높음 | std 0.122~0.168 → 불량 42개 한계. 반복 CV로 보완 |

## 핵심 수치 요약 (보고서 전반에서 참조)

| 항목 | 수치 |
|---|---|
| 학습 데이터 | 2,393샷, 불량 42개(1.75%) |
| 최종 모델 F1 | 0.343 (베이스라인 0.240 대비 +43%) |
| 최종 모델 PR-AUC | 0.258 (무작위 0.018 대비 14.3배) |
| 상위 20% 검사 | 불량 62% 포착 (무작위 대비 3.1배) |
| 상위 30% 검사 | 불량 76% 포착 (무작위 대비 2.5배) |
| Rolling 피처 효과 | 20%구간 포착률 52%→62% (+10%p) |
| SPC 피처 순위 | SHAP 4위 (spc_violation_count) |
