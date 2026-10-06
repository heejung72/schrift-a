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

#### 대시보드 UI 전면 개편 (3페이지 SPA)
- [x] 프리미엄 다크 "관제실" 디자인 (사이드바 내비 + 글래스 카드 + SVG 차트)
- [x] 페이지1 실시간 모니터링: KPI + 실시간 확률 스트림 + 검사 대기열 (금형 필터)
- [x] 페이지2 모델 성능: 지표 카드 + 4모델 비교 + 영향변수 Top12 + 성능향상 단계
- [x] 페이지3 검사 우선순위: 우선순위 곡선 + 3단계 판정 + 불량유형 군집 + 오류분석
- [x] `/model-info` 엔드포인트 (cv_predictions 라이브 계산 + 실제 피처 중요도)
- [x] 해시 딥링크(#live/#model/#prio) + 대시보드 스크린샷 3장 (outputs/dashboard_*.png)
- [x] 2·3페이지는 시뮬레이터 없이도 /model-info만으로 완전히 채워짐 (캡처 용이)

## 10/06 완료 (마무리 작업)

### 보고서 초안
- [x] report_notes/ch4.md (현장 활용방안 — 사전경보·우선순위·공정점검·대시보드)
- [x] report_notes/ch5.md (창의성 — SPC+ML 결합/불균형/앙상블/보정/3단계/군집, 2·3장 연결)
- [x] report_notes/ch6.md (코드 재현성 — run_all.py·시드·버전고정·폴더구조)
- [x] 보고서 본문 초안 통합본 (report_notes/보고서_본문_초안.md, hwpx 붙여넣기용 6장 전체)
- [x] 대시보드 스크린샷 3장 자동 생성 (outputs/dashboard_1~3_*.png)

### 마무리
- [x] README.md 작성 (핵심결과·실행환경·실행순서·폴더구조·재현성)
- [x] run_all.py 서브프로세스를 sys.executable로 변경 (venv 혼입 방지)
- [x] 재현성 확인 완료 — run_all.py 8.0분, 전 수치 일치, cv_predictions.csv md5 완전 일치
- [x] PROGRESS.md 갱신

### 수상 각도 강화 (10/06 추가)
- [x] **중대 교정**: ch3 SHAP 순위 오류 수정 — 실제 Top4는 Plasticizing_Time_delta→ewma_resid_norm→Clamp_Close_Time→barrel_temp_range. spc_violation_count는 labeled 전구간 0(미발동)이라 "4위" 주장 삭제(ch2/ch3/PROGRESS 정정)
- [x] src/rules.py — 공정 조건 규칙 R1~R3(적용 21%→불량 60% 포착, lift 2.84배) + 구분불가 증명(라벨충돌 86%) → outputs/process_rules.json
- [x] src/advanced.py — 확률보정(Brier 0.015 이미 우수, isotonic 개선 0%) + 불확실성(RF↔LGB 불일치, 사분위 0→4.2% 단조) + 드리프트(PSI 다수>1) → 13·14·15.png, advanced_metrics.json
- [x] ch3/4/5/6 + 본문초안 전면 재구성 (구분불가 증명→규칙→현장조치, 경제성 ROI, 단기/장기 로드맵, 유형별 대책)
- [x] run_all.py에 rules·advanced 2단계 추가 (5단계 파이프라인)
- [x] **앱 작업자 개편**: /rules·/predict 설명 확장(규칙·유형·불확실성·조치), 대기열 클릭→shot 상세 드로어(왜+무엇을), 작업자 가이드 페이지(4번째 탭), ?shot= 딥링크
- [x] 대시보드 캡처 5장 (live/model/priority/guide/shot_detail)

### 남은 일 (사용자 몫)
- [ ] 보고서_본문_초안.md → hwpx 양식에 붙여넣고 그래프·캡처 삽입 후 PDF 저장
  (그래프: 01~15.png / 캡처: dashboard_1~5)
- [ ] 발표자료 PPT·PDF
- [ ] 만족도 설문 캡처 첨부
- [ ] 참가 계획서 내용과 본문 대조 (CLAUDE.md 3장 주의)
- [ ] (권장) 최종 제출 전 `python run_all.py` 1회 — rules·advanced 포함 5단계 재현 확인

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
| SHAP 실제 Top4 | Plasticizing_Time_delta → ewma_resid_norm → Clamp_Close_Time → barrel_temp_range |
| spc_violation_count | labeled 전 구간 0(미발동) → 규칙·중요도에서 제외. EWMA 잔차가 SPC 역할 |
| 공정 규칙 R1~R3 | 적용 21%로 불량 60% 포착(lift 2.84배) — ML과 동등, 설명가능 |
| 구분 불가 | 라벨 충돌로 불량 86%(36/42)가 양품과 동일 측정값 |
