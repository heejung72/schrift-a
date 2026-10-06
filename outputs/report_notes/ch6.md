# 6장 보고서 수치 — 코드 및 재현성

평가 요건: 동일 환경에서 **전처리→학습→추론→결과 생성**까지 자동 실행.

## 6.1 한 번에 실행

```bash
python run_all.py
```

- `run_all.py`가 5단계를 순서대로 호출(서브프로세스는 `sys.executable`로 현재 가상환경 보장):
  1. `src.train` — 피처 엔지니어링 + 4모델 CV 비교 + 금형 교차검증 + 최종 모델 저장 (1·2장)
  2. `src.analysis` — SHAP·상호작용·FN/FP·군집화 (3장)
  3. `src.rules` — 공정 조건 규칙(R1~R3) + 구분불가 증명(라벨 충돌) (3·4·5장)
  4. `src.advanced` — 확률 보정·불확실성·드리프트 심화분석 (5·6장)
  5. `src.predict` — 검사 우선순위 곡선 + 3단계 판정 + unlabeled 예측 파일
- 소요 시간: **약 12~17분** (CV 5-fold × 10회 반복 × 4모델 + SHAP + 심화 CV).

## 6.2 재현성 보장 장치

| 요소 | 조치 |
|---|---|
| 랜덤 시드 | `SEED=42` 단일 상수(`src/features.py`)를 전 모듈이 공유. CV는 `SEED+repeat`로 반복별 고정 |
| 라이브러리 버전 | `requirements.txt` 전부 **버전 고정**(==) |
| 인터프리터 | `run_all.py`가 `sys.executable` 사용 → venv 밖 python 혼입 방지 |
| 피처 순서 | `feature_columns.json`에 학습 시 컬럼 순서 저장 → 추론·앱에서 동일 순서 강제 |
| 결정론성 | LGB/RF/SGKFold 모두 `random_state` 지정 → 동일 결과 재현 |

## 6.3 환경

- Python 3.9 (.venv)
- 주요 패키지: scikit-learn 1.6.1, lightgbm 4.6.0, shap 0.49.1, fastapi 0.128.8 (전체 `requirements.txt`)

## 6.4 폴더 구조

```
injection-ai/
├── data/                  # KAMP 원본 CSV (labeled/unlabeled)
├── src/
│   ├── features.py        # 로드·정규화 대응·Rolling/SPC 피처
│   ├── train.py           # 4모델 CV 비교·금형 교차검증·모델 저장
│   ├── analysis.py        # SHAP·군집화·FN/FP
│   └── predict.py         # 우선순위 곡선·3단계·unlabeled 예측
├── app/                   # FastAPI 서버 + 3페이지 대시보드 + 시뮬레이터
├── outputs/               # 모델·예측·그래프·report_notes
├── run_all.py             # 전체 파이프라인 1회 실행
├── requirements.txt       # 버전 고정
└── README.md
```

## 6.5 생성물 (run_all.py 출력)

| 파일 | 설명 |
|---|---|
| `final_model.pkl` | 앙상블 RF+LGB (예측·앱용) |
| `final_lgbm.pkl` | LightGBM (SHAP 분석용) |
| `feature_columns.json` | 피처 컬럼 순서(37개) |
| `cv_predictions.csv` | CV holdout 예측 (3장 분석·대시보드 /model-info 라이브 계산용) |
| `predictions_unlabeled.csv` | **제출용 예측 파일** (원본 행 순서 + fail_prob + PassOrFail + tier) |
| `process_rules.json` | 공정 조건 규칙 R1~R3 + 구분불가 통계 (앱·보고서용) |
| `advanced_metrics.json` | 보정·불확실성·드리프트 지표 |
| `0*.png`, `13~15.png` | 그래프 01~12 + 보정·불확실성·드리프트 |

## 6.6 앱 실행 (4장 시연)

```bash
uvicorn app.main:app --reload            # http://localhost:8000
python app/simulator.py --rate 5 --limit 500   # 실시간 데이터 주입
```

## 6.7 재현성 확인 결과 (10/6 재실행)

`run_all.py`를 초기 상태에서 재실행하여 전처리→학습→추론→결과 생성이 오류 없이 완료됨을 확인하였다(총 **8.0분**: 학습 471초 + 분석 5초 + 예측 2초).

**재현 일치 검증:**
- 4모델 CV 성능 전부 일치 — F1 0.240 / 0.336 / 0.303 / 0.343, PR-AUC 0.161 / 0.260 / 0.235 / 0.258 (2.2절과 동일).
- 혼동행렬 TP=15·FN=27·FP=177·TN=2174, 3단계 판정 121/359/1913건, 우선순위 36/62/76% 모두 일치(3장과 동일).
- `cv_predictions.csv`는 재실행 전후 **md5 해시 완전 일치**(비트 단위 동일). 모델·예측 파일·그래프도 바이트 동일.
- 유일한 차이: `07_shap_beeswarm.png`의 점 jitter(시각적 차이, SHAP 값 동일) — 내용상 영향 없음.

**5단계 파이프라인 재검증 (rules·advanced 추가 후, 총 19.0분):**
- 공정 규칙 R1~R3(240/262/84건, lift 3.3/2.6/2.7), 구분불가 86%, 불확실성 사분위(0→4.2%), 드리프트 PSI 모두 재현.
- `process_rules.json`·`advanced_metrics.json`·`cv_predictions.csv` **3개 모두 md5 완전 일치** → 신규 단계 포함 완전 재현 확인.

→ 시드 고정·버전 고정으로 **5단계 전체 완전 재현** 확인.

## 6.8 분포 드리프트 & 확장성 (운영 신뢰성)

학습(labeled)과 실제 운영(unlabeled 71,180건)의 피처 분포 차이를 PSI로 측정:

| 피처 | PSI | 판정 |
|---|---|---|
| Cushion_Position | 6.71 | 유의한 드리프트 |
| ewma_resid_norm | 4.35 | 유의 |
| Cycle_Time | 3.75 | 유의 |
| Max_Injection_Pressure | 3.48 | 유의 |
| barrel_temp_range | 1.69 | 유의 |

(PSI<0.1 안정 / 0.1~0.25 보통 / >0.25 유의)

- **해석**: labeled는 소량의 특정 구간 표본이라 운영 데이터와 분포가 크게 다르다(다수 PSI>1). 이는 금형 교차검증 F1=0(2.4절)과 같은 원인(분포·정규화 차이).
- **확장·운영 설계**:
  1. 운영 중 입력 분포를 PSI로 상시 감시 → 임계 초과 시 경보·재학습 트리거.
  2. 신규 금형·설비 도입 시 소량 라벨로 재학습/파인튜닝(4.6절 로드맵 중·장기).
  3. 그룹 단위 검증(StratifiedGroupKFold)으로 '처음 보는 구간'에서의 일반화를 사전 점검.
- **한계 명시**: 현재 unlabeled 예측은 분포 차이로 불확실성이 크므로, 제출 예측은 '순위(우선순위)' 용도로 해석하고 절대 확률은 현장 보정 후 사용 권장.
