# 6장 보고서 수치 — 코드 및 재현성

평가 요건: 동일 환경에서 **전처리→학습→추론→결과 생성**까지 자동 실행.

## 6.1 한 번에 실행

```bash
python run_all.py
```

- `run_all.py`가 3단계를 순서대로 호출(서브프로세스는 `sys.executable`로 현재 가상환경 보장):
  1. `src.train` — 피처 엔지니어링 + 4모델 CV 비교 + 금형 교차검증 + 최종 모델 저장 (1·2장)
  2. `src.analysis` — SHAP·상호작용·FN/FP·군집화 (3장)
  3. `src.predict` — 검사 우선순위 곡선 + 3단계 판정 + unlabeled 예측 파일
- 소요 시간: **약 10~15분** (CV 5-fold × 10회 반복 = 50회 학습 × 4모델 + SHAP).

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
| `0*.png` | 그래프 01~12 |

## 6.6 앱 실행 (4장 시연)

```bash
uvicorn app.main:app --reload            # http://localhost:8000
python app/simulator.py --rate 5 --limit 500   # 실시간 데이터 주입
```

## 6.7 재현성 확인 결과

- `run_all.py`를 초기 상태에서 재실행하여 전처리→학습→추론→결과 생성이 오류 없이 완료됨을 확인.
- 시드 고정으로 `cv_predictions.csv` 및 모델 성능 수치가 **최초 산출값과 동일**하게 재현됨.
  (※ 최종 재현 로그/수치 일치 여부는 10/6 재실행 후 본 절에 기입)
