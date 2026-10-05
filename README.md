# 사출성형 공정데이터 기반 품질불량 사전예측 및 검사 우선순위 결정

제6회 K-인공지능 제조데이터 분석 경진대회 (과제 ①). 최종 검사 이전 공정 데이터만으로
사출 1회(shot)의 불량 확률을 예측하고, 위험도 순으로 **검사 우선순위**를 자동 결정한다.

## 핵심 결과

- 학습 2,393 shot, 불량 42개(1.75%, 1:56 불균형)
- 최종 모델 **앙상블(RF+LightGBM)** — F1 **0.343**(베이스라인 대비 +43%), PR-AUC **0.258**(무작위 14.3배)
- **검사량 20%로 불량 62% 포착**(무작위 3.1배), 30%로 76% 포착
- 전통 SPC + EWMA 추세 신호를 ML 피처로 결합 → 상위 20% 포착률 +10%p

## 실행 환경

- Python 3.9 (가상환경 `.venv`)
- 패키지: `requirements.txt` (버전 고정). scikit-learn 1.6.1 / lightgbm 4.6.0 / shap 0.49.1 / fastapi 0.128.8

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 실행 순서

### 1) 전체 파이프라인 (전처리→학습→추론→결과) — 약 10~15분

```bash
python run_all.py
```

생성물: `outputs/final_model.pkl`, `outputs/cv_predictions.csv`,
`outputs/predictions_unlabeled.csv`(제출용 예측), `outputs/0*.png`(그래프 01~12)

### 2) 실시간 대시보드 (4장 현장 활용 시연)

```bash
uvicorn app.main:app --reload                   # http://localhost:8000
python app/simulator.py --rate 5 --limit 500    # 다른 터미널에서 실시간 데이터 주입
```

대시보드 3페이지: **실시간 모니터링 / 모델 성능 / 검사 우선순위**
(탭 딥링크: `/#live`, `/#model`, `/#prio`)

## 폴더 구조

```
├── data/            # KAMP 원본 CSV (labeled 학습 / unlabeled 예측 대상)
├── src/             # features · train · analysis · predict
├── app/             # FastAPI 서버 + 대시보드 + 시뮬레이터
├── outputs/         # 모델 · 예측파일 · 그래프 · report_notes(장별 수치)
├── run_all.py       # 전체 파이프라인 1회 실행
└── requirements.txt
```

## 재현성

- 모든 난수 `SEED=42` 고정, `run_all.py`는 `sys.executable`로 현재 가상환경을 사용.
- 동일 환경에서 재실행 시 모델 성능·예측 결과가 동일하게 재현된다.

## 데이터

- KAMP 제공 「사출성형기 AI 데이터셋」. 용량·공개 제한으로 저장소에서 제외(`.gitignore`).
  `data/1. 사출성형기 AI 데이터셋/` 아래에 원본 CSV를 두고 실행한다.
- 외부 데이터 미사용.
