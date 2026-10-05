"""
전처리→학습→추론→결과 생성을 한 번에 실행한다 (6장 재현성 요건).
실행: python run_all.py
소요 시간: 약 10~15분 (CV 5-fold × 10회 반복)
"""

import subprocess, sys, time

PY = sys.executable  # 현재 인터프리터(가상환경)를 그대로 사용 — 재현성 보장
STEPS = [
    ("1장·2장: 피처 엔지니어링 + 모델 학습", [PY, "-m", "src.train"]),
    ("3장: SHAP·군집화·오류분석",              [PY, "-m", "src.analysis"]),
    ("예측: unlabeled 예측 파일 생성",          [PY, "-m", "src.predict"]),
]


def run_step(label: str, cmd: list):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[오류] {label} 실패 (종료 코드 {result.returncode})")
        sys.exit(result.returncode)
    print(f"\n  완료 ({elapsed:.0f}초)")


if __name__ == "__main__":
    print("=" * 60)
    print("  사출성형 불량 예측 — 전체 파이프라인 실행")
    print("=" * 60)
    total_start = time.time()

    for label, cmd in STEPS:
        run_step(label, cmd)

    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  전체 완료 ({total/60:.1f}분)")
    print(f"  생성된 파일:")
    print(f"    outputs/final_lgbm.pkl       — LightGBM (SHAP 분석용)")
    print(f"    outputs/final_model.pkl      — 앙상블 RF+LGB (예측·앱용)")
    print(f"    outputs/feature_columns.json — 피처 컬럼 순서")
    print(f"    outputs/cv_predictions.csv   — CV holdout 예측 (3장 분석용)")
    print(f"    outputs/predictions_unlabeled.csv — unlabeled 예측 결과")
    print(f"    outputs/0*.png               — 그래프 01~12번")
    print(f"{'='*60}")
    print(f"\n  앱 실행: uvicorn app.main:app --reload")
    print(f"  시뮬레이터: python app/simulator.py --rate 5 --limit 500")
