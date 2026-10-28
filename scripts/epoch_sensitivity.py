"""전체 데이터 학습 에포크 선택의 민감도 확인.

전체 데이터 학습은 검증 데이터가 없어 에포크를 미리 정해야 한다. 5-Fold의 fold별 best epoch 중앙값을 쓰는데,
이것은 휴리스틱이므로 "에포크를 고정했을 때 각 fold에서 얼마나 손해인가"를 fold별 학습 기록으로 확인한다.

    손해 = (그 fold의 최고 검증 mAP50-95) − (고정 에포크에서의 검증 mAP50-95)

후보 에포크는 fold별 best epoch의 P25 · 중앙값 · P75. 조기 종료로 그 에포크까지 학습하지 않은 fold는 비워 둔다.
한계: fold 학습은 100 에포크 스케줄이라 학습률 감소 시점이 전체 데이터 학습(50·83 에포크)과 다르다.
따라서 "정확한 최적 에포크"가 아니라 "최고점 주변이 얼마나 평평한가"를 보는 진단이다.

사용법
    python scripts/epoch_sensitivity.py e0_baseline e2_yolo11m

산출물
    reports/experiments/epoch_sensitivity.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

from pilldet.config import N_FOLDS, get_paths  # noqa: E402

COL = "metrics/mAP50-95(B)"


def main():
    exps = sys.argv[1:]
    if not exps:
        raise SystemExit("실험 이름을 하나 이상 넣자. 예: python scripts/epoch_sensitivity.py e0_baseline")
    rows = []
    for exp in exps:
        curves = []
        for k in range(N_FOLDS):
            f = REPO_DIR / "runs" / (exp if k == 0 else f"{exp}_f{k}") / "results.csv"
            if not f.exists():
                raise SystemExit(f"{f} 가 없다")
            d = pd.read_csv(f)
            d.columns = d.columns.str.strip()
            curves.append(d.set_index("epoch")[COL])
        best_epochs = [int(c.idxmax()) for c in curves]
        candidates = {"P25": np.percentile(best_epochs, 25), "중앙값": np.median(best_epochs),
                      "P75": np.percentile(best_epochs, 75)}
        for label, e in candidates.items():
            e = int(round(e))
            loss = [round(c.max() - c[e], 4) if e in c.index else None for c in curves]
            done = [v for v in loss if v is not None]
            rows.append({"experiment": exp, "후보": label, "epoch": e, "best_epochs": "/".join(map(str, best_epochs)),
                         "fold별_손해": " / ".join("–" if v is None else f"{v:.3f}" for v in loss),
                         "평균_손해": round(float(np.mean(done)), 4), "최대_손해": round(max(done), 4),
                         "학습한_fold": len(done)})
    res = pd.DataFrame(rows)
    res.to_csv(get_paths(REPO_DIR).out / "experiments" / "epoch_sensitivity.csv", index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
