"""이미 학습한 5-Fold 모델로 검증 예측(OOF)만 다시 만든다 - 학습 없이 추론 설정(TTA 등)을 비교할 때 쓴다.

fold k의 모델(runs/<exp>[_f<k>]/weights/best.pt)로 fold k의 검증 이미지만 예측하므로, 학습 때와 같은 누수 없는 OOF가 된다.
결과는 새 실험 이름(<exp>_<tag>)으로 저장해 analyze_oof.py · compare_oof.py에 그대로 넣을 수 있다.

사용법
    python scripts/predict_oof.py e0_baseline --tta          # → reports/experiments/e0_baseline_tta[_f<k>]/val_predictions.csv
    python scripts/compare_oof.py e0_baseline e0_baseline_tta
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

from pilldet.config import N_FOLDS, get_paths  # noqa: E402
from pilldet.predict import predict_images  # noqa: E402
from pilldet.preprocess import preprocess_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp", help="5-Fold로 학습한 실험 이름 (fold 0 이름)")
    ap.add_argument("--tta", action="store_true", help="Ultralytics TTA (좌우 뒤집기 + 0.83·0.67배 축소, 결과를 합침)")
    args = ap.parse_args()
    if not args.tta:
        raise SystemExit("바꿀 추론 설정이 없다 (--tta)")
    tag = "tta"

    P = get_paths(REPO_DIR)
    folds = pd.read_csv(P.out / "folds.csv")
    class_map = pd.read_csv(P.out / "class_map.csv")
    for k in range(N_FOLDS):
        run = REPO_DIR / "runs" / (args.exp if k == 0 else f"{args.exp}_f{k}")
        snap = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
        if f"data_fold{k}" not in Path(snap["data"]).stem:
            raise SystemExit(f"{run.name}: fold {k} 모델이 아니다 ({snap['data']})")
        files = folds.loc[folds["fold"] == k, "file_name"]
        pred = predict_images(run / "weights" / "best.pt", [P.train_img / f for f in files], class_map,
                              imgsz=int(snap["imgsz"]), augment=args.tta, preprocess=preprocess_of(snap["data"]))
        out = P.out / "experiments" / (f"{args.exp}_{tag}" if k == 0 else f"{args.exp}_{tag}_f{k}")
        out.mkdir(parents=True, exist_ok=True)
        pred.to_csv(out / "val_predictions.csv", index=False)
        print(f"fold {k}: {run.name} → {out.relative_to(REPO_DIR)} ({len(files)}장, 예측 {len(pred):,}개)")


if __name__ == "__main__":
    main()
