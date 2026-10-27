"""알약 crop 분류기 비교 실험 (VGG16 / ResNet50 / EfficientNet-B0).

같은 fold, 같은 증강, 같은 학습 설정에서 구조만 바꿔 비교한다.
검증은 "학습 fold에 있는 약"만으로 한다 (처음 보는 약은 어떤 분류기도 맞힐 수 없으므로 비교에 의미가 없다).

사용법
    python scripts/train_classifier.py --fold 0 --archs vgg16 resnet50 efficientnet_b0 --epochs 30

산출물
    runs/cls_<arch>_f<k>/best.pt, history.csv    (git 제외)
    reports/experiments/classifiers_f<k>.csv      구조별 지표 비교
    reports/logs/classifiers_f<k>.log
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import logging  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

from pilldet.classifier import crop_boxes, ordered, train_classifier  # noqa: E402
from pilldet.config import SEED, get_paths  # noqa: E402
from pilldet.data import load_clean_annotations  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--archs", nargs="+", default=["vgg16", "resnet50", "efficientnet_b0"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    P = get_paths(REPO_DIR)
    log_path = P.out / "logs" / f"classifiers_f{args.fold}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, "w", "utf-8")], force=True)
    log = logging.getLogger().info

    gt = load_clean_annotations(P.out / "annotations_clean.json")
    folds = pd.read_csv(P.out / "folds.csv")
    class_map = pd.read_csv(P.out / "class_map.csv")
    cat2idx = dict(zip(class_map["category_id"], class_map["yolo_id"]))

    val_files = set(folds.loc[folds["fold"] == args.fold, "file_name"])
    tr = ordered(gt[~gt["file_name"].isin(val_files)])
    va = ordered(gt[gt["file_name"].isin(val_files)])
    va = va[va["category_id"].isin(set(tr["category_id"]))].reset_index(drop=True)  # 학습에 있는 약만 평가

    log(f"fold {args.fold}: 학습 crop {len(tr)}개 ({tr.category_id.nunique()}종) / 검증 crop {len(va)}개 ({va.category_id.nunique()}종, 학습에 있는 약만)")
    tr_crops, va_crops = crop_boxes(tr, P.train_img), crop_boxes(va, P.train_img)
    tr_y, va_y = tr["category_id"].map(cat2idx).tolist(), va["category_id"].map(cat2idx).tolist()

    results = []
    for arch in args.archs:
        out = REPO_DIR / "runs" / f"cls_{arch}_f{args.fold}"
        log(f"== {arch} 학습 시작")
        r = train_classifier(arch, tr_crops, tr_y, va_crops, va_y, len(class_map), out, epochs=args.epochs,
                             batch=args.batch, lr=args.lr, size=args.size, seed=SEED, log=log)
        log(f"== {arch} 최고: epoch {r['epoch']} acc {r['accuracy']:.4f} P {r['macro_precision']:.4f} "
            f"R {r['macro_recall']:.4f} F1 {r['macro_f1']:.4f} logloss {r['log_loss']:.4f} ({r['params_M']}M 파라미터, {r['train_min']}분)")
        exp_dir = P.out / "experiments" / f"cls_{arch}_f{args.fold}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out / "history.csv", exp_dir / "history.csv")
        results.append(r)

    res = pd.DataFrame(results).round(4)
    out_csv = P.out / "experiments" / f"classifiers_f{args.fold}.csv"
    if out_csv.exists():  # 다른 구조를 따로 돌린 결과와 합친다
        old = pd.read_csv(out_csv)
        res = pd.concat([old[~old["arch"].isin(res["arch"])], res], ignore_index=True)
    res.to_csv(out_csv, index=False)
    log(f"\n{res.to_string(index=False)}")


if __name__ == "__main__":
    main()
