"""2단계 모델(검출기 + 분류기) 평가와 단일 YOLO 비교.

1. 오류 분해: 클래스를 무시한 mAP(= 위치만) vs 클래스까지 맞힌 mAP → 오류가 위치에서 오는지 이름에서 오는지
2. 분류기 구조별(VGG16 / ResNet50 / EfficientNet-B0) 2단계 성능
3. YOLO 클래스 예측과 분류기 확률을 섞는 비율(yolo_weight)별 성능

사용법
    python scripts/eval_two_stage.py --det e0_baseline --fold 0

산출물
    reports/experiments/two_stage_f<k>.csv
    reports/logs/two_stage_f<k>.log
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import torch  # noqa: E402

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

from pilldet.classifier import load_classifier  # noqa: E402
from pilldet.config import get_paths  # noqa: E402
from pilldet.data import load_clean_annotations  # noqa: E402
from pilldet.evaluate import coco_map  # noqa: E402
from pilldet.predict import predict_images  # noqa: E402
from pilldet.two_stage import classify_detections, combine, ordered  # noqa: E402


def scores(pred, gt, seen):
    a = coco_map(pred, gt)
    s = coco_map(pred[pred["category_id"].isin(seen)], gt[gt["category_id"].isin(seen)])
    return {"mAP50_95": round(a["mAP50_95"], 4), "mAP50": round(a["mAP50"], 4), "seen_mAP50_95": round(s["mAP50_95"], 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--det", default="e0_baseline", help="검출기 실험 이름 (runs/<det>/weights/best.pt)")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--archs", nargs="+", default=["efficientnet_b0", "resnet50", "vgg16"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    P = get_paths(REPO_DIR)
    log_path = P.out / "logs" / f"two_stage_f{args.fold}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, "w", "utf-8")], force=True)
    log = logging.getLogger().info

    gt = load_clean_annotations(P.out / "annotations_clean.json")
    folds = pd.read_csv(P.out / "folds.csv")
    class_map = pd.read_csv(P.out / "class_map.csv")
    class_ids = class_map.sort_values("yolo_id")["category_id"].values
    val_files = sorted(folds.loc[folds["fold"] == args.fold, "file_name"])
    gt_val = gt[gt["file_name"].isin(val_files)]
    seen = set(gt.loc[~gt["file_name"].isin(val_files), "category_id"])
    weights = REPO_DIR / "runs" / args.det / "weights" / "best.pt"
    paths = [P.train_img / f for f in val_files]
    rows = []

    # 단일 YOLO (기준)
    yolo = predict_images(weights, paths, class_map, imgsz=args.imgsz)
    rows.append({"method": f"YOLO 단독 ({args.det})", **scores(yolo, gt_val, seen)})

    # 오류 분해: 클래스를 모두 같은 값으로 바꾸면 "위치"만 평가된다
    agn = predict_images(weights, paths, class_map, imgsz=args.imgsz, agnostic_nms=True)
    loc = coco_map(agn.assign(category_id=0), gt_val.assign(category_id=0))
    log(f"오류 분해 – 위치만(클래스 무시) mAP50-95 {loc['mAP50_95']:.4f} / mAP50 {loc['mAP50']:.4f}")
    rows.append({"method": "참고: 위치만 평가 (클래스 무시)", "mAP50_95": round(loc["mAP50_95"], 4), "mAP50": round(loc["mAP50"], 4)})

    # 2단계: 클래스 무관 검출 박스 → 분류기
    boxes = ordered(predict_images(weights, paths, class_map, imgsz=args.imgsz, conf=0.01, max_det=20, agnostic_nms=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for arch in args.archs:
        w = REPO_DIR / "runs" / f"cls_{arch}_f{args.fold}" / "best.pt"
        if not w.exists():
            log(f"분류기 가중치 없음, 건너뜀: {w}")
            continue
        prob = classify_detections(boxes, P.train_img, load_classifier(arch, len(class_ids), w, device), device, class_ids)
        for yw in (0.0, 0.3, 0.5):
            pred = combine(boxes, prob, class_ids, top_k=args.top_k, yolo_weight=yw)
            name = f"2단계: YOLO + {arch}" + ("" if yw == 0 else f" (YOLO 클래스 비중 {yw})")
            rows.append({"method": name, **scores(pred, gt_val, seen)})
            log(f"{name}: {rows[-1]}")

    res = pd.DataFrame(rows)
    res.to_csv(P.out / "experiments" / f"two_stage_f{args.fold}.csv", index=False, encoding="utf-8-sig")
    log(f"\n{res.to_string(index=False)}")


if __name__ == "__main__":
    main()
