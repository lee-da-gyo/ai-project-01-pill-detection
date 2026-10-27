"""최종 제출 파일 생성 – 여러 모델의 테스트 예측을 WBF로 합친다.

사용법
    python scripts/make_submission.py --runs final_yolo11s final_yolo11m --name final_ensemble
    python scripts/make_submission.py --runs e0_baseline e0_baseline_f1 e0_baseline_f2 --name cv5_e0 --tta

    --runs : runs/<이름>/weights/ 아래 가중치를 쓴다. 전체 학습(검증 없음)은 last.pt, 그 밖에는 best.pt
    --weights : 모델별 WBF 가중치 (생략하면 모두 1)
    --tta : Ultralytics TTA (좌우 뒤집기 + 크기 변환)

산출물
    submissions/<name>.csv            (git 제외)
    reports/logs/submission_<name>.log
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

from pilldet.config import get_paths  # noqa: E402
from pilldet.data import list_test_images  # noqa: E402
from pilldet.ensemble import weighted_boxes_fusion  # noqa: E402
from pilldet.predict import predict_images  # noqa: E402
from pilldet.submission import load_submission_spec, to_submission, validate_submission  # noqa: E402


def weights_of(run: str) -> tuple[Path, int]:
    """실험 이름 → (가중치 경로, 학습 해상도).

    학습 당시의 설정 스냅샷(runs/<run>/args.yaml, Ultralytics가 저장)을 읽는다. 현재 configs/는 그 뒤에 바뀌었을 수 있다.
    검증 없이 학습한 전체 데이터 모델(val: false)은 last.pt, 그 밖에는 best.pt를 쓴다. 모호하면 추측하지 않고 멈춘다.
    """
    run_dir = REPO_DIR / "runs" / run
    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        raise FileNotFoundError(f"학습 설정 스냅샷이 없다: {args_path}")
    with open(args_path, encoding="utf-8") as f:
        snap = yaml.safe_load(f)
    if snap.get("single_cls"):
        raise ValueError(f"{run}은 1-클래스 검출기라 약 이름을 낼 수 없다")
    full = Path(str(snap["data"])).stem.startswith("data_full")
    if full and snap.get("val", True):
        raise ValueError(f"{run}: 전체 데이터 학습인데 검증이 켜져 있다 → best.pt를 믿을 수 없다")
    w = run_dir / "weights" / ("last.pt" if full else "best.pt")
    if not w.exists():
        raise FileNotFoundError(f"가중치가 없다: {w}")
    return w, int(snap["imgsz"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--iou-thr", type=float, default=0.55)
    ap.add_argument("--conf", type=float, default=0.001, help="mAP 평가는 낮은 점수 예측까지 순위로 쓴다")
    ap.add_argument("--max-det", type=int, default=100, help="COCO maxDets=100")
    args = ap.parse_args()

    P = get_paths(REPO_DIR)
    log_path = P.out / "logs" / f"submission_{args.name}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, "w", "utf-8")], force=True)
    log = logging.getLogger().info

    class_map = pd.read_csv(P.out / "class_map.csv")
    test_paths = list_test_images(P.test_img)
    preds = []
    for run in args.runs:
        w, imgsz = weights_of(run)
        p = predict_images(w, test_paths, class_map, imgsz=imgsz, augment=args.tta, conf=args.conf, max_det=args.max_det)
        log(f"{run}: {w.name}, imgsz {imgsz}, TTA {args.tta} → 예측 {len(p):,}개")
        preds.append(p)
    pred = preds[0] if len(preds) == 1 else weighted_boxes_fusion(preds, args.weights, iou_thr=args.iou_thr)
    # WBF 후에도 이미지당 상위 100개만 남긴다 (COCO maxDets)
    pred = pred.sort_values("score", ascending=False).groupby("file_name", group_keys=False).head(args.max_det)

    spec = load_submission_spec(REPO_DIR / "configs" / "submission.yaml")
    sub = to_submission(pred, spec)
    check = validate_submission(sub, spec, {int(p.stem): Image.open(p).size for p in test_paths}, set(class_map["category_id"]),
                                max_per_image=args.max_det)
    out = REPO_DIR / "submissions" / f"{args.name}.csv"
    out.parent.mkdir(exist_ok=True)
    sub.to_csv(out, index=False)
    log(f"제출 파일 {out.relative_to(REPO_DIR)}: {check}" + ("" if spec.get("verified") else "  ⚠️ 제출 형식 미확인"))


if __name__ == "__main__":
    main()
