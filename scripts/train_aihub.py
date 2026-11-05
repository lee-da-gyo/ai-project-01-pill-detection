"""AI Hub 데이터를 더한 학습과 대회 fold 0 검증을 실행한다.

사용법
    python scripts/train_aihub.py --config configs/a1_aihub_yolo11s.yaml
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

from pilldet.config import get_paths  # noqa: E402
from pilldet.data import load_clean_annotations  # noqa: E402
from pilldet.evaluate import coco_map  # noqa: E402
from pilldet.predict import predict_images  # noqa: E402
from pilldet.train import load_config, train  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    P = get_paths(REPO_DIR)
    cfg = load_config(args.config)
    run_dir = REPO_DIR / "runs" / cfg["name"]
    if not args.skip_train:
        overrides = {"val": False} if "full" in cfg["data"] else {}  # 전체 학습은 검증 없이 last.pt를 쓴다
        run_dir = train(cfg, REPO_DIR / "datasets" / "aihub", REPO_DIR / "runs", **overrides)
    if "full" in cfg["data"]:
        print("전체 데이터 학습은 검증 없이 끝낸다")
        return
    class_map = pd.read_csv(P.out / "class_map_aihub.csv")
    comp = load_clean_annotations(P.out / "annotations_clean.json")
    extra = pd.read_csv(REPO_DIR / "datasets/aihub/boxes.csv")
    for title, list_name, gt in [("대회 fold 0", "comp_fold0_val.txt", comp), ("추가 데이터 검증 조합", "aihub_holdout_val.txt", extra)]:
        paths = [Path(s) for s in (REPO_DIR / "datasets/aihub" / list_name).read_text(encoding="utf-8").splitlines() if s.strip()]
        names = {p.name for p in paths}
        pred = predict_images(run_dir / "weights" / "best.pt", paths, class_map, imgsz=cfg["imgsz"])
        r = coco_map(pred, gt[gt.file_name.isin(names)])
        print(title, len(paths), "장", {k: round(v, 4) for k, v in r.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
