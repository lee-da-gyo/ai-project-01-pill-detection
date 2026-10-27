"""fold별 Copy-Paste 합성 데이터를 만들고, 합성 이미지를 학습 목록에 더한 yaml을 생성한다.

사용법
    python scripts/make_copy_paste.py --fold 0 --n 400

산출물 (datasets/pill/ 아래, git 제외)
    images/cp_f{k}_XXXX.png, labels/cp_f{k}_XXXX.txt
    fold{k}_train_cp.txt      = fold{k}_train.txt + 합성 이미지
    data_fold{k}_cp.yaml      = 학습만 합성 포함, 검증은 원본 그대로
    reports/experiments/copy_paste_f{k}.csv  합성 박스 기록
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

from pilldet.augment import extract_patches, synthesize  # noqa: E402
from pilldet.config import SEED, get_paths  # noqa: E402
from pilldet.data import load_clean_annotations  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", default="0", help="fold 번호 또는 full (전체 학습용)")
    ap.add_argument("--n", type=int, default=400, help="합성 이미지 수")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    P = get_paths(REPO_DIR)
    t0 = time.time()
    gt = load_clean_annotations(P.out / "annotations_clean.json")
    folds = pd.read_csv(P.out / "folds.csv")
    if args.fold == "full":
        train_files, tag, base_list, val_list = set(folds["file_name"]), "full", "all_train.txt", "fold0_val.txt"
    else:
        k = int(args.fold)
        train_files, tag = set(folds.loc[folds["fold"] != k, "file_name"]), f"f{k}"
        base_list, val_list = f"fold{k}_train.txt", f"fold{k}_val.txt"
    gt_train = gt[gt["file_name"].isin(train_files)]

    class_map = pd.read_csv(P.out / "class_map.csv")
    cat2yolo = dict(zip(class_map["category_id"], class_map["yolo_id"]))
    for old in [*(P.dataset / "images").glob(f"cp_{tag}_*"), *(P.dataset / "labels").glob(f"cp_{tag}_*")]:
        old.unlink()

    patches = extract_patches(gt_train, P.train_img)
    kinds = pd.Series([p.mask_type for p in patches]).value_counts().to_dict()
    synth = synthesize(gt_train, P.train_img, P.dataset / "images", P.dataset / "labels", cat2yolo,
                       n_images=args.n, prefix=f"cp_{tag}", seed=args.seed, patches=patches)
    exp_dir = P.out / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    synth.to_csv(exp_dir / f"copy_paste_{tag}.csv", index=False)

    base = (P.dataset / base_list).read_text(encoding="utf-8").split()
    names = sorted(synth["file_name"].unique())
    (P.dataset / base_list.replace(".txt", "_cp.txt")).write_text(
        "\n".join(base + [f"./images/{n}" for n in names]) + "\n", encoding="utf-8")
    with open(P.dataset / ("data_full.yaml" if tag == "full" else f"data_fold{args.fold}.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.update(train=base_list.replace(".txt", "_cp.txt"), val=val_list)
    out_yaml = P.dataset / ("data_full_cp.yaml" if tag == "full" else f"data_fold{args.fold}_cp.yaml")
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    pasted = synth[synth["source"].str.startswith("paste")]
    before = gt_train["category_id"].value_counts()
    after = before.add(pasted["category_id"].value_counts(), fill_value=0)
    print(f"알약 조각 {len(patches)}개 (마스크: {kinds})")
    print(f"합성 이미지 {len(names)}장, 붙인 알약 {len(pasted)}개 → {out_yaml.name}")
    print(f"클래스별 박스 수 최소 {int(before.min())} → {int(after.min())}, 중앙값 {int(before.median())} → {int(after.median())}")
    print(f"완료 ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
