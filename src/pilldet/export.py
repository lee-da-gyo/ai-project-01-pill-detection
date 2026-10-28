"""YOLO / COCO 내보내기와 왕복 검증.

YOLO 데이터셋 구조
    datasets/pill/
    ├── images/*.png              원본 링크 또는 복사
    ├── labels/*.txt              class cx cy w h (0~1 정규화)
    ├── fold{k}_train.txt / fold{k}_val.txt
    ├── data_fold{k}.yaml         K-Fold 실험용
    └── data_full.yaml            최종 학습용 (전체 학습, val은 fold0 – 모니터링 전용)

Ultralytics는 이미지 경로의 /images/를 /labels/로 바꿔 라벨을 찾으므로 이 구조를 따른다.
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .boxes import xywh_to_yolo, yolo_to_xywh


def build_class_map(clean: pd.DataFrame) -> pd.DataFrame:
    """category_id(약 코드) 순으로 yolo_id 0~N-1을 붙인다. 제출 시 이 표로 원래 코드로 되돌린다."""
    cm = (clean.groupby("category_id")
          .agg(dl_name=("dl_name", "first"), dl_name_en=("dl_name_en", "first"),
               n_boxes=("file_name", "size"), n_images=("file_name", "nunique"), n_combos=("combo", "nunique"))
          .reset_index().sort_values("category_id").reset_index(drop=True))
    cm.insert(0, "yolo_id", range(len(cm)))
    return cm


def _place_image(src: Path, dst: Path) -> str:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    try:
        os.symlink(src, dst)
        return "symlink"
    except OSError:  # Windows 일반 권한에서는 심볼릭 링크가 막혀 있다 → 복사
        shutil.copy2(src, dst)
        return "copy"


def export_yolo(clean: pd.DataFrame, folds: pd.DataFrame, class_map: pd.DataFrame,
                img_dir: Path, dataset_dir: Path) -> dict:
    """YOLO 라벨, fold별 목록, yaml을 만든다. 폴더는 지우지 않고 내용만 새로 쓴다 (Windows 권한 문제 회피)."""
    img_out, lbl_out = dataset_dir / "images", dataset_dir / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for stale in [*lbl_out.glob("*.txt"), *dataset_dir.glob("*.txt"), *dataset_dir.glob("*.yaml")]:
        stale.unlink()

    modes = {_place_image(img_dir / f, img_out / f) for f in folds["file_name"]}

    cat2yolo = dict(zip(class_map["category_id"], class_map["yolo_id"]))
    for f, g in clean.groupby("file_name"):
        lines = []
        for r in g.assign(yolo_id=g["category_id"].map(cat2yolo)).sort_values("yolo_id").itertuples():
            cx, cy, w, h = xywh_to_yolo(r.x, r.y, r.w, r.h, r.img_w, r.img_h)
            lines.append(f"{r.yolo_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (lbl_out / f.replace(".png", ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 학습 그래프에서 글자가 깨지지 않게 클래스 이름은 영문명을 쓴다.
    # 원본에 영문명이 빈 약(32310 글리아타민연질캡슐)이 있어, 그때는 ASCII인 약 코드로 대신한다
    names = {int(r.yolo_id): (str(r.dl_name_en).strip() if pd.notna(r.dl_name_en) and str(r.dl_name_en).strip()
                              else f"id_{r.category_id}") for r in class_map.itertuples()}

    def write_list(name, files):
        (dataset_dir / name).write_text("\n".join(f"./images/{f}" for f in files) + "\n", encoding="utf-8")
        return name

    def write_yaml(name, train, val):
        cfg = {"path": str(dataset_dir), "train": train, "val": val, "nc": len(names), "names": names}
        with open(dataset_dir / name, "w", encoding="utf-8") as fp:
            yaml.safe_dump(cfg, fp, allow_unicode=True, sort_keys=False)

    for k in sorted(folds["fold"].unique()):
        write_yaml(f"data_fold{k}.yaml",
                   write_list(f"fold{k}_train.txt", folds.loc[folds["fold"] != k, "file_name"]),
                   write_list(f"fold{k}_val.txt", folds.loc[folds["fold"] == k, "file_name"]))
    write_yaml("data_full.yaml", write_list("all_train.txt", folds["file_name"]), "fold0_val.txt")

    return {"images": len(list(img_out.iterdir())), "labels": len(list(lbl_out.glob("*.txt"))),
            "link_mode": "/".join(sorted(modes)), "files": sorted(p.name for p in dataset_dir.iterdir() if p.is_file())}


def export_coco(clean: pd.DataFrame, class_map: pd.DataFrame, train_files: list[str], path: Path) -> dict:
    """원래 약 코드를 유지한 COCO 정제본. torchvision 모델과 pycocotools 평가에 쓴다."""
    size = clean.groupby("file_name")[["img_w", "img_h"]].first()
    img_id = {f: i + 1 for i, f in enumerate(train_files)}
    coco = {
        "images": [{"id": img_id[f], "file_name": f, "width": int(size.loc[f, "img_w"]),
                    "height": int(size.loc[f, "img_h"])} for f in train_files],
        "annotations": [{"id": i, "image_id": img_id[r.file_name], "category_id": int(r.category_id),
                         "bbox": [round(r.x, 1), round(r.y, 1), round(r.w, 1), round(r.h, 1)],
                         "area": round(r.w * r.h, 1), "iscrowd": 0}
                        for i, r in enumerate(clean.sort_values(["file_name", "category_id"]).itertuples(), start=1)],
        "categories": [{"id": int(r.category_id), "name": r.dl_name, "supercategory": "pill"}
                       for r in class_map.itertuples()],
    }
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(coco, fp, ensure_ascii=False)
    return {k: len(v) for k, v in coco.items()}


def verify_yolo_roundtrip(dataset_dir: Path, clean: pd.DataFrame, class_map: pd.DataFrame) -> float:
    """저장한 YOLO 라벨을 다시 읽어 픽셀 좌표로 되돌리고 정제 라벨과 비교한다. 최대 좌표 오차(px)를 반환."""
    yolo2cat = dict(zip(class_map["yolo_id"], class_map["category_id"]))
    size = clean.groupby("file_name")[["img_w", "img_h"]].first()
    back = []
    for txt in (dataset_dir / "labels").glob("*.txt"):
        f = txt.stem + ".png"
        W, H = size.loc[f]
        for line in txt.read_text(encoding="utf-8").strip().splitlines():
            c, cx, cy, w, h = line.split()
            cx, cy, w, h = map(float, (cx, cy, w, h))
            assert 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1, f"정규화 범위 오류: {txt.name}"
            x, y, bw, bh = yolo_to_xywh(cx, cy, w, h, W, H)
            back.append({"file_name": f, "category_id": yolo2cat[int(c)], "x_b": x, "y_b": y, "w_b": bw, "h_b": bh})
    m = pd.DataFrame(back).merge(clean, on=["file_name", "category_id"], how="outer", indicator=True)
    assert (m["_merge"] == "both").all(), "라벨 개수가 맞지 않는다"
    return float(np.abs(m[["x_b", "y_b", "w_b", "h_b"]].values - m[["x", "y", "w", "h"]].values).max())
