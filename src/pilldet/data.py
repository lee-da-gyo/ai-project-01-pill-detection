"""라벨 로딩, 이미지 읽기, 파일명 파싱."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# 분석에 쓰는 이미지 메타데이터 키 (JSON의 images[0] 안에 있다)
META_KEYS = [
    "dl_name", "dl_name_en", "drug_N", "drug_dir", "back_color", "light_color",
    "camera_la", "camera_lo", "size", "drug_shape", "color_class1", "color_class2",
    "print_front", "print_back", "leng_long", "leng_short", "thick", "form_code_name",
]


def load_rgb(path: Path) -> np.ndarray:
    """이미지를 RGB 배열로 읽는다. cv2.imread는 Windows에서 한글 경로를 읽지 못해 PIL을 쓴다."""
    return np.asarray(Image.open(path).convert("RGB"))


def combo_of(file_name: str) -> str:
    """파일명 앞부분의 조합 코드. 예: K-001900-016548-019607-029451"""
    return file_name.split("_")[0]


def expected_codes(file_name: str) -> list[int]:
    """파일명의 조합 코드에 적힌 약 코드 목록 = 사진에 놓인 알약."""
    return sorted(int(c) for c in combo_of(file_name).split("-")[1:])


def camera_angle_of(file_name: str) -> int:
    """파일명에 적힌 카메라 각도. 예: ..._0_2_0_2_70_000_200.png → 70"""
    return int(file_name.split("_")[5])


def list_train_images(img_dir: Path) -> list[str]:
    return sorted(p.name for p in img_dir.glob("*.png"))


def list_test_images(img_dir: Path) -> list[Path]:
    """테스트 이미지를 숫자 순서(1, 2, …, 10)로 정렬한다."""
    return sorted(img_dir.glob("*.png"), key=lambda p: (not p.stem.isdigit(), int(p.stem) if p.stem.isdigit() else p.stem))


def load_annotations(ann_dir: Path) -> pd.DataFrame:
    """흩어진 COCO JSON(파일 하나에 알약 하나)을 박스 단위 테이블로 합친다.

    반환 컬럼: file_name, category_id, cat_name, x, y, w, h, img_w, img_h,
             n_ann_in_json, json_path, META_KEYS…, combo, source
    """
    rows = []
    for p in sorted(Path(ann_dir).rglob("*.json")):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        img = d["images"][0]
        for a in d["annotations"]:
            x, y, w, h = a["bbox"]
            rows.append({
                "file_name": img["file_name"],
                "category_id": a["category_id"],
                "cat_name": d["categories"][0]["name"],
                "x": float(x), "y": float(y), "w": float(w), "h": float(h),
                "img_w": img["width"], "img_h": img["height"],
                "n_ann_in_json": len(d["annotations"]),
                "json_path": str(p.relative_to(ann_dir)),
                **{k: img.get(k) for k in META_KEYS},
            })
    df = pd.DataFrame(rows)
    df["combo"] = df["file_name"].map(combo_of)
    df["source"] = "original"
    return df


def class_names(df: pd.DataFrame) -> pd.DataFrame:
    """category_id → 한글·영문 약품명 표."""
    return df.drop_duplicates("category_id").set_index("category_id")[["dl_name", "dl_name_en"]].sort_index()
