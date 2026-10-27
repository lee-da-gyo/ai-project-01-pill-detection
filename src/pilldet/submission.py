"""Kaggle 제출 파일 생성과 형식 검사.

제출 컬럼 구성은 configs/submission.yaml 에서 정한다. 대회 형식이 바뀌어도 코드는 그대로 두고 설정만 고친다.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_submission_spec(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def image_id_of(file_name: str) -> int:
    """테스트 이미지 '123.png' → 123"""
    return int(Path(file_name).stem)


def to_submission(pred: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """예측 테이블(file_name, category_id, x, y, w, h, score) → 제출 형식."""
    c = spec["columns"]
    p = pred.sort_values(["file_name", "score"], ascending=[True, False]).reset_index(drop=True)
    sub = pd.DataFrame({
        c["annotation_id"]: range(1, len(p) + 1),
        c["image_id"]: p["file_name"].map(image_id_of),
        c["category_id"]: p["category_id"].astype(int),
        c["bbox_x"]: p["x"].round(spec.get("round", 2)),
        c["bbox_y"]: p["y"].round(spec.get("round", 2)),
        c["bbox_w"]: p["w"].round(spec.get("round", 2)),
        c["bbox_h"]: p["h"].round(spec.get("round", 2)),
        c["score"]: p["score"].round(5),
    })
    return sub[[c[k] for k in spec["order"]]]


def _require(ok, msg: str):
    """제출을 막는 검사. assert는 python -O에서 사라지므로 명시적 예외를 쓴다."""
    if not ok:
        raise ValueError(msg)


def validate_submission(sub: pd.DataFrame, spec: dict, test_sizes: dict, valid_categories: set,
                        max_per_image: int = 100, tol: float = 1.0) -> dict:
    """제출 전 형식 검사. 문제가 있으면 ValueError.

    test_sizes: {image_id: (width, height)} – 박스가 이미지 안에 있는지 확인하는 데 쓴다.
    예측이 하나도 없는 테스트 이미지는 오류가 아니지만(그 이미지는 0점) 개수를 반환해 확인할 수 있게 한다.
    """
    c = spec["columns"]
    cols = [c[k] for k in spec["order"]]
    _require(list(sub.columns) == cols, f"컬럼 이름·순서가 다르다: {list(sub.columns)}")
    _require(np.isfinite(sub[cols].to_numpy(dtype=float)).all(), "NaN 또는 Inf가 있다")
    ann = sub[c["annotation_id"]]
    _require(ann.is_unique and ann.min() == 1 and ann.max() == len(sub), "annotation_id는 1부터 빠짐없이 이어져야 한다")
    ids = sub[c["image_id"]]
    _require(set(ids) <= set(test_sizes), "테스트에 없는 image_id")
    _require(set(sub[c["category_id"]]) <= valid_categories, "학습에 없는 category_id")
    x, y, w, h = (sub[c[k]] for k in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"))
    _require(((w > 0) & (h > 0)).all(), "크기 0 박스")
    W = ids.map(lambda i: test_sizes[i][0])
    H = ids.map(lambda i: test_sizes[i][1])
    _require(((x >= -tol) & (y >= -tol) & (x + w <= W + tol) & (y + h <= H + tol)).all(), "이미지 밖으로 나간 박스")
    _require(sub[c["score"]].between(0, 1).all(), "score 범위 오류")
    per_image = ids.value_counts()
    _require(per_image.max() <= max_per_image, f"이미지당 예측이 {max_per_image}개를 넘는다 (최대 {per_image.max()})")
    dup = sub.duplicated([c["image_id"], c["category_id"], c["bbox_x"], c["bbox_y"], c["bbox_w"], c["bbox_h"]]).sum()
    _require(dup == 0, f"완전히 같은 예측이 {dup}개 중복된다")
    return {"rows": len(sub), "images_with_pred": int(ids.nunique()), "test_images": len(test_sizes),
            "images_without_pred": len(set(test_sizes) - set(ids)), "rows_per_image": round(len(sub) / max(ids.nunique(), 1), 2),
            "max_per_image": int(per_image.max())}
