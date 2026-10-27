"""Kaggle 제출 파일 생성과 형식 검사.

제출 컬럼 구성은 configs/submission.yaml 에서 정한다. 대회 형식이 바뀌어도 코드는 그대로 두고 설정만 고친다.
"""

from pathlib import Path

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


def validate_submission(sub: pd.DataFrame, spec: dict, test_ids: set, valid_categories: set) -> dict:
    """제출 전 형식 검사. 문제가 있으면 AssertionError."""
    c = spec["columns"]
    assert list(sub.columns) == [c[k] for k in spec["order"]], "컬럼 순서가 다르다"
    assert sub[c["annotation_id"]].is_unique, "annotation_id 중복"
    assert set(sub[c["image_id"]]) <= test_ids, "테스트에 없는 image_id"
    assert set(sub[c["category_id"]]) <= valid_categories, "학습에 없는 category_id"
    assert (sub[[c["bbox_w"], c["bbox_h"]]] > 0).all().all(), "크기 0 박스"
    assert sub[c["score"]].between(0, 1).all(), "score 범위 오류"
    return {"rows": len(sub), "images_with_pred": sub[c["image_id"]].nunique(), "test_images": len(test_ids),
            "rows_per_image": round(len(sub) / max(sub[c["image_id"]].nunique(), 1), 2)}
