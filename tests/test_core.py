"""핵심 함수 단위 테스트. 데이터 없이 돌아간다.

    python -m pytest tests -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pilldet.boxes import containment, iou_matrix, out_of_bounds, to_xyxy, xywh_to_yolo, yolo_to_xywh  # noqa: E402
from pilldet.data import camera_angle_of, combo_of, expected_codes  # noqa: E402
from pilldet.split import make_folds  # noqa: E402

FILE = "K-001900-016548-019607-029451_0_2_0_2_70_000_200.png"


# ---------- 파일명 파싱 ----------
def test_parse_file_name():
    assert combo_of(FILE) == "K-001900-016548-019607-029451"
    assert expected_codes(FILE) == [1900, 16548, 19607, 29451]
    assert camera_angle_of(FILE) == 70


# ---------- 박스 ----------
def test_iou_identical_and_disjoint():
    a = np.array([[0, 0, 10, 10]])
    assert iou_matrix(a, a)[0, 0] == pytest.approx(1.0)
    assert iou_matrix(a, np.array([[20, 20, 30, 30]]))[0, 0] == 0


def test_iou_half_overlap():
    # 겹침 50, 합집합 150 → 1/3
    assert iou_matrix([0, 0, 10, 10], [5, 0, 15, 10])[0, 0] == pytest.approx(1 / 3)


def test_iou_empty_inputs():
    assert iou_matrix(np.zeros((0, 4)), np.array([[0, 0, 1, 1]])).shape == (0, 1)


def test_containment_inner_box():
    assert containment([2, 2, 4, 4], [0, 0, 10, 10])[0, 0] == pytest.approx(1.0)


def test_out_of_bounds():
    xyxy = to_xyxy(pd.DataFrame({"x": [6567, 10], "y": [625, 10], "w": [311, 50], "h": [315, 50]}))
    assert out_of_bounds(xyxy, np.array([976, 976]), np.array([1280, 1280])).tolist() == [True, False]


@pytest.mark.parametrize("box", [(567, 625, 311, 315), (0, 0, 976, 1280), (100.5, 200.25, 50, 60)])
def test_yolo_roundtrip(box):
    back = yolo_to_xywh(*xywh_to_yolo(*box, 976, 1280), 976, 1280)
    assert np.allclose(back, box)


def test_yolo_normalized_range():
    cx, cy, w, h = xywh_to_yolo(567, 625, 311, 315, 976, 1280)
    assert all(0 <= v <= 1 for v in (cx, cy, w, h))


# ---------- 분할 ----------
def test_folds_keep_combo_together():
    """같은 조합의 사진은 반드시 같은 fold에 들어가야 한다 (누수 방지)."""
    rows = []
    for c in range(20):
        for angle in (70, 75, 90):
            rows.append({"file_name": f"K-{c:06d}-999999_0_2_0_2_{angle}_000_200.png", "category_id": c % 7,
                         "combo": f"K-{c:06d}-999999", "camera_la": angle})
    folds = make_folds(pd.DataFrame(rows), n_folds=5, seed=0)
    assert folds.groupby("combo")["fold"].nunique().max() == 1
    assert set(folds["fold"]) == set(range(5))
