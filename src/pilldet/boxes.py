"""박스 좌표 변환과 겹침 계산.

박스 표기
- xywh : [x, y, 너비, 높이] (COCO, 원본 라벨)
- xyxy : [x1, y1, x2, y2] (계산용)
"""

import numpy as np
import pandas as pd


def to_xyxy(df: pd.DataFrame) -> np.ndarray:
    """x, y, w, h 컬럼을 가진 테이블 → (N, 4) xyxy 배열."""
    return np.c_[df["x"], df["y"], df["x"] + df["w"], df["y"] + df["h"]].astype(float)


def _intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    return np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)


def area(b: np.ndarray) -> np.ndarray:
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a(N,4), b(M,4) xyxy → (N, M) IoU 행렬."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    inter = _intersection(a, b)
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def containment(inner: np.ndarray, outer: np.ndarray) -> np.ndarray:
    """inner(M,4)의 면적 중 outer(N,4) 안에 들어간 비율 → (M, N)."""
    inner, outer = np.asarray(inner, float).reshape(-1, 4), np.asarray(outer, float).reshape(-1, 4)
    if len(inner) == 0 or len(outer) == 0:
        return np.zeros((len(inner), len(outer)))
    return _intersection(inner, outer) / (area(inner)[:, None] + 1e-9)


def out_of_bounds(xyxy: np.ndarray, img_w: np.ndarray, img_h: np.ndarray) -> np.ndarray:
    """이미지 범위를 벗어난 박스 여부."""
    return (xyxy[:, 0] < 0) | (xyxy[:, 1] < 0) | (xyxy[:, 2] > img_w) | (xyxy[:, 3] > img_h)


def xywh_to_yolo(x, y, w, h, img_w, img_h) -> tuple[float, float, float, float]:
    """픽셀 xywh → YOLO 정규화 (cx, cy, w, h)."""
    return (x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h


def yolo_to_xywh(cx, cy, w, h, img_w, img_h) -> tuple[float, float, float, float]:
    """YOLO 정규화 (cx, cy, w, h) → 픽셀 xywh."""
    return (cx - w / 2) * img_w, (cy - h / 2) * img_h, w * img_w, h * img_h
