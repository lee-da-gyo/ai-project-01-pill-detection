"""Copy-Paste 증강 – 희귀 약을 다양한 배치·조합으로 합성한다.

EDA 결론: 56종 중 29종이 박스 10개 미만이고, 17종은 같은 장면을 각도만 바꾼 3장뿐이다.
베이스라인(E0) 결론: 학습에 있는 약은 거의 완벽히 맞히므로, 남은 과제는 희귀 약이 새로운 배치에서도 맞는가이다.

방법
1. 학습 이미지에서 알약을 오려 낸다. 배경 차분(Lab 거리 + Otsu)으로 알약 윤곽 마스크를 만들고,
   분할이 실패하면(반투명 캡슐 등) 박스에 내접하는 타원 마스크를 쓴다. 가장자리는 부드럽게(feathering) 섞는다.
2. 다른 학습 이미지의 빈 자리에 붙인다. 기존 알약과 겹치지 않게 놓고, 90° 회전·뒤집기·±10% 크기 변화를 준다.
3. 붙일 약은 희귀할수록 자주 뽑는다 (확률 ∝ 1/√빈도).

누수 방지: fold k의 합성 이미지는 fold k의 학습 이미지만으로 만든다.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .boxes import iou_matrix, xywh_to_yolo
from .data import load_rgb


@dataclass
class PillPatch:
    category_id: int
    rgb: np.ndarray  # (h, w, 3) 라벨 박스 영역
    alpha: np.ndarray  # (h, w) 0~1, 알약 = 1
    source: str  # 원본 파일명
    mask_type: str  # "segment" 또는 "ellipse"


def pill_mask(crop: np.ndarray, min_fill: float = 0.35, max_fill: float = 0.95, min_span: float = 0.8) -> tuple[np.ndarray, str]:
    """라벨 박스 crop에서 알약 마스크를 만든다. 테두리 픽셀의 중앙값을 배경색으로 본다.

    라벨 박스는 알약을 거의 꽉 채우므로, 마스크가 박스 가로·세로의 min_span 미만만 덮으면
    알약의 일부만 잡힌 것(흰 알약 등)으로 보고 타원 마스크로 대체한다. 잘린 알약에 전체 박스 라벨이 붙는 것을 막는다.
    """
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    dist = np.linalg.norm(lab - np.median(border, axis=0), axis=2)
    dist = cv2.GaussianBlur(dist, (5, 5), 0)
    _, m = cv2.threshold(np.clip(dist, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = np.where(lbl == big, 255, 0).astype(np.uint8)
        # 알약은 볼록한 모양이다 → 볼록 껍질로 각인·반사로 생긴 구멍과 오목한 틈을 채운다
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(np.vstack(contours))
        m = cv2.fillConvexPoly(np.zeros_like(m), hull, 255)
    fill = (m > 0).mean()
    ys, xs = np.nonzero(m)
    span_ok = len(xs) > 0 and (np.ptp(xs) + 1) / m.shape[1] >= min_span and (np.ptp(ys) + 1) / m.shape[0] >= min_span
    if min_fill <= fill <= max_fill and span_ok:
        return m.astype(np.float32) / 255, "segment"
    h, w = crop.shape[:2]  # 분할 실패 → 박스에 내접하는 타원
    e = cv2.ellipse(np.zeros((h, w), np.uint8), (w // 2, h // 2), (int(w * 0.46), int(h * 0.46)), 0, 0, 360, 255, -1)
    return e.astype(np.float32) / 255, "ellipse"


def extract_patches(gt: pd.DataFrame, img_dir: Path) -> list[PillPatch]:
    patches = []
    for f, g in gt.groupby("file_name"):
        img = load_rgb(Path(img_dir) / f)
        for r in g.itertuples():
            x1, y1 = int(round(r.x)), int(round(r.y))
            x2, y2 = int(round(r.x + r.w)), int(round(r.y + r.h))
            crop = img[y1:y2, x1:x2]
            if crop.shape[0] < 20 or crop.shape[1] < 20:
                continue
            alpha, kind = pill_mask(crop)
            alpha = cv2.GaussianBlur(alpha, (7, 7), 0)  # 가장자리 feathering
            patches.append(PillPatch(int(r.category_id), crop.copy(), alpha, f, kind))
    return patches


def _transform(p: PillPatch, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    rgb, a = p.rgb, p.alpha
    k = int(rng.integers(4))
    rgb, a = np.rot90(rgb, k), np.rot90(a, k)
    if rng.random() < 0.5:
        rgb, a = rgb[:, ::-1], a[:, ::-1]
    s = rng.uniform(0.9, 1.1)
    size = (max(20, int(rgb.shape[1] * s)), max(20, int(rgb.shape[0] * s)))
    return cv2.resize(np.ascontiguousarray(rgb), size, interpolation=cv2.INTER_LINEAR), \
        cv2.resize(np.ascontiguousarray(a), size, interpolation=cv2.INTER_LINEAR)


def _find_spot(boxes: np.ndarray, w: int, h: int, W: int, H: int, rng, margin: int = 15, tries: int = 60):
    """기존 박스와 margin 이상 떨어진 빈 자리를 찾는다. 이미지 가장자리(비네팅)는 피한다."""
    edge = int(0.06 * min(W, H))
    for _ in range(tries):
        x = int(rng.integers(edge, max(edge + 1, W - w - edge)))
        y = int(rng.integers(edge, max(edge + 1, H - h - edge)))
        cand = np.array([[x - margin, y - margin, x + w + margin, y + h + margin]], float)
        if len(boxes) == 0 or iou_matrix(cand, boxes).max() == 0:
            return x, y
    return None


def synthesize(gt_train: pd.DataFrame, img_dir: Path, out_img_dir: Path, out_lbl_dir: Path, cat2yolo: dict,
               n_images: int, prefix: str, seed: int = 42, paste_per_image: tuple = (1, 3),
               patches: list[PillPatch] | None = None) -> pd.DataFrame:
    """합성 이미지와 YOLO 라벨을 저장하고, 합성 결과 박스 테이블을 반환한다."""
    rng = np.random.default_rng(seed)
    patches = patches or extract_patches(gt_train, img_dir)
    by_cls: dict[int, list[PillPatch]] = {}
    for p in patches:
        by_cls.setdefault(p.category_id, []).append(p)
    freq = gt_train["category_id"].value_counts()
    classes = np.array(sorted(by_cls))
    prob = 1 / np.sqrt(freq.loc[classes].values.astype(float))
    prob /= prob.sum()

    bases = sorted(gt_train["file_name"].unique())
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_images):
        base = bases[int(rng.integers(len(bases)))]
        img = load_rgb(Path(img_dir) / base).astype(np.float32)
        H, W = img.shape[:2]
        g = gt_train[gt_train["file_name"] == base]
        boxes = np.c_[g["x"], g["y"], g["x"] + g["w"], g["y"] + g["h"]].astype(float)
        labels = [(int(r.category_id), r.x, r.y, r.w, r.h, "original") for r in g.itertuples()]
        present = set(g["category_id"])

        for _ in range(int(rng.integers(paste_per_image[0], paste_per_image[1] + 1))):
            choices = [c for c in classes if c not in present]  # 같은 약은 한 이미지에 한 번만 (실제 데이터와 동일)
            if not choices:
                break
            pc = prob[np.isin(classes, choices)]
            cid = int(rng.choice(choices, p=pc / pc.sum()))
            patch = by_cls[cid][int(rng.integers(len(by_cls[cid])))]
            rgb, a = _transform(patch, rng)
            h, w = a.shape
            spot = _find_spot(boxes, w, h, W, H, rng)
            if spot is None:
                continue
            x, y = spot
            region = img[y:y + h, x:x + w]
            img[y:y + h, x:x + w] = a[..., None] * rgb + (1 - a[..., None]) * region
            boxes = np.vstack([boxes, [x, y, x + w, y + h]])
            labels.append((cid, x, y, w, h, f"paste:{patch.mask_type}"))
            present.add(cid)

        name = f"{prefix}_{i:04d}.png"
        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(out_img_dir / name, compress_level=1)
        lines = [f"{cat2yolo[c]} " + " ".join(f"{v:.6f}" for v in xywh_to_yolo(x, y, w, h, W, H))
                 for c, x, y, w, h, _ in labels]
        (out_lbl_dir / name.replace(".png", ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
        rows += [{"file_name": name, "base": base, "category_id": c, "x": x, "y": y, "w": w, "h": h,
                  "img_w": W, "img_h": H, "source": s} for c, x, y, w, h, s in labels]
    return pd.DataFrame(rows)
