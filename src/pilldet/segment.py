"""배경 차분 기반 알약 영역 분할 (고전 영상처리).

라벨을 검증하려면 라벨과 독립적인 기준이 필요하다. 배경이 단색에 가깝다는 EDA 결과를 이용한다.

1. 1/4 축소 (속도, 잡음 감소)
2. Lab 색공간 변환 – 사람 눈의 색 차이와 비슷한 거리
3. 큰 median blur로 배경만 남긴 이미지 추정 (알약보다 큰 커널)
4. 원본과 배경의 색 거리 → Otsu 이진화 → 모폴로지로 잡음 제거
5. 연결 요소 → 박스. 가장자리에 닿는 영역(비네팅)과 작은 영역은 제외

한계: 반투명 연질캡슐, 배경과 색이 비슷한 알약은 일부만 잡힌다.
그래서 결과를 정답으로 쓰지 않고, 라벨을 의심할 근거와 보완 후보로만 쓴다.
"""

from dataclasses import dataclass

import cv2
import numpy as np

SEG_SCALE = 4


@dataclass
class Segmentation:
    boxes: np.ndarray  # (K, 4) 원본 좌표 xyxy
    mask: np.ndarray  # 축소 해상도 전경 마스크 (0/255)
    dist: np.ndarray  # 축소 해상도 배경과의 색 거리
    scale: int = SEG_SCALE

    def fg_ratio(self, box: np.ndarray) -> float:
        """박스(원본 좌표) 안에서 전경 픽셀이 차지하는 비율. 0에 가까우면 빈 배경이다."""
        x1, y1, x2, y2 = np.clip((np.asarray(box) / self.scale).round().astype(int), 0, None)
        region = self.mask[y1:y2, x1:x2]
        return float((region > 0).mean()) if region.size else 0.0


def segment_pills(img_rgb: np.ndarray, scale: int = SEG_SCALE, blur: int = 101, min_area: int = 400) -> Segmentation:
    small = cv2.resize(img_rgb, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
    bg = np.dstack([cv2.medianBlur(lab[..., i], blur) for i in range(3)])
    dist = np.linalg.norm(lab.astype(np.float32) - bg.astype(np.float32), axis=2)
    dist = cv2.GaussianBlur(dist, (5, 5), 0)
    _, mask = cv2.threshold(np.clip(dist, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    H, W = mask.shape
    boxes = []
    for x, y, w, h, a in stats[1:]:
        if a < min_area or x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        boxes.append([x * scale, y * scale, (x + w) * scale, (y + h) * scale])
    return Segmentation(np.array(boxes, dtype=float).reshape(-1, 4), mask, dist, scale)
