"""여러 모델의 예측을 합치는 Weighted Boxes Fusion (WBF).

NMS는 겹치는 박스 중 하나만 남기고 나머지를 버리지만, WBF는 겹치는 박스들의 좌표를 점수로 가중 평균해 하나로 합친다.
여러 모델이 같은 알약을 조금씩 다르게 그렸을 때 위치가 더 정확해지고(mAP50-95에 유리),
일부 모델만 찾은 박스는 점수를 낮춰 오탐의 영향을 줄인다.

원 논문과 저자 공개 구현(conf_type="avg")의 규칙을 따른다.
1. 모델 가중치를 곱한 점수(score × weight)로 박스를 내림차순 정렬한다
2. 같은 클래스에서 합쳐진 박스와 IoU가 임계값을 넘으면 그 묶음(cluster)에 넣고, 좌표를 가중 점수로 다시 평균한다
3. 묶음 점수 = 가중 점수 평균 × min(모델 수, 묶음 박스 수) / 가중치 합

참고: Solovyev et al., "Weighted boxes fusion: Ensembling boxes from different object detection models" (2019)
"""

import numpy as np
import pandas as pd

from .boxes import iou_matrix


def _fuse_class(boxes: np.ndarray, wscores: np.ndarray, n_models: int, weight_sum: float,
                iou_thr: float) -> list[tuple[np.ndarray, float]]:
    """boxes(N,4) xyxy, wscores = score × 모델 가중치."""
    clusters: list[list[int]] = []
    fused: list[np.ndarray] = []
    for i in np.argsort(-wscores, kind="stable"):
        if fused:
            ious = iou_matrix(boxes[i], np.array(fused))[0]
            j = int(np.argmax(ious))
            if ious[j] > iou_thr:
                clusters[j].append(i)
                idx = np.array(clusters[j])
                fused[j] = (boxes[idx] * wscores[idx, None]).sum(0) / wscores[idx].sum()
                continue
        clusters.append([i])
        fused.append(boxes[i].copy())
    return [(box, float(wscores[idx].mean() * min(n_models, len(idx)) / weight_sum))
            for idx, box in zip(map(np.array, clusters), fused)]


def weighted_boxes_fusion(preds: list[pd.DataFrame], weights: list[float] | None = None, iou_thr: float = 0.55,
                          skip_score: float = 0.001) -> pd.DataFrame:
    """모델별 예측 테이블(file_name, category_id, x, y, w, h, score) 목록 → 합친 예측 테이블."""
    n = len(preds)
    weights = np.asarray(weights if weights is not None else [1.0] * n, dtype=float)
    if len(weights) != n:
        raise ValueError(f"가중치 개수({len(weights)})와 모델 수({n})가 다르다")
    if (weights <= 0).any():
        raise ValueError("가중치는 모두 양수여야 한다")
    allp = pd.concat([p.assign(_w=weights[i]) for i, p in enumerate(preds)], ignore_index=True)
    allp = allp[allp["score"] >= skip_score]
    rows = []
    for (f, c), g in allp.groupby(["file_name", "category_id"], sort=False):
        b = np.c_[g["x"], g["y"], g["x"] + g["w"], g["y"] + g["h"]].astype(float)
        ws = (g["score"] * g["_w"]).to_numpy(float)
        for box, s in _fuse_class(b, ws, n, weights.sum(), iou_thr):
            rows.append({"file_name": f, "category_id": int(c), "x": box[0], "y": box[1],
                         "w": box[2] - box[0], "h": box[3] - box[1], "score": s})
    return pd.DataFrame(rows, columns=["file_name", "category_id", "x", "y", "w", "h", "score"])
