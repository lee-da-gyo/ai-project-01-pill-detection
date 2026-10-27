"""여러 모델의 예측을 합치는 Weighted Boxes Fusion (WBF).

NMS는 겹치는 박스 중 하나만 남기고 나머지를 버리지만, WBF는 겹치는 박스들의 좌표를 점수로 가중 평균해 하나로 합친다.
여러 모델이 같은 알약을 조금씩 다르게 그렸을 때 위치가 더 정확해지고(mAP50-95에 유리),
일부 모델만 찾은 박스는 점수를 낮춰 오탐의 영향을 줄인다.

참고: Solovyev et al., "Weighted boxes fusion: Ensembling boxes from different object detection models" (2019)
"""

import numpy as np
import pandas as pd

from .boxes import iou_matrix


def _fuse_class(boxes: np.ndarray, scores: np.ndarray, model_ids: np.ndarray, weights: np.ndarray,
                iou_thr: float) -> list[tuple[np.ndarray, float]]:
    order = np.argsort(-scores)
    clusters: list[list[int]] = []
    fused: list[np.ndarray] = []
    for i in order:
        if fused:
            ious = iou_matrix(boxes[i], np.array(fused))[0]
            j = int(np.argmax(ious))
            if ious[j] > iou_thr:
                clusters[j].append(i)
                idx = np.array(clusters[j])
                w = scores[idx] * weights[model_ids[idx]]
                fused[j] = (boxes[idx] * w[:, None]).sum(0) / w.sum()
                continue
        clusters.append([i])
        fused.append(boxes[i].copy())
    out = []
    total_w = weights.sum()
    for idx, box in zip(clusters, fused):
        idx = np.array(idx)
        # 모델마다 가장 높은 점수 하나만 세서, 많은 모델이 동의할수록 높은 점수를 준다
        best = pd.Series(scores[idx] * weights[model_ids[idx]]).groupby(model_ids[idx]).max()
        out.append((box, float(best.sum() / total_w)))
    return out


def weighted_boxes_fusion(preds: list[pd.DataFrame], weights: list[float] | None = None, iou_thr: float = 0.55,
                          skip_score: float = 0.001) -> pd.DataFrame:
    """모델별 예측 테이블(file_name, category_id, x, y, w, h, score) 목록 → 합친 예측 테이블."""
    n = len(preds)
    weights = np.asarray(weights if weights is not None else [1.0] * n, dtype=float)
    allp = pd.concat([p.assign(_m=i) for i, p in enumerate(preds)], ignore_index=True)
    allp = allp[allp["score"] >= skip_score]
    rows = []
    for (f, c), g in allp.groupby(["file_name", "category_id"], sort=False):
        b = np.c_[g["x"], g["y"], g["x"] + g["w"], g["y"] + g["h"]].astype(float)
        for box, s in _fuse_class(b, g["score"].to_numpy(float), g["_m"].to_numpy(), weights, iou_thr):
            rows.append({"file_name": f, "category_id": int(c), "x": box[0], "y": box[1],
                         "w": box[2] - box[0], "h": box[3] - box[1], "score": s})
    return pd.DataFrame(rows, columns=["file_name", "category_id", "x", "y", "w", "h", "score"])
