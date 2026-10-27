"""2단계 결합: 검출기(YOLO)의 위치 + 분류기의 약 이름.

1. YOLO로 알약 위치를 찾는다 (클래스 무관 NMS → 알약 하나에 박스 하나)
2. 박스마다 crop을 분류기에 넣어 56종 확률을 얻는다
3. 최종 점수 = 검출 신뢰도 × 분류 확률. 박스마다 상위 k개 클래스를 후보로 낸다
   (mAP는 2·3순위 후보도 점수로 쳐 주므로, 분류가 애매한 박스에서 정답 후보를 놓치지 않는다)

YOLO 자체의 클래스 예측과 분류기를 섞는 앙상블 방식도 비교할 수 있게 yolo_weight를 둔다.
"""

import numpy as np
import pandas as pd

from .classifier import build_transforms, crop_boxes, ordered, predict_proba


def classify_detections(boxes: pd.DataFrame, img_dir, model, device, class_ids: np.ndarray,
                        size: int = 224, pad: float = 0.1) -> np.ndarray:
    """검출 박스마다 분류기 확률 (N, C). boxes는 ordered()로 정렬된 상태여야 한다."""
    _, tf_val = build_transforms(size)
    return predict_proba(model, crop_boxes(boxes, img_dir, pad), tf_val, device)


def combine(boxes: pd.DataFrame, cls_prob: np.ndarray, class_ids: np.ndarray, top_k: int = 3,
            yolo_weight: float = 0.0) -> pd.DataFrame:
    """박스별 상위 k개 클래스를 후보로 만든다.

    boxes: 클래스 무관 검출 결과 (file_name, category_id(YOLO 1순위), x, y, w, h, score)
    yolo_weight > 0 이면 YOLO 1순위 클래스에 그만큼 확률을 더해 섞는다 (간단한 앙상블).
    """
    prob = cls_prob.copy()
    if yolo_weight > 0:
        col = {c: i for i, c in enumerate(class_ids)}
        yolo_onehot = np.zeros_like(prob)
        yolo_onehot[np.arange(len(boxes)), boxes["category_id"].map(col).values] = 1
        prob = (1 - yolo_weight) * prob + yolo_weight * yolo_onehot
    top = np.argsort(-prob, axis=1)[:, :top_k]
    rows = []
    for i, r in enumerate(boxes.itertuples()):
        for j in top[i]:
            rows.append({"file_name": r.file_name, "category_id": int(class_ids[j]), "x": r.x, "y": r.y,
                         "w": r.w, "h": r.h, "score": float(r.score * prob[i, j])})
    return pd.DataFrame(rows)


__all__ = ["classify_detections", "combine", "ordered"]
