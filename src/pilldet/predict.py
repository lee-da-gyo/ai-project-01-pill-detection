"""학습한 YOLO로 예측해 박스 테이블을 만든다.

반환 형식은 라벨 테이블과 같다 (file_name, category_id, x, y, w, h + score).
그래서 정답과 같은 함수로 그리고, 같은 함수로 평가할 수 있다.
"""

from pathlib import Path

import pandas as pd


def predict_images(weights: Path, image_paths: list[Path], class_map: pd.DataFrame, imgsz: int = 640,
                   conf: float = 0.001, iou: float = 0.7, max_det: int = 100, augment: bool = False,
                   batch: int = 16, device=None, agnostic_nms: bool = False,
                   preprocess: str | None = None) -> pd.DataFrame:
    """이미지 목록을 예측한다. conf를 낮게 두는 이유: mAP는 낮은 점수의 예측까지 순위로 평가하기 때문이다.

    max_det=100: COCO 평가는 이미지당 상위 100개까지만 본다 (maxDets=100). 그 이상은 제출 파일만 키운다.

    agnostic_nms=True: 클래스와 상관없이 겹치는 박스를 하나로 합친다 (2단계 모델에서 "알약 위치"만 얻을 때).

    preprocess: 학습 때와 같은 전처리(예: "clahe")를 적용한 배열을 넣는다. 원본 파일은 바꾸지 않는다.
    """
    from ultralytics import YOLO

    from .preprocess import load_bgr

    model = YOLO(str(weights))
    yolo2cat = dict(zip(class_map["yolo_id"], class_map["category_id"]))
    rows = []
    for i in range(0, len(image_paths), batch):
        chunk = [str(p) for p in image_paths[i:i + batch]]
        source = chunk if preprocess is None else [load_bgr(Path(p), preprocess) for p in chunk]
        for path, r in zip(chunk, model.predict(source, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                                                 augment=augment, device=device, agnostic_nms=agnostic_nms,
                                                 verbose=False)):
            boxes = r.boxes
            for (x1, y1, x2, y2), c, s in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()):
                if x2 - x1 < 1 or y2 - y1 < 1:  # 크기가 1px 미만인 퇴화 박스는 버린다
                    continue
                rows.append({"file_name": Path(path).name, "category_id": int(yolo2cat[int(c)]),
                             "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "score": s})
    return pd.DataFrame(rows, columns=["file_name", "category_id", "x", "y", "w", "h", "score"])


def limit_per_image(pred: pd.DataFrame, max_per_image: int = 4, unique_class: bool = True) -> pd.DataFrame:
    """도메인 지식 후처리: 이미지당 알약 최대 4개, 같은 약은 한 번만 나온다."""
    p = pred.sort_values("score", ascending=False)
    if unique_class:
        p = p.drop_duplicates(["file_name", "category_id"])
    return p.groupby("file_name", group_keys=False).head(max_per_image).reset_index(drop=True)
