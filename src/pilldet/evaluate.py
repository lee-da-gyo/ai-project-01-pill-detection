"""COCO 방식 mAP 평가 (pycocotools).

YOLO 단일 모델, 2단계 모델, 앙상블, TTA 등 어떤 방식의 예측이든
같은 테이블 형식(file_name, category_id, x, y, w, h, score)이면 같은 기준으로 비교할 수 있다.
"""

import contextlib
import io

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _to_coco(gt: pd.DataFrame):
    files = sorted(gt["file_name"].unique())
    img_id = {f: i + 1 for i, f in enumerate(files)}
    size = gt.groupby("file_name")[["img_w", "img_h"]].first()
    ds = {
        "images": [{"id": img_id[f], "file_name": f, "width": int(size.loc[f, "img_w"]), "height": int(size.loc[f, "img_h"])}
                   for f in files],
        "annotations": [{"id": i, "image_id": img_id[r.file_name], "category_id": int(r.category_id),
                         "bbox": [r.x, r.y, r.w, r.h], "area": r.w * r.h, "iscrowd": 0}
                        for i, r in enumerate(gt.itertuples(), start=1)],
        "categories": [{"id": int(c)} for c in sorted(gt["category_id"].unique())],
    }
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = ds
        coco.createIndex()
    return coco, img_id


def coco_map(pred: pd.DataFrame, gt: pd.DataFrame) -> dict:
    """정답(gt)에 있는 이미지에 대해서만 평가한다. 반환: mAP50_95, mAP50, mAP75, 클래스별 AP 표."""
    coco_gt, img_id = _to_coco(gt)
    p = pred[pred["file_name"].isin(img_id)]
    if p.empty:
        return {"mAP50_95": 0.0, "mAP50": 0.0, "mAP75": 0.0, "per_class": pd.DataFrame()}
    dets = [{"image_id": img_id[r.file_name], "category_id": int(r.category_id), "bbox": [r.x, r.y, r.w, r.h],
             "score": float(r.score)} for r in p.itertuples()]
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(dets)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    # precision: [IoU 임계값 10, recall 101, 클래스, 면적 4, maxDets 3] → 면적 all, maxDets 100
    prec = ev.eval["precision"][:, :, :, 0, -1]
    rows = []
    for k, cid in enumerate(ev.params.catIds):
        pk = prec[:, :, k]
        valid = pk[pk > -1]
        ap50 = pk[0][pk[0] > -1]
        rows.append({"category_id": cid, "AP50_95": float(valid.mean()) if valid.size else np.nan,
                     "AP50": float(ap50.mean()) if ap50.size else np.nan,
                     "n_gt": int((gt["category_id"] == cid).sum())})
    return {"mAP50_95": float(ev.stats[0]), "mAP50": float(ev.stats[1]), "mAP75": float(ev.stats[2]),
            "per_class": pd.DataFrame(rows)}
