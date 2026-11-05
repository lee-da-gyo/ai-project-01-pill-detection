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

from .boxes import iou_matrix, to_xyxy


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
        return {"mAP50_95": 0.0, "mAP50": 0.0, "mAP75": 0.0, "mAP75_95": 0.0, "per_class": pd.DataFrame(), "per_iou": {}}
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
    # IoU 임계값별 mAP: 50과 50-95의 차이가 위치 정밀도에서 오는지 볼 때 쓴다
    per_iou = {round(float(t), 2): float(prec[i][prec[i] > -1].mean()) if (prec[i] > -1).any() else np.nan
               for i, t in enumerate(ev.params.iouThrs)}
    # 대회 평가 지표 mAP@[0.75:0.95]: IoU 0.75 ~ 0.95 (0.05 간격) 5개 기준의 평균
    strict = prec[[i for i, t in enumerate(ev.params.iouThrs) if t >= 0.75 - 1e-9]]
    return {"mAP50_95": float(ev.stats[0]), "mAP50": float(ev.stats[1]), "mAP75": float(ev.stats[2]),
            "mAP75_95": float(strict[strict > -1].mean()) if (strict > -1).any() else float("nan"),
            "per_class": pd.DataFrame(rows), "per_iou": per_iou}


GT_TYPES = ["정답", "위치 오류", "분류 오류", "놓침"]
PRED_TYPES = ["정답 매칭", "중복", "다른 약 위 오분류", "위치 부정확", "배경"]


def classify_errors(pred: pd.DataFrame, gt: pd.DataFrame, conf: float = 0.25, iou_hit: float = 0.5,
                    iou_near: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """신뢰도 conf 이상 예측으로 정답 박스와 예측 박스의 실패 유형을 나눈다 (mAP를 쪼개 보는 진단용, 가법적 손실 분해는 아니다).

    예측 하나는 정답 하나에만, 정답 하나는 예측 하나에만 배정한다 (1:1 매칭). 이미지마다 다음 순서로 처리한다.
    1. 예측을 점수 내림차순으로 보며, 아직 배정되지 않은 정답 중 IoU ≥ iou_hit이고 IoU가 가장 큰 것 하나에 배정한다 (클래스 무관)
       - 같은 약: 정답 박스 "정답", 예측 박스 "정답 매칭"
       - 다른 약: 정답 박스 "분류 오류", 예측 박스 "다른 약 위 오분류"
    2. 남은 정답마다, 아직 배정되지 않은 같은 약 예측 중 iou_near ≤ IoU < iou_hit이고 IoU가 가장 큰 것이 있으면
       정답 박스 "위치 오류", 그 예측 "위치 부정확". 없으면 "놓침" (conf 미만의 저신뢰 예측은 여기서 보지 않는다)
    3. 남은 예측: 같은 약 정답과 IoU ≥ iou_hit → "중복", 다른 약 정답과 IoU ≥ iou_hit → "다른 약 위 오분류",
       어떤 정답과든 IoU ≥ iou_near → "위치 부정확", 그 밖 → "배경"
    예측마다 가장 많이 겹친 정답의 약(near_gt_category_id)을 남긴다. IoU < iou_near이면 NaN이며, 이런 배경 예측은 어느 약 그룹에도 속하지 않는다.
    """
    p_all = pred[pred["score"] >= conf]
    g_rows, p_rows = [], []
    for f, g in gt.groupby("file_name", sort=False):
        g = g.reset_index(drop=True)
        p = p_all[p_all["file_name"] == f].sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
        iou = iou_matrix(to_xyxy(g), to_xyxy(p)) if len(p) else np.zeros((len(g), 0))
        same = g["category_id"].to_numpy()[:, None] == p["category_id"].to_numpy()[None, :]
        g_of = np.full(len(p), -1)  # 예측 → 배정된 정답
        p_of = np.full(len(g), -1)  # 정답 → 배정된 예측
        g_type, p_type = ["놓침"] * len(g), [""] * len(p)
        for j in range(len(p)):  # 1. 점수 순 1:1 매칭
            cand = np.where((p_of < 0) & (iou[:, j] >= iou_hit))[0]
            if cand.size:
                i = int(cand[np.argmax(iou[cand, j])])
                g_of[j], p_of[i] = i, j
                g_type[i], p_type[j] = ("정답", "정답 매칭") if same[i, j] else ("분류 오류", "다른 약 위 오분류")
        for i in np.where(p_of < 0)[0]:  # 2. 남은 정답: 위치 오류 또는 놓침
            cand = np.where((g_of < 0) & same[i] & (iou[i] >= iou_near) & (iou[i] < iou_hit))[0]
            if cand.size:
                j = int(cand[np.argmax(iou[i, cand])])
                g_of[j], p_of[i] = i, j
                g_type[i], p_type[j] = "위치 오류", "위치 부정확"
        for j in np.where(g_of < 0)[0]:  # 3. 남은 예측
            col = iou[:, j]
            if (col[same[:, j]] >= iou_hit).any():
                p_type[j] = "중복"
            elif (col >= iou_hit).any():
                p_type[j] = "다른 약 위 오분류"
            elif col.size and col.max() >= iou_near:
                p_type[j] = "위치 부정확"
            else:
                p_type[j] = "배경"
        for i, r in g.iterrows():
            j = p_of[i]
            g_rows.append({"file_name": f, "category_id": int(r.category_id), "x": r.x, "y": r.y, "w": r.w, "h": r.h,
                           "type": g_type[i], "matched_iou": float(iou[i, j]) if j >= 0 else 0.0,
                           "pred_category_id": int(p.loc[j, "category_id"]) if j >= 0 else np.nan,
                           "pred_score": float(p.loc[j, "score"]) if j >= 0 else np.nan})
        for j, r in p.iterrows():
            col = iou[:, j]
            near = int(np.argmax(col)) if col.size and col.max() >= iou_near else -1
            p_rows.append({"file_name": f, "category_id": int(r.category_id), "score": float(r.score), "type": p_type[j],
                           "max_iou": float(col.max()) if col.size else 0.0,
                           "near_gt_category_id": int(g.loc[near, "category_id"]) if near >= 0 else np.nan})
    return pd.DataFrame(g_rows), pd.DataFrame(p_rows)
