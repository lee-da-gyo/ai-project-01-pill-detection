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
        return {"mAP50_95": 0.0, "mAP50": 0.0, "mAP75": 0.0, "per_class": pd.DataFrame(), "per_iou": {}}
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
    return {"mAP50_95": float(ev.stats[0]), "mAP50": float(ev.stats[1]), "mAP75": float(ev.stats[2]),
            "per_class": pd.DataFrame(rows), "per_iou": per_iou}


GT_TYPES = ["정답", "위치 오류", "분류 오류", "놓침"]
PRED_TYPES = ["정답 매칭", "중복", "다른 약 위 오분류", "위치 부정확", "배경"]


def classify_errors(pred: pd.DataFrame, gt: pd.DataFrame, conf: float = 0.25, iou_hit: float = 0.5,
                    iou_near: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """신뢰도 conf 이상 예측으로 정답 박스와 예측 박스의 실패 유형을 나눈다 (mAP를 쪼개 보는 진단용).

    정답 박스 기준 (우선순위 순)
    - 정답: 같은 클래스 예측과 IoU ≥ iou_hit
    - 위치 오류: 같은 클래스 예측과 iou_near ≤ IoU < iou_hit
    - 분류 오류: 다른 클래스 예측과 IoU ≥ iou_hit (가장 점수가 높은 예측의 클래스를 기록)
    - 놓침: 위 어느 것도 아님
    예측 박스 기준
    - 정답 매칭: 같은 클래스 정답과 IoU ≥ iou_hit이고, 그 정답에 겹친 같은 클래스 예측 중 점수가 가장 높다
    - 중복: 같은 클래스 정답과 IoU ≥ iou_hit이지만 더 높은 점수의 예측이 이미 그 정답을 차지했다
    - 다른 약 위 오분류: 다른 클래스 정답과 IoU ≥ iou_hit
    - 위치 부정확: 어떤 정답과도 iou_near ≤ 최대 IoU < iou_hit
    - 배경: 모든 정답과 IoU < iou_near
    """
    p_all = pred[pred["score"] >= conf].reset_index(drop=True)
    g_rows, p_rows = [], []
    for f, g in gt.groupby("file_name", sort=False):
        g = g.reset_index(drop=True)
        p = p_all[p_all["file_name"] == f].sort_values("score", ascending=False).reset_index(drop=True)
        iou = iou_matrix(to_xyxy(g), to_xyxy(p)) if len(p) else np.zeros((len(g), 0))
        same = g["category_id"].to_numpy()[:, None] == p["category_id"].to_numpy()[None, :]
        for i, r in g.iterrows():
            s_iou = np.where(same[i], iou[i], 0.0)
            o_hit = np.where(~same[i] & (iou[i] >= iou_hit))[0]
            best = float(s_iou.max()) if s_iou.size else 0.0
            if best >= iou_hit:
                t, pc, sc = "정답", r.category_id, float(p.loc[int(np.argmax(s_iou)), "score"])
            elif best >= iou_near:
                t, pc, sc = "위치 오류", r.category_id, float(p.loc[int(np.argmax(s_iou)), "score"])
            elif o_hit.size:  # p는 점수 내림차순이므로 첫 번째가 가장 높은 점수
                t, pc, sc = "분류 오류", int(p.loc[o_hit[0], "category_id"]), float(p.loc[o_hit[0], "score"])
            else:
                t, pc, sc = "놓침", np.nan, np.nan
            g_rows.append({"file_name": f, "category_id": int(r.category_id), "x": r.x, "y": r.y, "w": r.w, "h": r.h,
                           "type": t, "best_same_iou": best, "pred_category_id": pc, "pred_score": sc})
        taken = set()
        for j, r in p.iterrows():
            col = iou[:, j] if len(g) else np.zeros(0)
            hit_same = [i for i in np.argsort(-col) if col[i] >= iou_hit and same[i, j]]
            if hit_same:
                t = "중복" if hit_same[0] in taken else "정답 매칭"
                taken.add(hit_same[0])
            elif (col >= iou_hit).any():
                t = "다른 약 위 오분류"
            elif col.size and col.max() >= iou_near:
                t = "위치 부정확"
            else:
                t = "배경"
            near = int(np.argmax(col)) if col.size else -1
            p_rows.append({"file_name": f, "category_id": int(r.category_id), "score": float(r.score), "type": t,
                           "max_iou": float(col.max()) if col.size else 0.0,
                           "near_gt_category_id": int(g.loc[near, "category_id"]) if near >= 0 else np.nan})
    return pd.DataFrame(g_rows), pd.DataFrame(p_rows)
