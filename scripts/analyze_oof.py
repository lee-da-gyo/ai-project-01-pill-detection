"""5-Fold OOF 심층 분석 – open-set 상한 분해와 WBF 앙상블 설정 선택.

조합 단위 분할에서는 한 fold에만 있는 약이 생긴다. 그 약은 검증될 때 항상 학습에 없으므로 AP가 0이 되고,
OOF mAP는 모델 성능보다 분할 구조에 좌우된다. 그래서 다음 두 가지를 나눠 계산한다.
1. 전체 OOF mAP와 그 상한 (2개 이상 fold에 있는 클래스 비율)
2. 학습에서 볼 수 있는 클래스만의 OOF mAP → 전체 데이터로 학습하는 최종 모델(56종 모두 학습)에 가까운 추정치

WBF IoU 임계값과 모델 가중치는 이 OOF에서만 고른다 (fold 0 하나로 고르면 과적합).

사용법
    python scripts/analyze_oof.py e0_baseline e2_yolo11m

산출물
    reports/experiments/oof_analysis.csv   (모델별 OOF)
    reports/experiments/wbf_oof.csv        (WBF 설정별 OOF, 모델이 2개 이상일 때)
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import itertools  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

from pilldet.config import N_FOLDS, get_paths  # noqa: E402
from pilldet.data import load_clean_annotations  # noqa: E402
from pilldet.ensemble import weighted_boxes_fusion  # noqa: E402
from pilldet.evaluate import coco_map  # noqa: E402


def load_oof(exp: str, exp_root: Path, folds: pd.DataFrame) -> pd.DataFrame:
    """fold별 검증 예측을 모아 232장 OOF 예측을 만든다. 하나라도 없으면 멈춘다."""
    parts = []
    for k in range(N_FOLDS):
        f = exp_root / (exp if k == 0 else f"{exp}_f{k}") / "val_predictions.csv"
        if not f.exists():
            raise SystemExit(f"{f} 가 없다. 5개 fold가 모두 있어야 한다")
        p = pd.read_csv(f)
        parts.append(p[p["file_name"].isin(set(folds.loc[folds["fold"] == k, "file_name"]))])
    return pd.concat(parts, ignore_index=True)


def scores(pred: pd.DataFrame, gt: pd.DataFrame, learnable: set) -> dict:
    m = coco_map(pred, gt)
    s = coco_map(pred[pred["category_id"].isin(learnable)], gt[gt["category_id"].isin(learnable)])
    return {"oof_mAP50_95": round(m["mAP50_95"], 4), "oof_mAP50": round(m["mAP50"], 4),
            "learnable_mAP50_95": round(s["mAP50_95"], 4), "learnable_mAP50": round(s["mAP50"], 4)}


def main():
    exps = sys.argv[1:]
    if not exps:
        raise SystemExit("실험 이름을 하나 이상 넣자. 예: python scripts/analyze_oof.py e0_baseline e2_yolo11m")
    P = get_paths(REPO_DIR)
    folds = pd.read_csv(P.out / "folds.csv")
    gt = load_clean_annotations(P.out / "annotations_clean.json")
    n_folds_of = gt.merge(folds[["file_name", "fold"]], on="file_name").groupby("category_id")["fold"].nunique()
    learnable = set(n_folds_of[n_folds_of > 1].index)  # 검증될 때 다른 fold의 학습 데이터에도 있는 클래스
    print(f"클래스 {len(n_folds_of)}종 중 한 fold에만 있는 약 {len(n_folds_of) - len(learnable)}종 "
          f"→ OOF mAP 상한 {len(learnable) / len(n_folds_of):.3f}")

    preds = {e: load_oof(e, P.out / "experiments", folds) for e in exps}
    res = pd.DataFrame([{"experiment": e, **scores(p, gt, learnable)} for e, p in preds.items()])
    res.to_csv(P.out / "experiments" / "oof_analysis.csv", index=False)
    print(res.to_string(index=False))

    if len(exps) > 1:
        rows = []
        grid = [w for w in itertools.product([1, 2], repeat=len(exps)) if min(w) == 1]  # (1,1) (1,2) (2,1) …
        for thr in [0.5, 0.55, 0.6, 0.7]:
            for w in grid:
                fused = weighted_boxes_fusion(list(preds.values()), list(w), iou_thr=thr)
                fused = fused.sort_values("score", ascending=False).groupby("file_name", group_keys=False).head(100)
                rows.append({"iou_thr": thr, "weights": ":".join(map(str, w)), **scores(fused, gt, learnable)})
        wbf = pd.DataFrame(rows).sort_values("learnable_mAP50_95", ascending=False)
        wbf.to_csv(P.out / "experiments" / "wbf_oof.csv", index=False)
        print(f"\nWBF ({' : '.join(exps)})\n" + wbf.to_string(index=False))


if __name__ == "__main__":
    main()
