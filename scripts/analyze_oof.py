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
    """fold별 검증 예측을 모아 232장 OOF 예측을 만든다.

    다음 중 하나라도 어긋나면 멈춘다 (조용히 틀린 OOF를 계산하지 않도록).
    - 이미지가 정확히 한 fold에만 속한다
    - 5개 fold 결과 파일이 모두 있다
    - 각 fold 파일에는 그 fold의 검증 이미지 예측만 있고, 검증 이미지마다 예측이 하나 이상 있다
    - 같은 예측 행이 중복되지 않는다
    - 합친 결과가 전체 이미지를 정확히 한 번씩 덮는다
    """
    if folds["file_name"].duplicated().any():
        raise SystemExit("folds.csv에 두 fold에 동시에 속한 이미지가 있다")
    parts = []
    for k in range(N_FOLDS):
        f = exp_root / (exp if k == 0 else f"{exp}_f{k}") / "val_predictions.csv"
        if not f.exists():
            raise SystemExit(f"{f} 가 없다. 5개 fold가 모두 있어야 한다")
        p = pd.read_csv(f)
        val_files = set(folds.loc[folds["fold"] == k, "file_name"])
        if extra := set(p["file_name"]) - val_files:
            raise SystemExit(f"{exp} fold {k}: 검증 이미지가 아닌 예측 {len(extra)}장 (예: {sorted(extra)[:3]})")
        if missing := val_files - set(p["file_name"]):
            raise SystemExit(f"{exp} fold {k}: 예측이 하나도 없는 검증 이미지 {len(missing)}장 (예: {sorted(missing)[:3]})")
        if (n_dup := p.duplicated().sum()):
            raise SystemExit(f"{exp} fold {k}: 중복 예측 행 {n_dup}개")
        parts.append(p)
    oof = pd.concat(parts, ignore_index=True)
    if set(oof["file_name"]) != set(folds["file_name"]):
        raise SystemExit(f"{exp}: OOF가 전체 {len(folds)}장을 덮지 않는다")
    return oof


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
        rows, fused_of = [], {}
        grid = [w for w in itertools.product([1, 2], repeat=len(exps)) if min(w) == 1]  # (1,1) (1,2) (2,1) …
        for thr in [0.5, 0.55, 0.6, 0.7]:
            for w in grid:
                cfg = (thr, ":".join(map(str, w)))
                fused = weighted_boxes_fusion(list(preds.values()), list(w), iou_thr=thr)
                fused_of[cfg] = fused.sort_values("score", ascending=False).groupby("file_name", group_keys=False).head(100)
                rows.append({"iou_thr": thr, "weights": cfg[1], **scores(fused_of[cfg], gt, learnable)})
        wbf = pd.DataFrame(rows).sort_values("learnable_mAP50_95", ascending=False)
        wbf.to_csv(P.out / "experiments" / "wbf_oof.csv", index=False)
        print(f"\nWBF ({' : '.join(exps)}) – 같은 OOF에서 고른 최댓값이라 낙관적이다 (탐색용)\n" + wbf.to_string(index=False))
        meta_validate(fused_of, gt, folds, learnable, P.out / "experiments" / "wbf_meta_validation.csv")


def meta_validate(fused_of: dict, gt: pd.DataFrame, folds: pd.DataFrame, learnable: set, out: Path):
    """WBF 설정 선택의 근사 메타 검증 (leave-one-fold-out, 설정 안정성 점검).

    fold k를 보류하고 나머지 4개 fold의 OOF로 설정을 고른 뒤, 그 설정을 fold k에만 적용한다.
    WBF는 이미지마다 독립으로 계산되므로 전체에 한 번 적용한 결과를 fold별로 잘라 써도 같다.

    한계: 설정을 고를 때 쓴 다른 fold의 OOF 예측은, 그 fold의 모델이 fold k를 학습에 포함해 만든 것이다.
    따라서 fold k가 선택 과정 전체에서 완전히 격리되지는 않는다 (엄밀한 비편향 추정이 아니다).
    엄밀한 nested CV는 fold k를 뺀 4개 fold 안에서 모델을 다시 inner 학습해야 하므로 저장된 OOF만으로는 할 수 없다.
    → 이 결과는 "fold마다 같은 설정이 뽑히는가(안정성)"의 근거로 쓰고, 합산 점수는 근사값으로만 보고한다.
    """
    picked, held_out = [], []
    for k in range(N_FOLDS):
        val = set(folds.loc[folds["fold"] == k, "file_name"])
        inner_gt = gt[~gt["file_name"].isin(val)]
        inner = {cfg: scores(p[~p["file_name"].isin(val)], inner_gt, learnable)["learnable_mAP50_95"]
                 for cfg, p in fused_of.items()}
        best = max(inner, key=inner.get)  # 동점이면 그리드 순서상 앞의 설정
        held = fused_of[best][fused_of[best]["file_name"].isin(val)]
        h = scores(held, gt[gt["file_name"].isin(val)], learnable)
        picked.append({"held_out_fold": str(k), "iou_thr": best[0], "weights": best[1],
                       "inner_learnable_mAP50_95": inner[best],
                       "held_out_learnable_mAP50_95": h["learnable_mAP50_95"], "held_out_mAP50_95": h["oof_mAP50_95"]})
        held_out.append(held)
    s = scores(pd.concat(held_out, ignore_index=True), gt, learnable)
    picked.append({"held_out_fold": "합산(5 fold 보류 예측)", "iou_thr": None, "weights": None, "inner_learnable_mAP50_95": None,
                   "held_out_learnable_mAP50_95": s["learnable_mAP50_95"], "held_out_mAP50_95": s["oof_mAP50_95"]})
    res = pd.DataFrame(picked)
    res.to_csv(out, index=False, encoding="utf-8-sig")
    print("\nWBF 근사 메타 검증 (fold별로 나머지 4 fold에서 설정 선택 → 보류 fold 평가. 완전 격리는 아님)\n" + res.to_string(index=False))


if __name__ == "__main__":
    main()
