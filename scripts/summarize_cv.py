"""5-Fold 교차검증 결과 집계.

fold별 점수의 평균·표준편차와 함께, 5개 fold의 검증 예측을 합쳐 **232장 전체에 대한 OOF(out-of-fold) mAP**를 계산한다.
OOF는 모든 이미지가 정확히 한 번씩, 그 이미지를 학습하지 않은 모델로 평가되므로 fold 하나보다 훨씬 안정적인 추정치다.

사용법
    python scripts/summarize_cv.py e0_baseline e2_yolo11m
    (fold 0은 <실험명>, fold 1~4는 <실험명>_f1 ~ _f4 의 결과를 쓴다)

산출물
    reports/cv_summary.csv
"""

import sys
from pathlib import Path

import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

from pilldet.config import N_FOLDS, get_paths  # noqa: E402
from pilldet.data import load_clean_annotations  # noqa: E402
from pilldet.evaluate import coco_map  # noqa: E402


def fold_dirs(exp: str, exp_root: Path) -> list[Path | None]:
    return [exp_root / (exp if k == 0 else f"{exp}_f{k}") for k in range(N_FOLDS)]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    allow_partial = "--partial" in sys.argv  # 진행 중 점검용. 결과에 complete=False로 표시된다
    exps = args
    if not exps:
        raise SystemExit("실험 이름을 하나 이상 넣자. 예: python scripts/summarize_cv.py e0_baseline [--partial]")
    P = get_paths(REPO_DIR)
    gt = load_clean_annotations(P.out / "annotations_clean.json")
    folds = pd.read_csv(P.out / "folds.csv")
    rows = []
    for exp in exps:
        per_fold, preds, used = [], [], []
        for k, d in enumerate(fold_dirs(exp, P.out / "experiments")):
            f = d / "val_predictions.csv"
            if not f.exists():
                continue
            p = pd.read_csv(f)
            val_files = set(folds.loc[folds["fold"] == k, "file_name"])
            p = p[p["file_name"].isin(val_files)]
            per_fold.append(coco_map(p, gt[gt["file_name"].isin(val_files)])["mAP50_95"])
            preds.append(p)
            used.append(k)
        complete = used == list(range(N_FOLDS))
        if not complete and not allow_partial:
            raise SystemExit(f"{exp}: fold {sorted(set(range(N_FOLDS)) - set(used))} 결과가 없다. "
                             "5개 fold가 모두 있어야 OOF라고 부를 수 있다 (진행 중 점검은 --partial)")
        if not preds:
            continue
        oof = pd.concat(preds)
        covered = set(folds.loc[folds["fold"].isin(used), "file_name"])  # 결과가 있는 fold의 검증 이미지 전체
        if complete:
            assert len(covered) == len(folds) == folds["file_name"].nunique(), "OOF가 전체 이미지를 정확히 한 번씩 덮지 않는다"
        m = coco_map(oof, gt[gt["file_name"].isin(covered)])
        s = pd.Series(per_fold)
        rows.append({"experiment": exp, "complete": complete, "n_images": len(covered),
                     "folds": ",".join(map(str, used)), "fold_mean": round(s.mean(), 4),
                     "fold_std": round(s.std(ddof=1), 4) if len(s) > 1 else None, "fold_min": round(s.min(), 4),
                     "oof_mAP50_95": round(m["mAP50_95"], 4), "oof_mAP50": round(m["mAP50"], 4),
                     "per_fold": " / ".join(f"{v:.3f}" for v in per_fold)})
    res = pd.DataFrame(rows)
    out = P.out / "cv_summary.csv"
    if out.exists():
        old = pd.read_csv(out)
        res = pd.concat([old[~old["experiment"].isin(res["experiment"])], res], ignore_index=True)
    res.to_csv(out, index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
