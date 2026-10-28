"""두 실험의 5-Fold OOF 예측을 짝지어 비교한다 (기준 실험 대비 한 가지만 바꾼 실험).

같은 232장을 같은 분할로 평가했으므로 점수 차이를 짝지어 볼 수 있다.
1. fold별 차이: 5개 fold 중 몇 개에서 좋아졌는가
2. 조합 단위 부트스트랩: 조합(같은 약 묶음을 여러 각도로 찍은 이미지 묶음)을 복원 추출해 OOF mAP 차이의 95% 구간을 구한다.
   같은 조합의 이미지는 서로 비슷해 독립이 아니므로, 분할과 같은 단위(조합)로 다시 뽑는다.

주 지표는 37종 조건부 mAP(learnable)다. 56종 전체 OOF는 한 fold에만 있는 19종이 항상 0이라 두 실험 모두 같은 상한에 묶인다.

사용법
    python scripts/compare_oof.py e0_baseline e4_clahe

산출물
    reports/experiments/compare_<기준>_vs_<실험>.csv    (지표별 차이, 부트스트랩 구간, fold별 차이)
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, str(REPO_DIR / "scripts"))

from analyze_oof import load_oof, scores  # noqa: E402
from pilldet.config import N_FOLDS, SEED, get_paths  # noqa: E402
from pilldet.data import load_clean_annotations  # noqa: E402

METRICS = ["learnable_mAP50_95", "learnable_mAP50", "oof_mAP50_95", "oof_mAP50"]


def resample(df: pd.DataFrame, picks: list[tuple[int, list[str]]]) -> pd.DataFrame:
    """뽑힌 조합의 이미지를 모은다. 같은 조합이 여러 번 뽑히면 파일 이름에 #번호를 붙여 다른 이미지로 센다."""
    by_file = dict(tuple(df.groupby("file_name")))
    parts = [by_file[f].assign(file_name=f"{f}#{i}") for i, files in picks for f in files if f in by_file]
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[:0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("exp")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    P = get_paths(REPO_DIR)
    t0 = time.time()
    folds = pd.read_csv(P.out / "folds.csv")
    gt = load_clean_annotations(P.out / "annotations_clean.json")
    n_folds_of = gt.merge(folds[["file_name", "fold"]], on="file_name").groupby("category_id")["fold"].nunique()
    learnable = set(n_folds_of[n_folds_of > 1].index)
    root = P.out / "experiments"
    base, exp = load_oof(args.base, root, folds), load_oof(args.exp, root, folds)

    rows = []
    sb, se = scores(base, gt, learnable), scores(exp, gt, learnable)
    for m in METRICS:
        rows.append({"section": "OOF", "metric": m, args.base: sb[m], args.exp: se[m], "delta": round(se[m] - sb[m], 4)})

    # fold별 차이 (각 fold의 검증 이미지만, 같은 37종 기준)
    for k in range(N_FOLDS):
        val = set(folds.loc[folds["fold"] == k, "file_name"])
        g = gt[gt["file_name"].isin(val)]
        b, e = scores(base[base["file_name"].isin(val)], g, learnable), scores(exp[exp["file_name"].isin(val)], g, learnable)
        for m in ("learnable_mAP50_95", "oof_mAP50_95"):
            rows.append({"section": f"fold {k}", "metric": m, args.base: b[m], args.exp: e[m], "delta": round(e[m] - b[m], 4)})

    # 조합 단위 부트스트랩 (두 실험에 같은 표본을 써서 차이의 분포를 본다)
    combos = folds.groupby("combo")["file_name"].apply(list)
    rng = np.random.default_rng(args.seed)
    deltas = {m: [] for m in ("learnable_mAP50_95", "oof_mAP50_95")}
    for _ in range(args.n_boot):
        picks = list(enumerate(combos.iloc[rng.integers(0, len(combos), len(combos))]))
        g = resample(gt, picks)
        b, e = scores(resample(base, picks), g, learnable), scores(resample(exp, picks), g, learnable)
        for m in deltas:
            deltas[m].append(e[m] - b[m])
    for m, d in deltas.items():
        d = np.array(d)
        rows.append({"section": f"bootstrap (조합 {len(combos)}개, {args.n_boot}회)", "metric": m,
                     args.base: None, args.exp: None, "delta": round(float(np.median(d)), 4),
                     "ci95_low": round(float(np.percentile(d, 2.5)), 4), "ci95_high": round(float(np.percentile(d, 97.5)), 4),
                     "p_improve": round(float((d > 0).mean()), 3)})

    res = pd.DataFrame(rows)
    out = root / f"compare_{args.base}_vs_{args.exp}.csv"
    res.to_csv(out, index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))
    print(f"→ {out.relative_to(REPO_DIR)} ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
