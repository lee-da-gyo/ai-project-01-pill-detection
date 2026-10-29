"""현재 최종 후보(11s + 11m WBF)의 5-Fold OOF 오류 분석 – 보고서 8장 재료.

mAP 한 숫자를 나눠서 "어디서 점수를 잃는가"를 본다.
1. 클래스별 AP: 학습 가능한 약(2개 이상 fold에 있음)과 처음 보는 약(한 fold에만 있음)을 나눈다
2. IoU 임계값별 AP: mAP50과 mAP50-95의 차이가 위치 정밀도에서 오는지 확인한다
3. 실패 유형: 정답 박스는 정답 / 위치 오류 / 분류 오류 / 놓침, 예측 박스는 정답 매칭 / 중복 / 오분류 / 위치 부정확 / 배경
4. 혼동 쌍: 실제 약 → 예측한 약
5. 신뢰도로 처음 보는 약을 가를 수 있는가 (AUROC, 같은 알약에 이름이 둘 붙는 비율. 운영 임계값은 정하지 않는다)
6. 그림: 클래스별 AP, 실패 유형, 실패 사례, 테스트 예측 예시 (테스트는 정답이 없어 정확도를 주장하지 않는다)

새 학습은 없다. fold별 검증 예측(val_predictions.csv)을 다시 쓴다.

사용법
    python scripts/error_analysis.py                       # 기본: e0_baseline + e2_yolo11m, WBF IoU 0.7, 1:1
    python scripts/error_analysis.py --conf 0.25 0.5       # 실패 유형 신뢰도 임계값 (첫 값이 주 결과, 나머지는 민감도)

산출물
    reports/experiments/error_per_class.csv, error_per_iou.csv, error_types.csv, error_confusions.csv, error_gt_boxes.csv,
                          error_unknown_scores.csv, error_unknown_summary.csv
    reports/figures/20_per_class_ap.png, 21_error_types.png, 22_failure_cases.png, 23_test_predictions.png
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, str(REPO_DIR / "scripts"))

from analyze_oof import load_oof  # noqa: E402
from pilldet.boxes import iou_matrix, to_xyxy  # noqa: E402
from pilldet.config import get_paths  # noqa: E402
from pilldet.data import combo_of, load_clean_annotations  # noqa: E402
from pilldet.ensemble import weighted_boxes_fusion  # noqa: E402
from pilldet.evaluate import GT_TYPES, PRED_TYPES, classify_errors, coco_map  # noqa: E402
from pilldet.viz import save_fig, setup_korean_font  # noqa: E402

LEARN, UNSEEN = "학습 가능 37종", "처음 보는 19종"
C_LEARN, C_UNSEEN = "#1f5fa8", "#d9822b"


def short(name: str, n: int = 12) -> str:
    return name if len(name) <= n else name[:n] + "…"


def train_boxes_when_validated(gt: pd.DataFrame) -> pd.Series:
    """정답 박스마다, 그 박스가 검증될 때 학습 fold에 같은 약 박스가 몇 개 있었는지 → 클래스별 평균."""
    per_fold = gt.groupby(["category_id", "fold"]).size().unstack(fill_value=0)
    total = per_fold.sum(axis=1)
    n_train = gt.apply(lambda r: total[r.category_id] - per_fold.loc[r.category_id, r.fold], axis=1)
    return n_train.groupby(gt["category_id"]).mean()


def low_conf_top(pred: pd.DataFrame, boxes: pd.DataFrame) -> tuple[list, list]:
    """정답 박스와 IoU ≥ 0.5로 겹친 예측 중 신뢰도가 가장 높은 것의 (클래스, 점수). 없으면 NaN."""
    cls, sc = [], []
    for _, r in boxes.iterrows():
        p = pred[pred.file_name == r.file_name]
        iou = iou_matrix(to_xyxy(pd.DataFrame([r])), to_xyxy(p))[0] if len(p) else np.zeros(0)
        q = p[iou >= 0.5]
        best = q.loc[q.score.idxmax()] if len(q) else None
        cls.append(int(best.category_id) if best is not None else np.nan)
        sc.append(float(best.score) if best is not None else np.nan)
    return cls, sc


def plot_per_class(pc: pd.DataFrame, fig_dir: Path):
    d = pc.sort_values(["learnable", "AP50_95"], ascending=[False, False]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(14, 4.6))
    colors = np.where(d["learnable"], C_LEARN, C_UNSEEN)
    ax.bar(range(len(d)), d["AP50_95"], color=colors)
    ax.set_xticks(range(len(d)), [short(n, 10) for n in d["dl_name"]], rotation=90, fontsize=6.5)
    ax.set_ylabel("AP50-95 (OOF)")
    ax.set_ylim(0, 1.05)
    n_l = int(d["learnable"].sum())
    ax.axvline(n_l - 0.5, color="gray", ls=":", lw=1)
    ax.text(n_l / 2, 1.0, f"{LEARN}: 평균 {d.loc[d.learnable, 'AP50_95'].mean():.3f}", ha="center", fontsize=9, color=C_LEARN)
    ax.text(n_l + (len(d) - n_l) / 2, 1.0, f"{UNSEEN}: 검증 때 학습에 없음 → AP 0", ha="center", fontsize=9, color=C_UNSEEN)
    ax.set_title("클래스별 AP50-95 (11s+11m WBF, 5-Fold OOF)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, fig_dir, "20_per_class_ap")
    plt.close(fig)


def plot_error_types(gtb: pd.DataFrame, pb: pd.DataFrame, per_iou: pd.DataFrame, conf: float, fig_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    gt_colors = {"정답": "#3c8d40", "위치 오류": "#e0b100", "분류 오류": "#c0392b", "놓침": "#7f8c8d"}
    pred_colors = {"정답 매칭": "#3c8d40", "중복": "#9b59b6", "다른 약 위 오분류": "#c0392b", "위치 부정확": "#e0b100",
                   "배경": "#7f8c8d"}
    for ax, df, types, cols, title in [(axes[0], gtb, GT_TYPES, gt_colors, f"정답 박스 기준 (신뢰도 ≥ {conf})"),
                                       (axes[1], pb, PRED_TYPES, pred_colors, f"예측 박스 기준 (신뢰도 ≥ {conf})")]:
        tab = pd.crosstab(df["group"], df["type"], normalize="index").reindex(index=[LEARN, UNSEEN], columns=types, fill_value=0)
        cnt = pd.crosstab(df["group"], df["type"]).reindex(index=[LEARN, UNSEEN], columns=types, fill_value=0)
        left = np.zeros(len(tab))
        for t in types:
            ax.barh(range(len(tab)), tab[t], left=left, color=cols[t], label=t)
            for i, (v, c) in enumerate(zip(tab[t], cnt[t])):
                if v >= 0.06:
                    ax.text(left[i] + v / 2, i, f"{c}", ha="center", va="center", color="white", fontsize=8.5)
            left += tab[t].to_numpy()
        ax.set_yticks(range(len(tab)), [f"{g}\n(n={cnt.loc[g].sum()})" for g in tab.index], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("비율 (막대 안 숫자는 개수)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    ax = axes[2]
    ax.plot(per_iou["iou"], per_iou["mAP"], "o-", color=C_LEARN)
    for x, y in zip(per_iou["iou"], per_iou["mAP"]):
        ax.text(x, y - 0.045, f"{y:.3f}", ha="center", fontsize=7)
    ax.set_xlabel("IoU 임계값")
    ax.set_ylabel("mAP (학습 가능 37종)")
    ax.set_ylim(max(0, per_iou["mAP"].min() - 0.15), 1.02)
    ax.set_title("IoU 임계값별 mAP: 위치 정밀도", fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, fig_dir, "21_error_types")
    plt.close(fig)


def crop_panel(ax, img_dir: Path, r: pd.Series, pred: pd.DataFrame, names: dict, title: str):
    """정답 박스(초록)와 그 주변의 예측 박스(빨강, 신뢰도 ≥ conf)를 잘라 보여 준다."""
    img = Image.open(img_dir / r.file_name)
    W, H = img.size
    m = 0.35 * max(r.w, r.h)
    x1, y1 = max(0, r.x - m), max(0, r.y - m)
    x2, y2 = min(W, r.x + r.w + m), min(H, r.y + r.h + m)
    ax.imshow(img.crop((int(x1), int(y1), int(x2), int(y2))))
    ax.add_patch(patches.Rectangle((r.x - x1, r.y - y1), r.w, r.h, fill=False, ec="lime", lw=2))
    near = pred[(pred.file_name == r.file_name) & (pred.x < r.x + r.w) & (pred.x + pred.w > r.x) &
                (pred.y < r.y + r.h) & (pred.y + pred.h > r.y)].sort_values("score", ascending=False).head(2)
    for _, p in near.iterrows():
        ax.add_patch(patches.Rectangle((p.x - x1, p.y - y1), p.w, p.h, fill=False, ec="red", lw=1.6, ls="--"))
    ax.set_title(title, fontsize=7.2)
    ax.axis("off")


def plot_failures(gtb: pd.DataFrame, pred: pd.DataFrame, names: dict, img_dir: Path, conf: float, fig_dir: Path):
    # 같은 약·같은 혼동이 한 줄을 다 채우지 않도록 (실제, 예측) 쌍마다 신뢰도가 가장 높은 사례 하나만 쓴다
    mis = gtb[gtb.type == "분류 오류"].sort_values("pred_score", ascending=False).drop_duplicates(["category_id", "pred_category_id"])
    miss = gtb[(gtb.group == LEARN) & (gtb.type == "놓침")].drop_duplicates("category_id")
    rows = [("학습 가능한 약의 분류 오류 (신뢰도 높은 순)", mis[mis.group == LEARN]),
            ("처음 보는 약의 분류 오류 (신뢰도 높은 순)", mis[mis.group == UNSEEN]),
            (f"학습 가능한 약의 놓침 (신뢰도 {conf} 미만, 약마다 하나)", miss)]
    rows = [(t, d.head(4)) for t, d in rows if len(d)]
    fig = plt.figure(figsize=(12, 3.6 * len(rows)))
    subfigs = np.atleast_1d(fig.subfigures(len(rows), 1))
    p = pred[pred.score >= conf]
    for (title, d), sf in zip(rows, subfigs):
        ax_row = sf.subplots(1, 4)
        sf.subplots_adjust(top=0.78, bottom=0.02, wspace=0.15)
        sf.suptitle(title, x=0.01, y=0.99, ha="left", fontsize=10, fontweight="bold")
        for ax, (_, r) in zip(ax_row, d.iterrows()):
            if r.type == "분류 오류":
                cap = f"실제 {short(names[r.category_id])}\n→ 예측 {short(names[int(r.pred_category_id)])} ({r.pred_score:.2f})"
            elif r.type == "놓침":
                low = (f"{short(names[int(r.low_pred_category_id)], 8)} {r.low_pred_score:.2f}"
                       if pd.notna(r.low_pred_score) else "없음")
                cap = f"실제 {short(names[r.category_id])}\n놓침 · 겹친 최고 예측: {low}"
            else:
                cap = f"{short(names[r.category_id])}\n{r.type}, IoU {r.best_same_iou:.2f}"
            crop_panel(ax, img_dir, r, p, names, cap)
        for ax in ax_row[len(d):]:
            ax.axis("off")
    fig.suptitle(f"실패 사례 (초록 실선 = 정답, 빨강 점선 = 신뢰도 ≥ {conf} 예측)", fontsize=11, y=1.01)
    save_fig(fig, fig_dir, "22_failure_cases")
    plt.close(fig)


def plot_test(sub_csv: Path, test_dir: Path, names: dict, conf: float, fig_dir: Path, n: int = 6, seed: int = 0):
    sub = pd.read_csv(sub_csv)
    ids = sorted(sub["image_id"].unique())
    pick = sorted(np.random.default_rng(seed).choice(ids, n, replace=False))
    fig, axes = plt.subplots(2, 3, figsize=(13, 9.5))
    for ax, i in zip(axes.flat, pick):
        img = Image.open(test_dir / f"{i}.png")
        ax.imshow(img)
        d = sub[(sub.image_id == i) & (sub.score >= conf)].sort_values("score", ascending=False).reset_index(drop=True)
        xy = to_xyxy(d.rename(columns={"bbox_x": "x", "bbox_y": "y", "bbox_w": "w", "bbox_h": "h"}))
        iou = iou_matrix(xy, xy)
        stacked = {}
        for k, r in d.iterrows():
            # 같은 알약에 다른 이름이 또 붙으면(점수가 낮은 쪽) 노란 점선 박스와 아래쪽 이름표로 구분한다
            second = bool((iou[k, :k] >= 0.5).any())
            if second:
                first = int(np.argmax(iou[k, :k] >= 0.5))
                stacked[first] = stacked.get(first, 0) + 1
            ax.add_patch(patches.Rectangle((r.bbox_x, r.bbox_y), r.bbox_w, r.bbox_h, fill=False,
                                           ec="gold" if second else "red", lw=1.8, ls="--" if second else "-"))
            ax.text(r.bbox_x, r.bbox_y + r.bbox_h + 30 * stacked[first] if second else max(r.bbox_y - 8, 0),
                    f"{short(names[r.category_id], 9)} {r.score:.2f}", fontsize=6.5,
                    bbox=dict(fc="lightyellow" if second else "white", alpha=0.9, pad=1))
        n_second = int(sum((iou[k, :k] >= 0.5).any() for k in range(len(d))))
        ax.set_title(f"test {i}.png – 신뢰도 ≥ {conf} 박스 {len(d)}개" + (f" (같은 알약의 두 번째 이름 {n_second}개)" if n_second else ""),
                     fontsize=8.5)
        ax.axis("off")
    fig.suptitle(f"테스트 예측 예시 (현재 최종 후보, 고정 시드 {seed}로 6장 무작위 선택, 정답 없음)", fontsize=11)
    fig.tight_layout()
    save_fig(fig, fig_dir, "23_test_predictions")
    plt.close(fig)


def overlap_scores(pred: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """정답 박스마다 IoU ≥ 0.5로 겹친 예측의 최고·두 번째 신뢰도 (클래스 무관). 처음 보는 약을 신뢰도로 가를 수 있는지 볼 때 쓴다."""
    rows = {}
    for f, g in gt.groupby("file_name", sort=False):
        p = pred[pred.file_name == f]
        iou = iou_matrix(to_xyxy(g), to_xyxy(p)) if len(p) else np.zeros((len(g), 0))
        for i, idx in enumerate(g.index):
            s = np.sort(p["score"].to_numpy()[iou[i] >= 0.5])[::-1]
            rows[idx] = {"top": s[0] if s.size else 0.0, "second": s[1] if s.size > 1 else 0.0}
    return pd.DataFrame.from_dict(rows, orient="index").reindex(gt.index)


def competing_labels(boxes: pd.DataFrame, conf: float) -> tuple[int, int]:
    """신뢰도 ≥ conf 박스를 IoU ≥ 0.5로 묶었을 때 (묶음 수, 이름이 2개 이상 붙은 묶음 수). 정답이 없는 테스트에도 쓸 수 있다."""
    n = amb = 0
    for _, g in boxes[boxes.score >= conf].groupby("file_name"):
        g = g.sort_values("score", ascending=False)
        iou = iou_matrix(to_xyxy(g), to_xyxy(g))
        used = set()
        for a in range(len(g)):
            if a in used:
                continue
            grp = set(np.where(iou[a] >= 0.5)[0]) - used
            used |= grp
            n, amb = n + 1, amb + (len(grp) >= 2)
    return n, amb


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exps", nargs="+", default=["e0_baseline", "e2_yolo11m"])
    ap.add_argument("--iou-thr", type=float, default=0.7, help="WBF IoU (README 7장에서 고른 값)")
    ap.add_argument("--conf", nargs="+", type=float, default=[0.25, 0.5])
    ap.add_argument("--submission", default="final_11s_11m_wbf", help="테스트 예측 그림에 쓸 제출 파일")
    args = ap.parse_args()

    P = get_paths(REPO_DIR)
    exp_dir = P.out / "experiments"
    setup_korean_font()
    folds = pd.read_csv(P.out / "folds.csv")
    class_map = pd.read_csv(P.out / "class_map.csv")
    names = dict(zip(class_map["category_id"], class_map["dl_name"]))
    gt = load_clean_annotations(P.out / "annotations_clean.json").merge(folds[["file_name", "fold"]], on="file_name")
    n_folds_of = gt.groupby("category_id")["fold"].nunique()
    learnable = set(n_folds_of[n_folds_of > 1].index)
    group_of = lambda c: LEARN if c in learnable else UNSEEN  # noqa: E731

    preds = [load_oof(e, exp_dir, folds) for e in args.exps]
    pred = preds[0] if len(preds) == 1 else weighted_boxes_fusion(preds, None, iou_thr=args.iou_thr)
    pred = pred.sort_values("score", ascending=False).groupby("file_name", group_keys=False).head(100)
    tag = " + ".join(args.exps) + (f" WBF IoU {args.iou_thr}" if len(preds) > 1 else "")

    # 1. 클래스별 AP
    full = coco_map(pred, gt)
    learn = coco_map(pred[pred.category_id.isin(learnable)], gt[gt.category_id.isin(learnable)])
    pc = full["per_class"].merge(class_map[["category_id", "dl_name", "n_combos"]], on="category_id")
    pc["n_folds"] = pc["category_id"].map(n_folds_of)
    pc["learnable"] = pc["category_id"].isin(learnable)
    pc["train_boxes_when_validated"] = pc["category_id"].map(train_boxes_when_validated(gt)).round(1)
    pc = pc.sort_values(["learnable", "AP50_95"], ascending=[False, True])
    pc.round(4).to_csv(exp_dir / "error_per_class.csv", index=False, encoding="utf-8-sig")
    lp = pc[pc.learnable]
    print(f"[{tag}] OOF mAP50-95 {full['mAP50_95']:.4f} · 학습 가능 37종 {learn['mAP50_95']:.4f} (AP50 {learn['mAP50']:.4f})")
    print(f"학습 가능 37종 클래스별 AP50-95: 중앙값 {lp.AP50_95.median():.3f}, 최소 {lp.AP50_95.min():.3f}, "
          f"0.9 미만 {int((lp.AP50_95 < 0.9).sum())}종, AP50 1.0 {int((lp.AP50 >= 0.999).sum())}종")
    rho = lp[["AP50_95", "train_boxes_when_validated"]].corr(method="spearman").iloc[0, 1]
    print(f"학습 박스 수와 AP의 Spearman ρ = {rho:.2f} (n=37, 참고값)")
    print("AP 하위 5종:\n" + lp.head(5)[["dl_name", "AP50_95", "AP50", "n_gt", "train_boxes_when_validated"]].to_string(index=False))

    # 2. IoU 임계값별 AP (학습 가능 37종)
    per_iou = pd.DataFrame({"iou": list(learn["per_iou"]), "mAP": list(learn["per_iou"].values())}).round(4)
    per_iou.to_csv(exp_dir / "error_per_iou.csv", index=False)
    print("IoU별 mAP (37종): " + " ".join(f"{r.iou:.2f}:{r.mAP:.3f}" for r in per_iou.itertuples()))

    # 3. 실패 유형 (주 임계값 + 민감도)
    type_rows, main_gt, main_pred = [], None, None
    # 놓침이 "박스를 못 찾음"인지 "찾았지만 신뢰도가 낮음"인지 보려고, 모든 예측(≥ 0.001)으로 다시 분류해 붙인다
    low, _ = classify_errors(pred, gt, conf=0.001)
    for conf in args.conf:
        gtb, pb = classify_errors(pred, gt, conf=conf)
        gtb["group"] = gtb["category_id"].map(group_of)
        gtb["type_any_conf"] = low["type"]
        gtb["low_pred_category_id"], gtb["low_pred_score"] = low["pred_category_id"], low["pred_score"]
        hit = gtb.type == "놓침"
        top = low_conf_top(pred, gtb[hit])
        gtb.loc[hit, "low_pred_category_id"], gtb.loc[hit, "low_pred_score"] = top[0], top[1]
        pb["group"] = pb["near_gt_category_id"].map(lambda c: group_of(c) if pd.notna(c) else LEARN)
        for kind, df, types in [("정답 박스", gtb, GT_TYPES), ("예측 박스", pb, PRED_TYPES)]:
            for g in [LEARN, UNSEEN]:
                sub = df[df.group == g]
                for t in types:
                    type_rows.append({"conf": conf, "basis": kind, "group": g, "type": t, "n": int((sub.type == t).sum()),
                                      "share": round(float((sub.type == t).mean()), 4) if len(sub) else np.nan})
        if main_gt is None:
            main_gt, main_pred = gtb, pb
    types = pd.DataFrame(type_rows)
    types.to_csv(exp_dir / "error_types.csv", index=False, encoding="utf-8-sig")
    print("\n실패 유형 (개수)\n" + types.pivot_table(index=["conf", "basis", "group"], columns="type", values="n",
                                                  aggfunc="sum").fillna(0).astype(int).to_string())
    ok = main_gt[(main_gt.type == "정답") & (main_gt.group == LEARN)]["best_same_iou"]
    print(f"정답 박스의 IoU (37종, 신뢰도 ≥ {args.conf[0]}): 중앙값 {ok.median():.3f}, 0.9 미만 {(ok < 0.9).mean():.1%}, "
          f"0.75 미만 {(ok < 0.75).mean():.1%}")
    m = main_gt[main_gt.type == "놓침"]
    print(f"놓침 {len(m)}개 중 신뢰도 {args.conf[0]} 미만의 겹친 예측이 있는 것 {int(m.low_pred_score.notna().sum())}개 · "
          f"그 예측이 맞는 약인 것 {int((m.low_pred_category_id == m.category_id).sum())}개 · "
          f"모든 예측으로 다시 분류: {m.type_any_conf.value_counts().to_dict()}")
    main_gt.assign(dl_name=main_gt.category_id.map(names)).to_csv(exp_dir / "error_gt_boxes.csv", index=False,
                                                                    encoding="utf-8-sig")

    # 4. 혼동 쌍
    mis = main_gt[main_gt.type == "분류 오류"]
    conf_pairs = (mis.groupby(["group", "category_id", "pred_category_id"])
                  .agg(n=("pred_score", "size"), mean_score=("pred_score", "mean"), max_score=("pred_score", "max"))
                  .reset_index().sort_values(["group", "n", "mean_score"], ascending=[True, False, False]))
    conf_pairs["true_name"] = conf_pairs.category_id.map(names)
    conf_pairs["pred_name"] = conf_pairs.pred_category_id.astype(int).map(names)
    conf_pairs["pred_is_learnable"] = conf_pairs.pred_category_id.isin(learnable)
    conf_pairs.round(3).to_csv(exp_dir / "error_confusions.csv", index=False, encoding="utf-8-sig")
    print("\n혼동 쌍\n" + conf_pairs[["group", "true_name", "pred_name", "n", "mean_score", "max_score"]].round(3).to_string(index=False))

    # 5. 신뢰도로 처음 보는 약을 가를 수 있는가 (운영 임계값을 정하는 분석이 아니다: 처음 보는 약 63개, 같은 촬영 환경)
    sc = overlap_scores(pred, gt).assign(group=gt["category_id"].map(group_of).to_numpy())
    sc.to_csv(exp_dir / "error_unknown_scores.csv", index=False, encoding="utf-8-sig")
    u, k = sc[sc.group == UNSEEN], sc[sc.group == LEARN]
    auroc = roc_auc_score(sc.group == UNSEEN, -sc.top)
    # 같은 조합의 사진(1~3장)은 독립 표본이 아니므로 조합 단위로 다시 뽑아 95% 구간을 낸다
    sc["combo"] = gt["file_name"].map(combo_of).to_numpy()
    by_combo = dict(tuple(sc.groupby("combo")))
    combos, rng, boot = list(by_combo), np.random.default_rng(0), []
    for _ in range(1000):
        b = pd.concat([by_combo[c] for c in rng.choice(combos, len(combos))])
        if b.group.nunique() == 2:
            boot.append(roc_auc_score(b.group == UNSEEN, -b.top))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    n_o, a_o = competing_labels(pred[pred.file_name.isin(gt.file_name)], args.conf[0])
    amb = {g: float((d.second >= args.conf[0]).mean()) for g, d in [(LEARN, k), (UNSEEN, u)]}
    test = pd.read_csv(REPO_DIR / "submissions" / f"{args.submission}.csv").rename(
        columns={"image_id": "file_name", "bbox_x": "x", "bbox_y": "y", "bbox_w": "w", "bbox_h": "h"})
    n_t, a_t = competing_labels(test, args.conf[0])
    unk = pd.DataFrame([
        {"item": "최고 신뢰도 중앙값", LEARN: k.top.median(), UNSEEN: u.top.median()},
        {"item": "최고 신뢰도 ≥ 0.9 비율", LEARN: (k.top >= 0.9).mean(), UNSEEN: (u.top >= 0.9).mean()},
        {"item": "최고 신뢰도 < 0.5 비율", LEARN: (k.top < 0.5).mean(), UNSEEN: (u.top < 0.5).mean()},
        {"item": f"두 번째 이름도 ≥ {args.conf[0]} 비율", LEARN: amb[LEARN], UNSEEN: amb[UNSEEN]},
        {"item": "AUROC (최고 신뢰도가 낮을수록 처음 보는 약)", "both": auroc},
        {"item": "AUROC 95% 구간 하한 (조합 단위 부트스트랩 1000회)", "both": lo},
        {"item": "AUROC 95% 구간 상한", "both": hi},
    ]).round(4)
    unk.to_csv(exp_dir / "error_unknown_summary.csv", index=False, encoding="utf-8-sig")
    print("\n신뢰도와 처음 보는 약 (OOF 정답 박스 기준)\n" + unk.to_string(index=False))
    print(f"이름이 2개 이상 붙은 알약 묶음 (신뢰도 ≥ {args.conf[0]}): OOF {a_o}/{n_o} ({a_o / n_o:.1%}), "
          f"테스트 {a_t}/{n_t} ({a_t / n_t:.1%}) – 테스트는 정답이 없어 정확도로 해석하지 않는다")

    # 6. 그림
    plot_per_class(pc, P.fig)
    plot_error_types(main_gt, main_pred, per_iou, args.conf[0], P.fig)
    plot_failures(main_gt, pred, names, P.train_img, args.conf[0], P.fig)
    plot_test(REPO_DIR / "submissions" / f"{args.submission}.csv", P.test_img, names, args.conf[0], P.fig)
    print("\n그림: 20_per_class_ap, 21_error_types, 22_failure_cases, 23_test_predictions")


if __name__ == "__main__":
    main()
