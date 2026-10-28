"""전처리한 학습 이미지와 데이터셋 yaml을 만들고, 전처리가 Train/Test 분포 차이를 줄였는지 측정한다.

사용법
    python scripts/make_preprocessed.py --method clahe

산출물 (datasets/pill/ 아래, git 제외)
    <method>/images/*.png, <method>/labels/*.txt   전처리 이미지와 원본 라벨 복사본
    fold{k}_train_<method>.txt, fold{k}_val_<method>.txt, all_train_<method>.txt
    data_fold{k}_<method>.yaml, data_full_<method>.yaml
    reports/experiments/preprocess_<method>_stats.csv   Train/Test 밝기·대비 전후 비교
    reports/figures/19_<method>_before_after.png        Train 평균에 가까운 이미지와 가장 어두운 Test 이미지의 전후

Ultralytics는 이미지 경로의 마지막 /images/를 /labels/로 바꿔 라벨을 찾으므로 <method>/images, <method>/labels 구조를 쓴다.
테스트 이미지는 저장하지 않는다. 예측할 때 같은 함수로 그때그때 전처리한다 (predict_images의 preprocess 인자).
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

from pilldet.config import N_FOLDS, get_paths  # noqa: E402
from pilldet.data import list_test_images  # noqa: E402
from pilldet.preprocess import METHODS, load_bgr  # noqa: E402
from pilldet.viz import save_fig, setup_korean_font  # noqa: E402


def image_stats(bgr: np.ndarray) -> dict:
    """EDA(01_eda.ipynb)와 같은 정의: 244×320 썸네일의 RGB 평균(밝기)과 표준편차(대비)."""
    im = Image.fromarray(bgr[..., ::-1])
    im.thumbnail((244, 320))
    a = np.asarray(im, dtype=np.float32)
    return {"brightness": float(a.mean()), "contrast": float(a.std())}


def plot_examples(per_img: pd.DataFrame, m: str, fig_dir: Path):
    import matplotlib.pyplot as plt

    setup_korean_font()
    before = per_img[per_img["stage"] == "before"]
    tr = before[before["split"] == "train"]
    picks = [("Train (평균 밝기에 가장 가까운 이미지)", tr.loc[(tr["brightness"] - tr["brightness"].mean()).abs().idxmin(), "path"]),
             ("Test (가장 어두운 이미지)", before.loc[before[before["split"] == "test"]["brightness"].idxmin(), "path"])]
    fig, axes = plt.subplots(2, 2, figsize=(9, 11))
    for row, (title, path) in zip(axes, picks):
        for ax, method in zip(row, (None, m)):
            img = load_bgr(path, method)
            s = image_stats(img)
            ax.imshow(img[..., ::-1])
            ax.set_title(f"{title}\n{method or '원본'} | 밝기 {s['brightness']:.1f} · 대비 {s['contrast']:.1f}", fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    save_fig(fig, fig_dir, f"19_{m}_before_after")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=METHODS, required=True)
    args = ap.parse_args()
    m = args.method

    P = get_paths(REPO_DIR)
    t0 = time.time()
    img_out, lbl_out = P.dataset / m / "images", P.dataset / m / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    import cv2

    files = pd.read_csv(P.out / "folds.csv")["file_name"].tolist()
    rows, names = [], []
    for f in files:
        raw, pre = load_bgr(P.train_img / f), load_bgr(P.train_img / f, m)
        ok, buf = cv2.imencode(".png", pre)  # PNG는 무손실 → 예측 때 그때그때 전처리한 배열과 같다
        if not ok:
            raise RuntimeError(f"PNG 인코딩 실패: {f}")
        buf.tofile(str(img_out / f))
        shutil.copy2(P.dataset / "labels" / f.replace(".png", ".txt"), lbl_out / f.replace(".png", ".txt"))
        rows += [{"split": "train", "stage": "before", **image_stats(raw)}, {"split": "train", "stage": m, **image_stats(pre)}]
        names += [P.train_img / f] * 2
    for p in list_test_images(P.test_img):
        names += [p] * 2
        rows += [{"split": "test", "stage": "before", **image_stats(load_bgr(p))},
                 {"split": "test", "stage": m, **image_stats(load_bgr(p, m))}]

    # 원본 목록의 ./images/ 를 ./<method>/images/ 로 바꾼 목록과 yaml
    def rewrite(name: str) -> str:
        lines = (P.dataset / name).read_text(encoding="utf-8").split()
        out = name.replace(".txt", f"_{m}.txt")
        (P.dataset / out).write_text("\n".join(l.replace("./images/", f"./{m}/images/") for l in lines) + "\n",
                                     encoding="utf-8")
        return out

    for base in [*(f"data_fold{k}.yaml" for k in range(N_FOLDS)), "data_full.yaml"]:
        with open(P.dataset / base, encoding="utf-8") as fp:
            cfg = yaml.safe_load(fp)
        cfg.update(train=rewrite(cfg["train"]), val=rewrite(cfg["val"]))
        with open(P.dataset / base.replace(".yaml", f"_{m}.yaml"), "w", encoding="utf-8") as fp:
            yaml.safe_dump(cfg, fp, allow_unicode=True, sort_keys=False)

    per_img = pd.DataFrame(rows).assign(path=names)
    plot_examples(per_img, m, P.fig)
    stats = per_img.groupby(["stage", "split"])[["brightness", "contrast"]].agg(["mean", "std"]).round(2)
    stats.columns = [f"{a}_{b}" for a, b in stats.columns]
    stats = stats.reset_index()
    gap = stats.pivot(index="stage", columns="split", values=["brightness_mean", "contrast_mean"])
    stats["test_minus_train_brightness"] = stats["stage"].map(gap["brightness_mean"]["test"] - gap["brightness_mean"]["train"]).round(2)
    stats["test_minus_train_contrast"] = stats["stage"].map(gap["contrast_mean"]["test"] - gap["contrast_mean"]["train"]).round(2)
    out = P.out / "experiments" / f"preprocess_{m}_stats.csv"
    stats.to_csv(out, index=False, encoding="utf-8-sig")
    print(stats.to_string(index=False))
    print(f"이미지 {len(files)}장 → {img_out.relative_to(REPO_DIR)}, yaml data_fold*_{m}.yaml · data_full_{m}.yaml "
          f"({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
