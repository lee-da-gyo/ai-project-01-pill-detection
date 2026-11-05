"""허용된 AI Hub 조합 데이터를 YOLO 학습 데이터로 만든다.

대회 안내상 금지 파일은 TL_2_조합.zip, TS_2_조합(원천)이다. 경로에 이 이름이 있으면 열지 않고 건너뛴다.

사용법
    python scripts/prepare_aihub.py --labels <라벨 zip 폴더> --images <이미지 zip 폴더>

걸러내는 것
    - 빈 JSON, 형식이 맞지 않는 박스, 이미지 밖 박스
    - 파일명의 약 코드 수와 박스 수가 다른 이미지 (라벨이 빠진 이미지)
    - 테스트 이미지와 같은 이미지. 해시가 같거나, 32px 차이가 작은 후보 중 128px 비교까지 통과하면 같은 사진으로 본다.
      이때는 그 조합 전체를 학습에서 빼고 라벨을 쓰지 않는다. 후보와 판정은 aihub_excluded.csv에 남긴다.

검증
    - 추가 데이터의 10% 조합을 떼어 새 약까지 채점한다 (같은 조합의 각도가 학습과 검증에 나뉘지 않는다)
    - 대회 fold 0과 함께 data_aihub_fold0.yaml의 검증으로 쓴다

약 코드는 라벨 경로의 K-코드를 쓴다. 대회 category_id와 값이 같다. dl_idx는 이상값이 있어 쓰지 않는다.

산출물
    datasets/aihub/images, labels, data_aihub_fold0.yaml, data_aihub_full.yaml, *.txt
    reports/class_map_aihub.csv         (yolo_id 0~55는 대회 클래스와 같다)
    reports/logs/prepare_aihub.log
"""

import argparse
import collections
import hashlib
import json
import logging
import random
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

from pilldet.config import SEED, get_paths  # noqa: E402
from pilldet.data import list_test_images  # noqa: E402

FORBIDDEN = re.compile(r"T[LS]_2_조합")
SET_ID = re.compile(r"([TV])[LS]_(\d+)_조합\.zip$")
DUP_CANDIDATE = 2.0  # 32x32 흑백 축소본의 평균 절대 차이. 단색 배경이 비슷해 오탐이 있으므로 후보로만 쓴다
DUP_CONFIRM = 1.0  # 128x128 축소본에서도 이보다 가까우면 같은 사진으로 확정한다 (학습 전에 고정)
HOLDOUT_RATE = 0.1
MIN_BOX = 10


def allowed_zips(folder: Path, log) -> dict[str, Path]:
    found = {}
    for p in sorted(folder.rglob("*.zip")):
        if FORBIDDEN.search(p.name):
            log(f"금지 파일 건너뜀 (열지 않음): {p.name}")
            continue
        m = SET_ID.search(p.name)
        if m and not zipfile.is_zipfile(p):
            log(f"아직 받는 중이거나 손상된 파일 건너뜀: {p.name}")
        elif m:
            found[f"{m.group(1)}{m.group(2)}"] = p
    return found


def read_labels(label_zip: Path) -> tuple[list[dict], collections.Counter]:
    rows, stat = [], collections.Counter()
    with zipfile.ZipFile(label_zip) as z:
        for name in z.namelist():
            if not name.endswith(".json"):
                continue
            try:
                d = json.loads(z.read(name))
                img = d["images"][0]
                code = int(re.fullmatch(r"K-(\d+)", Path(name).parent.name).group(1))
            except (ValueError, KeyError, IndexError, AttributeError):
                stat["읽을 수 없는 JSON"] += 1
                continue
            if not d.get("annotations"):
                stat["박스가 없는 JSON"] += 1
            if str(img.get("dl_idx")) != str(code - 1):
                stat["dl_idx가 K-코드 - 1과 다른 JSON"] += 1
            for a in d.get("annotations", []):
                b = a.get("bbox")
                if not isinstance(b, list) or len(b) != 4:
                    stat["형식이 틀린 박스"] += 1
                    continue
                rows.append({"file_name": img["file_name"], "code": code, "x": float(b[0]), "y": float(b[1]),
                             "w": float(b[2]), "h": float(b[3]), "img_w": img["width"], "img_h": img["height"],
                             "name_en": (img.get("dl_name_en") or "").strip() or (img.get("dl_name") or "").strip() or str(code)})
    return rows, stat


def filter_complete(df: pd.DataFrame) -> tuple[pd.DataFrame, collections.Counter]:
    """이미지의 약 코드와 박스가 정확히 맞는 이미지만 남긴다."""
    stat = collections.Counter()
    in_image = (df.x >= 0) & (df.y >= 0) & (df.x + df.w <= df.img_w) & (df.y + df.h <= df.img_h)
    big_enough = (df.w >= MIN_BOX) & (df.h >= MIN_BOX)
    bad_images = set(df.loc[~(in_image & big_enough), "file_name"])
    stat["이미지 밖 또는 너무 작은 박스가 있는 이미지"] = len(bad_images)
    keep = []
    for file_name, g in df[~df.file_name.isin(bad_images)].groupby("file_name"):
        codes = {int(c) for c in file_name.split("_")[0].split("-")[1:]}
        if set(g.code) == codes and len(g) == len(codes):
            keep.append(file_name)
        else:
            stat["약 코드와 박스가 맞지 않는 이미지"] += 1
    return df[df.file_name.isin(keep)].copy(), stat


def thumb(path_or_file, size: int = 32) -> np.ndarray:
    with Image.open(path_or_file) as im:
        return np.asarray(im.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32)


def split_holdout(combos: list[str], rate: float = HOLDOUT_RATE, seed: int = SEED) -> set[str]:
    """조합 단위로 검증용을 고른다. 입력 순서와 무관하게 같은 결과가 나온다."""
    ordered = sorted(set(combos))
    random.Random(seed).shuffle(ordered)
    return set(ordered[:round(len(ordered) * rate)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--images", type=Path, nargs="+", required=True)
    args = ap.parse_args()

    P = get_paths(REPO_DIR)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(P.out / "logs" / "prepare_aihub.log", "w", "utf-8")], force=True)
    log = logging.getLogger().info

    label_zips = allowed_zips(args.labels, log)
    image_zips = {}
    for folder in args.images:
        image_zips.update(allowed_zips(folder, log))
    sets = sorted(set(label_zips) & set(image_zips))
    log(f"라벨 {sorted(label_zips)} / 이미지 {sorted(image_zips)} → 사용 {sets}")
    if not sets:
        raise SystemExit("라벨과 이미지가 짝이 맞는 묶음이 없다")

    out = REPO_DIR / "datasets" / "aihub"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(exist_ok=True)

    class_map = pd.read_csv(P.out / "class_map.csv")
    test_paths = list_test_images(P.test_img)
    test_thumbs = np.stack([thumb(p) for p in test_paths])
    test_md5 = {hashlib.md5(p.read_bytes()).hexdigest() for p in list_test_images(P.test_img)}
    comp_combos = set(pd.read_csv(P.out / "folds.csv")["combo"])

    kept_tables, stat, min_dist, excluded = [], collections.Counter(), [], []
    for s in sets:
        rows, st = read_labels(label_zips[s])
        stat.update(st)
        df, st = filter_complete(pd.DataFrame(rows))
        stat.update(st)
        stat["같은 조합이 대회 학습 데이터에도 있음"] += int(df.file_name.str.split("_").str[0].isin(comp_combos).sum())
        wanted = set(df.file_name)
        kept = set()
        with zipfile.ZipFile(image_zips[s]) as z:
            members = {Path(n).name: n for n in z.namelist() if n.endswith(".png")}
            for fname in sorted(wanted):
                if fname not in members:
                    stat["이미지 파일 없음"] += 1
                    continue
                dst = out / "images" / fname
                data = z.read(members[fname])
                if hashlib.md5(data).hexdigest() in test_md5:
                    stat["테스트와 완전히 같은 이미지"] += 1
                    excluded.append({"file_name": fname, "reason": "md5 일치", "confirmed": True})
                    continue
                dst.write_bytes(data)
                dists = np.abs(test_thumbs - thumb(dst)).mean(axis=(1, 2))
                dist = float(dists.min())
                min_dist.append(dist)
                if dist < DUP_CANDIDATE:
                    fine = float(np.abs(thumb(test_paths[int(dists.argmin())], 128) - thumb(dst, 128)).mean())
                    confirmed = fine < DUP_CONFIRM
                    stat["중복 후보"] += 1
                    stat["중복 확정"] += int(confirmed)
                    excluded.append({"file_name": fname, "reason": f"32px {dist:.2f}, 128px {fine:.2f}", "confirmed": confirmed})
                kept.add(fname)
        kept_tables.append(df[df.file_name.isin(kept)])
        log(f"{s}: 후보 {len(wanted):,}장 → 사용 {len(kept):,}장")
    data = pd.concat(kept_tables, ignore_index=True)
    data["combo"] = data.file_name.str.split("_").str[0]
    report = pd.DataFrame(excluded, columns=["file_name", "reason", "confirmed"])
    report.to_csv(P.out / "aihub_excluded.csv", index=False, encoding="utf-8-sig")
    bad_combos = set(report.loc[report.confirmed.astype(bool), "file_name"].str.split("_").str[0])
    removed = data[data.combo.isin(bad_combos)].file_name.unique()
    for f in removed:
        (out / "images" / f).unlink(missing_ok=True)
    data = data[~data.combo.isin(bad_combos)]
    stat["중복 때문에 조합째 뺀 이미지"] = len(removed)

    new_codes = sorted(set(data.code) - set(class_map.category_id))
    names = data.drop_duplicates("code").set_index("code")["name_en"]
    extra = pd.DataFrame({"yolo_id": range(len(class_map), len(class_map) + len(new_codes)), "category_id": new_codes,
                          "dl_name_en": [names[c] for c in new_codes]})
    full_map = pd.concat([class_map[["yolo_id", "category_id", "dl_name_en"]], extra], ignore_index=True)
    assert full_map.category_id.is_unique and full_map.yolo_id.is_unique
    full_map.to_csv(P.out / "class_map_aihub.csv", index=False, encoding="utf-8-sig")
    to_yolo = dict(zip(full_map.category_id, full_map.yolo_id))
    data.rename(columns={"code": "category_id"})[["file_name", "category_id", "x", "y", "w", "h", "img_w", "img_h", "combo"]].to_csv(
        out / "boxes.csv", index=False, encoding="utf-8-sig")

    for file_name, g in data.groupby("file_name"):
        lines = [f"{to_yolo[r.code]} {(r.x + r.w / 2) / r.img_w:.6f} {(r.y + r.h / 2) / r.img_h:.6f} "
                 f"{r.w / r.img_w:.6f} {r.h / r.img_h:.6f}" for r in g.itertuples()]
        (out / "labels" / f"{Path(file_name).stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    holdout = split_holdout(data.combo.tolist())
    data_files = data.drop_duplicates("file_name")
    to_path = lambda fs: [str((out / "images" / f).resolve()) for f in sorted(fs)]  # noqa: E731
    aihub = to_path(data_files.file_name)
    aihub_train = to_path(data_files[~data_files.combo.isin(holdout)].file_name)
    aihub_val = to_path(data_files[data_files.combo.isin(holdout)].file_name)
    pill = REPO_DIR / "datasets" / "pill"
    read = lambda n: [str((pill / x).resolve()) for x in (pill / n).read_text(encoding="utf-8").splitlines() if x.strip()]  # noqa: E731
    lists = {"aihub_fold0_train.txt": aihub_train + read("fold0_train.txt"), "aihub_full_train.txt": aihub + read("all_train.txt"),
             "aihub_holdout_val.txt": aihub_val, "comp_fold0_val.txt": read("fold0_val.txt"),
             "combined_val.txt": read("fold0_val.txt") + aihub_val}
    for name, files in lists.items():
        (out / name).write_text("\n".join(files) + "\n", encoding="utf-8")
    for yaml_name, train in [("data_aihub_fold0.yaml", "aihub_fold0_train.txt"), ("data_aihub_full.yaml", "aihub_full_train.txt")]:
        with open(out / yaml_name, "w", encoding="utf-8") as f:
            yaml.safe_dump({"path": str(out.resolve()), "train": train, "val": "combined_val.txt", "nc": len(full_map),
                            "names": dict(zip(full_map.yolo_id.astype(int).tolist(), full_map.dl_name_en.astype(str).tolist()))},
                           f, allow_unicode=True, sort_keys=False)

    box = data.groupby("code").size()
    held = data[data.combo.isin(holdout)]
    log(f"검증용 조합 {len(holdout)}개 {len(aihub_val):,}장, 약 {held.code.nunique()}종. 검증용에 없는 약: "
        f"{sorted(set(data.code) - set(held.code))}, 학습 쪽에 없는 약: {sorted(set(data.code) - set(data[~data.combo.isin(holdout)].code))}")
    log(f"class_map_aihub.csv SHA-256 {hashlib.sha256((P.out / 'class_map_aihub.csv').read_bytes()).hexdigest()[:16]}, nc {len(full_map)}")
    q = np.quantile(min_dist, [0, .001, .01, .5]).round(2).tolist() if min_dist else []
    log(f"걸러낸 내역: {dict(stat)}")
    log(f"테스트와의 최소 거리 분위수 (최소, 0.1%, 1%, 50%): {q}")
    log(f"사용 이미지 {data.file_name.nunique():,}장, 박스 {len(data):,}개, 약 {data.code.nunique()}종")
    log(f"전체 클래스 {len(full_map)}종 (대회 56 + 새 약 {len(new_codes)}), 대회 56종 중 추가 데이터에 없는 약: "
        f"{sorted(set(class_map.category_id) - set(data.code))}")
    log(f"약별 박스 수: 최소 {box.min()}, 중앙값 {int(box.median())}, 최대 {box.max()}")


if __name__ == "__main__":
    main()
