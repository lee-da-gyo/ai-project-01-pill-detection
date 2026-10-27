"""조합 단위 Stratified Group K-Fold.

- Group = 조합(combo): 같은 조합을 각도만 바꿔 찍은 사진이 train/val에 나뉘면 누수다.
- Stratify = 이미지의 가장 희귀한 클래스: 희귀 클래스가 한 fold에 몰리지 않게 분산한다.
"""

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import N_FOLDS, SEED


def make_folds(clean: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = SEED) -> pd.DataFrame:
    """이미지 단위 테이블(file_name, combo, camera_la, classes, rarest, fold)을 반환한다."""
    img = clean.groupby("file_name").agg(
        combo=("combo", "first"), camera_la=("camera_la", "first"),
        classes=("category_id", lambda s: sorted(s)),
    ).reset_index()
    freq = clean["category_id"].value_counts()
    img["rarest"] = img["classes"].apply(lambda cs: min(cs, key=lambda c: freq[c]))

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    img["fold"] = -1
    for k, (_, val_idx) in enumerate(sgkf.split(img, y=img["rarest"], groups=img["combo"])):
        img.loc[val_idx, "fold"] = k

    assert img.groupby("combo")["fold"].nunique().max() == 1, "조합이 여러 fold에 걸쳐 있다 (누수)"
    return img


def fold_summary(img: pd.DataFrame) -> pd.DataFrame:
    """fold별 검증 데이터 크기와, 학습에 없는(처음 보는) 클래스 수."""
    rows = []
    for k in sorted(img["fold"].unique()):
        va, tr = img[img["fold"] == k], img[img["fold"] != k]
        va_cls = {c for cs in va["classes"] for c in cs}
        tr_cls = {c for cs in tr["classes"] for c in cs}
        rows.append({"fold": k, "val_images": len(va), "val_combos": va["combo"].nunique(),
                     "val_boxes": int(va["classes"].str.len().sum()), "val_classes": len(va_cls),
                     "val_classes_unseen_in_train": len(va_cls - tr_cls)})
    return pd.DataFrame(rows)
