"""이미지 전처리 – 학습·검증·테스트에 같은 함수를 적용한다.

CLAHE (Contrast Limited Adaptive Histogram Equalization)
    EDA에서 Test가 Train보다 조금 어둡고(밝기 125.9 vs 131.5) 대비가 컸다(26.5 vs 23.6).
    이미지마다 국소 대비를 평탄화하면 밝기·대비 차이가 줄어들 것이라는 가설을 검증한다.
    색조를 건드리지 않도록 LAB 색공간의 밝기(L) 채널에만 적용한다 (알약 색은 클래스를 가르는 단서다).

전처리 이름은 데이터셋 yaml 이름의 접미사로 정한다 (data_fold0_clahe.yaml → clahe).
학습 기록(args.yaml)에 남는 data 경로만 보고도 예측 때 같은 전처리를 다시 적용할 수 있게 하기 위해서다.
"""

from pathlib import Path

import numpy as np

# OpenCV 기본값 그대로 쓴다. 이 값을 val 점수로 고르면 선택 편향이 생기므로 조정하지 않는다
CLAHE_CLIP = 2.0
CLAHE_TILE = 8
METHODS = ("clahe",)


def clahe_bgr(img: np.ndarray, clip: float = CLAHE_CLIP, tile: int = CLAHE_TILE) -> np.ndarray:
    """BGR uint8 이미지의 L 채널에만 CLAHE를 적용한다."""
    import cv2

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply(img: np.ndarray, method: str | None) -> np.ndarray:
    if method is None:
        return img
    if method == "clahe":
        return clahe_bgr(img)
    raise ValueError(f"알 수 없는 전처리: {method} (가능: {METHODS})")


def preprocess_of(data_yaml: str) -> str | None:
    """data_fold0_clahe.yaml → 'clahe', data_fold0.yaml · data_fold0_cp.yaml → None."""
    stem = Path(str(data_yaml)).stem
    return next((m for m in METHODS if stem.endswith(f"_{m}")), None)


def load_bgr(path: Path, method: str | None = None) -> np.ndarray:
    """이미지를 BGR로 읽고 전처리한다. 한글 경로에서도 읽히도록 imdecode를 쓴다."""
    import cv2

    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽지 못했다: {path}")
    return apply(img, method)
