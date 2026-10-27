"""경로 탐색과 공통 설정.

로컬, Kaggle 어디서 실행해도 같은 코드가 돌도록 데이터와 출력 경로를 자동으로 정한다.
"""

import os
from dataclasses import dataclass
from pathlib import Path

SEED = 42
N_FOLDS = 5
DATA_DIR_NAME = "sprint_ai_project1_data"
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")


def is_kaggle() -> bool:
    return KAGGLE_WORKING.exists()


def find_data_root(start: Path | None = None) -> Path:
    """`train_annotations` 폴더를 가진 데이터 루트를 찾는다.

    우선순위: 환경 변수 PILL_DATA_DIR → Kaggle 입력 폴더 → 상위 폴더와 그 옆의 데이터 폴더
    """
    if os.environ.get("PILL_DATA_DIR"):
        return Path(os.environ["PILL_DATA_DIR"])
    if KAGGLE_INPUT.exists():
        for depth in ("*", "*/*", "*/*/*"):
            hits = sorted(KAGGLE_INPUT.glob(f"{depth}/train_annotations"))
            if hits:
                return hits[0].parent
    start = Path(start or Path.cwd()).resolve()
    for cand in (start, *start.parents):
        for root in (cand, cand / DATA_DIR_NAME):
            if (root / "train_annotations").exists():
                return root
    raise FileNotFoundError("데이터 폴더를 찾지 못했다. 환경 변수 PILL_DATA_DIR로 경로를 지정하자.")


@dataclass(frozen=True)
class Paths:
    data_root: Path
    train_img: Path
    test_img: Path
    ann: Path
    out: Path  # 표·라벨 등 산출물
    fig: Path  # 그래프
    dataset: Path  # YOLO 데이터셋


def get_paths(repo_dir: Path, data_root: Path | None = None) -> Paths:
    """실행 환경에 맞는 경로 묶음을 만든다. 출력 폴더는 미리 만들어 둔다."""
    data_root = data_root or find_data_root(repo_dir)
    base = KAGGLE_WORKING if is_kaggle() else repo_dir
    out = KAGGLE_WORKING if is_kaggle() else repo_dir / "reports"
    paths = Paths(
        data_root=data_root,
        train_img=data_root / "train_images",
        test_img=data_root / "test_images",
        ann=data_root / "train_annotations",
        out=out,
        fig=out / "figures",
        dataset=base / "datasets" / "pill",
    )
    paths.fig.mkdir(parents=True, exist_ok=True)
    return paths
