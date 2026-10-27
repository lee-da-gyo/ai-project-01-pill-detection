"""YOLO 학습 – 설정 파일(configs/*.yaml) 하나가 실험 하나다.

설정 파일의 키 중 name, model, data, description을 뺀 나머지는 Ultralytics `train()` 인자로 그대로 전달한다.
그래서 증강·하이퍼파라미터 실험은 코드 수정 없이 설정 파일만 바꿔서 한다.
"""

from pathlib import Path

import pandas as pd
import yaml

from .config import SEED

META_KEYS = {"name", "model", "data", "description"}
DEFAULTS = {"seed": SEED, "deterministic": True, "plots": True, "exist_ok": True}


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    missing = {"name", "model", "data"} - cfg.keys()
    assert not missing, f"설정 파일에 필수 키가 없다: {missing}"
    return cfg


def train(cfg: dict, dataset_dir: Path, runs_dir: Path, **overrides) -> Path:
    """설정대로 학습하고 실행 폴더(runs/<name>)를 반환한다."""
    from ultralytics import YOLO

    args = {**DEFAULTS, **{k: v for k, v in cfg.items() if k not in META_KEYS}, **overrides}
    model = YOLO(cfg["model"])
    model.train(data=str(Path(dataset_dir) / cfg["data"]), project=str(runs_dir), name=cfg["name"], **args)
    return Path(runs_dir) / cfg["name"]


def read_results(run_dir: Path) -> pd.DataFrame:
    """에포크별 학습 기록 (Ultralytics results.csv)."""
    df = pd.read_csv(Path(run_dir) / "results.csv")
    df.columns = df.columns.str.strip()
    return df


def best_epoch(run_dir: Path) -> dict:
    """val mAP@50-95가 가장 높은 에포크의 지표."""
    df = read_results(run_dir)
    r = df.loc[df["metrics/mAP50-95(B)"].idxmax()]
    return {"epoch": int(r["epoch"]), "precision": r["metrics/precision(B)"], "recall": r["metrics/recall(B)"],
            "mAP50": r["metrics/mAP50(B)"], "mAP50_95": r["metrics/mAP50-95(B)"], "epochs_run": len(df)}
