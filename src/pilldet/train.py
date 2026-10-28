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
# 학습 결과와 무관하게 실행마다 달라지는 기록 (스냅샷 대조에서 제외)
RUN_ONLY_KEYS = {"model", "data", "project", "name", "save_dir", "exist_ok", "resume", "cfg", "tracker", "device"}


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    missing = {"name", "model", "data"} - cfg.keys()
    if missing:
        raise ValueError(f"설정 파일에 필수 키가 없다: {missing}")
    return cfg


def train_args(cfg: dict, **overrides) -> dict:
    """설정 파일 → Ultralytics train() 인자 (학습과 스냅샷 대조가 같은 규칙을 쓰도록 한 곳에 둔다)."""
    return {**DEFAULTS, **{k: v for k, v in cfg.items() if k not in META_KEYS}, **overrides}


def snapshot_mismatches(cfg: dict, run_dir: Path, **overrides) -> list[str]:
    """학습 당시 설정 기록(runs/<name>/args.yaml)과 지금 설정이 다른 항목 목록. 기록이 없으면 그 자체가 불일치다.

    학습을 건너뛰고 기존 가중치로 평가할 때, 설정 파일이 그 뒤에 바뀌었으면 다른 실험의 결과로 기록되는 것을 막는다.
    model·data는 경로가 환경마다 달라 파일 이름만 비교한다.
    설정 파일에 없는 키는 Ultralytics 기본값을 기대값으로 삼아, 예전 실험에서만 바꿨던 증강 같은 차이도 잡는다.
    """
    from ultralytics.cfg import DEFAULT_CFG_DICT

    path = Path(run_dir) / "args.yaml"
    if not path.exists():
        return [f"학습 기록 없음: {path}"]
    with open(path, encoding="utf-8") as f:
        snap = yaml.safe_load(f)
    diffs = [f"{k}: 설정 {Path(str(cfg[k])).name} ≠ 학습 {Path(str(snap.get(k))).name}"
             for k in ("model", "data") if Path(str(cfg[k])).name != Path(str(snap.get(k))).name]
    expected = {**DEFAULT_CFG_DICT, **train_args(cfg, **overrides)}
    for k, v in expected.items():
        if k in snap and k not in RUN_ONLY_KEYS and snap[k] != v:
            diffs.append(f"{k}: 설정 {v!r} ≠ 학습 {snap[k]!r}")
    return diffs


def train(cfg: dict, dataset_dir: Path, runs_dir: Path, **overrides) -> Path:
    """설정대로 학습하고 실행 폴더(runs/<name>)를 반환한다."""
    from ultralytics import YOLO

    args = train_args(cfg, **overrides)
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
