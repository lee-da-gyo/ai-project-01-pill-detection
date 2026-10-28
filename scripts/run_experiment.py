"""실험 하나를 처음부터 끝까지 실행한다.

학습 → val 평가 (Ultralytics + COCO mAP) → 테스트 예측 → 제출 파일 생성·검사 → 실험 기록 추가

사용법
    python scripts/run_experiment.py --config configs/e0_baseline.yaml
    python scripts/run_experiment.py --config configs/e0_baseline.yaml --skip-train   # 학습 결과로 평가만 다시

산출물
    runs/<name>/                          학습 결과 (가중치, 그래프) – git 제외
    submissions/<name>.csv                Kaggle 제출 파일 – git 제외
    reports/experiments.csv               실험 기록 (한 줄 추가)
    reports/experiments/<name>/           지표, 클래스별 AP, 학습 곡선, 혼동 행렬
    reports/logs/<name>.log               실행 로그
"""

import os

# Windows + Anaconda에서 MKL과 PyTorch가 OpenMP 런타임을 각각 불러와 충돌하는 문제를 피한다 (Kaggle·Linux에서는 영향 없음)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

from pilldet.config import N_FOLDS, get_paths  # noqa: E402
from pilldet.data import list_test_images, load_clean_annotations  # noqa: E402
from pilldet.evaluate import coco_map  # noqa: E402
from pilldet.predict import predict_images  # noqa: E402
from pilldet.preprocess import preprocess_of  # noqa: E402
from pilldet.submission import load_submission_spec, to_submission, validate_submission  # noqa: E402
from pilldet.train import best_epoch, load_config, snapshot_mismatches, train  # noqa: E402

PLOTS = ["results.png", "confusion_matrix_normalized.png", "BoxPR_curve.png", "val_batch0_pred.jpg", "val_batch0_labels.jpg"]


def setup_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("experiment")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, mode="w", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def fold_of(data_yaml: str) -> int | None:
    """data_fold0.yaml → 0, data_fold0_cp.yaml → 0, data_full*.yaml → None (전체 학습, 검증 없음)."""
    m = re.match(r"data_fold(\d+)", Path(data_yaml).stem)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--skip-train", action="store_true", help="이미 학습한 가중치로 평가·제출만 다시 한다")
    ap.add_argument("--fold", type=int, default=None,
                    help="설정 파일의 fold를 바꿔 실행한다 (5-Fold 교차검증용). 실험 이름에 _f<k>가 붙는다")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.fold is not None:
        if fold_of(cfg["data"]) is None:
            raise ValueError("전체 학습 설정에는 --fold를 쓸 수 없다")
        if not 0 <= args.fold < N_FOLDS:
            raise ValueError(f"--fold는 0~{N_FOLDS - 1} 범위여야 한다: {args.fold}")
        cfg["data"] = re.sub(r"data_fold\d+", f"data_fold{args.fold}", cfg["data"])
        # 이름 규칙: fold 0은 기본 이름 그대로, fold 1~4는 _f<k> (summarize_cv.py와 같은 규칙)
        if args.fold != 0:
            cfg["name"] = f"{cfg['name']}_f{args.fold}"
        cfg["description"] = re.sub(r"fold \d+", f"fold {args.fold}", cfg.get("description", ""))
    name = cfg["name"]
    P = get_paths(REPO_DIR)
    runs_dir = REPO_DIR / "runs"
    exp_dir = P.out / "experiments" / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logger(P.out / "logs" / f"{name}.log")
    log.info(f"실험 {name}: {cfg.get('description', '')}")
    log.info(f"설정: {cfg}")

    # 1. 학습
    # 전체 데이터 학습(data_full*)은 검증 데이터가 학습에 포함되므로 검증 점수로 모델을 고르면 안 된다.
    # → 검증을 끄고, 교차검증으로 정한 고정 에포크만큼 학습한 마지막 가중치(last.pt)를 쓴다.
    k = fold_of(cfg["data"])
    overrides = {} if k is not None else {"val": False, "patience": 0}
    t0 = time.time()
    if args.skip_train and (diffs := snapshot_mismatches(cfg, runs_dir / name, **overrides)):
        raise SystemExit(f"{name}: 지금 설정이 학습 당시와 다르다 → 다시 학습하거나 설정을 되돌리자\n  " + "\n  ".join(diffs))
    run_dir = runs_dir / name if args.skip_train else train(cfg, P.dataset, runs_dir, **overrides)
    train_min = (time.time() - t0) / 60
    if k is not None:
        best = best_epoch(run_dir)
        log.info(f"[1/4] 학습 완료 ({train_min:.1f}분): 최고 에포크 {best['epoch']} / {best['epochs_run']} "
                 f"| P {best['precision']:.3f} R {best['recall']:.3f} mAP50 {best['mAP50']:.4f} mAP50-95 {best['mAP50_95']:.4f}")
        weights = run_dir / "weights" / "best.pt"
    else:
        best = {"epoch": cfg["epochs"], "epochs_run": cfg["epochs"], "precision": float("nan"), "recall": float("nan"),
                "mAP50": float("nan"), "mAP50_95": float("nan")}
        log.info(f"[1/4] 전체 데이터 학습 완료 ({train_min:.1f}분): 고정 {cfg['epochs']} 에포크, last.pt 사용 (검증 없음)")
        weights = run_dir / "weights" / "last.pt"
    # 전체 데이터 학습의 PR 곡선·혼동 행렬은 학습에 쓴 이미지로 계산되므로 손실 곡선만 남긴다
    for p in PLOTS if k is not None else ["results.png"]:
        if (run_dir / p).exists():
            shutil.copy2(run_dir / p, exp_dir / p)

    class_map = pd.read_csv(P.out / "class_map.csv")
    gt_all = load_clean_annotations(P.out / "annotations_clean.json")
    imgsz = cfg.get("imgsz", 640)
    # 전처리 데이터셋(data_fold0_clahe.yaml)으로 학습했으면 val·테스트 예측에도 같은 전처리를 적용한다
    pre = preprocess_of(cfg["data"])
    if pre:
        log.info(f"      전처리 {pre}: val·테스트 이미지에도 같은 함수로 적용해 예측")

    # 2. val 평가 – Ultralytics와 별개로 COCO mAP를 직접 계산해 이후 앙상블·2단계 모델과 같은 기준으로 비교한다
    coco = {"mAP50_95": None, "mAP50": None, "mAP75": None, "seen_mAP50_95": None}
    single = bool(cfg.get("single_cls"))
    if k is not None and single:
        # 1-클래스 검출기: 약 이름이 없으므로 위치만 평가한다 (모든 클래스를 0으로 통일)
        val_files = [l.strip().split("/")[-1] for l in (P.dataset / f"fold{k}_val.txt").read_text().splitlines() if l.strip()]
        pred_val = predict_images(weights, [P.train_img / f for f in val_files], class_map, imgsz=imgsz, preprocess=pre)
        gt_val = gt_all[gt_all["file_name"].isin(val_files)]
        coco = coco_map(pred_val.assign(category_id=0), gt_val.assign(category_id=0))
        coco.pop("per_class")
        coco["seen_mAP50_95"] = None
        log.info(f"[2/4] fold {k} 위치만 평가 (1-클래스): mAP50-95 {coco['mAP50_95']:.4f} | mAP50 {coco['mAP50']:.4f}")
    elif k is not None:
        val_files = [l.strip().split("/")[-1] for l in (P.dataset / f"fold{k}_val.txt").read_text().splitlines() if l.strip()]
        pred_val = predict_images(weights, [P.train_img / f for f in val_files], class_map, imgsz=imgsz, preprocess=pre)
        gt_val = gt_all[gt_all["file_name"].isin(val_files)]
        # 보조 진단 지표: 학습 fold에 있는 클래스만으로 다시 평가한다 (가장 어려운 "처음 보는 약"을 뺀 조건부 점수라
        # 낙관적이다). 모델 선택의 주 지표는 전체 mAP이고, 이 값은 오류를 분해해 보는 용도로만 쓴다.
        seen = set(gt_all.loc[~gt_all["file_name"].isin(val_files), "category_id"])
        seen_map = coco_map(pred_val[pred_val["category_id"].isin(seen)], gt_val[gt_val["category_id"].isin(seen)])
        coco = coco_map(pred_val, gt_val)
        coco["seen_mAP50_95"] = seen_map["mAP50_95"]
        pc = coco.pop("per_class").merge(class_map[["category_id", "dl_name", "n_boxes"]], on="category_id")
        pc.sort_values("AP50_95").to_csv(exp_dir / "per_class_ap.csv", index=False, encoding="utf-8-sig")
        pred_val.to_csv(exp_dir / "val_predictions.csv", index=False)
        log.info(f"[2/4] fold {k} COCO 평가: mAP50-95 {coco['mAP50_95']:.4f} | mAP50 {coco['mAP50']:.4f} | mAP75 {coco['mAP75']:.4f}"
                 f" | 학습에 있는 클래스만 mAP50-95 {coco['seen_mAP50_95']:.4f}")
        log.info(f"      AP50-95 하위 5개 클래스:\n{pc.sort_values('AP50_95').head(5)[['dl_name', 'n_boxes', 'AP50_95']].to_string(index=False)}")

    # 3. 테스트 예측 → 제출 파일 (1-클래스 검출기는 약 이름이 없어 제출하지 않는다)
    if single:
        log.info("[3/4] 1-클래스 검출기 → 제출 파일 생략 (2단계 모델에서 분류기와 결합해 제출)")
    else:
        test_paths = list_test_images(P.test_img)
        pred_test = predict_images(weights, test_paths, class_map, imgsz=imgsz, preprocess=pre)
        spec = load_submission_spec(REPO_DIR / "configs" / "submission.yaml")
        sub = to_submission(pred_test, spec)
        test_sizes = {int(p.stem): Image.open(p).size for p in test_paths}
        check = validate_submission(sub, spec, test_sizes, set(class_map["category_id"]))
        sub_dir = REPO_DIR / "submissions"
        sub_dir.mkdir(exist_ok=True)
        sub.to_csv(sub_dir / f"{name}.csv", index=False)
        log.info(f"[3/4] 제출 파일: submissions/{name}.csv {check}"
                 + ("" if spec.get("verified") else "  ⚠️ 제출 형식 미확인 (configs/submission.yaml)"))

    # 4. 실험 기록
    row = {"id": name.split("_")[0].upper(), "name": name, "date": date.today().isoformat(),
           "description": cfg.get("description", ""), "model": cfg["model"], "imgsz": imgsz, "fold": k,
           "epochs_run": best["epochs_run"], "best_epoch": best["epoch"],
           "precision": round(best["precision"], 4), "recall": round(best["recall"], 4),
           "yolo_mAP50": round(best["mAP50"], 4), "yolo_mAP50_95": round(best["mAP50_95"], 4),
           "coco_mAP50": None if coco["mAP50"] is None else round(coco["mAP50"], 4),
           "coco_mAP50_95": None if coco["mAP50_95"] is None else round(coco["mAP50_95"], 4),
           "seen_mAP50_95": None if coco["seen_mAP50_95"] is None else round(coco["seen_mAP50_95"], 4),
           "train_min": round(train_min, 1), "kaggle_public": None}
    log_path = P.out / "experiments.csv"
    exp = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    prev = exp[exp["name"] == name] if "name" in exp else pd.DataFrame()
    if args.skip_train and not prev.empty:  # 평가만 다시 할 때는 원래 학습 시간과 Kaggle 점수를 유지한다
        row["train_min"] = prev["train_min"].iloc[0]
        row["kaggle_public"] = prev["kaggle_public"].iloc[0]
    exp = pd.concat([exp[exp.get("name", pd.Series(dtype=str)) != name], pd.DataFrame([row])], ignore_index=True)
    exp.to_csv(log_path, index=False, encoding="utf-8-sig")
    log.info(f"[4/4] 실험 기록 추가: reports/experiments.csv ({len(exp)}개 실험)")


if __name__ == "__main__":
    main()
