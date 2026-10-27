"""데이터 준비 파이프라인을 한 번에 실행한다.

라벨 병합 → 알약 분할 → 라벨 감사·수정 → 무결성 검사 → 조합 단위 K-Fold → YOLO/COCO 내보내기 → 왕복 검증

사용법
    python scripts/prepare_data.py                       # 데이터 경로 자동 탐색
    python scripts/prepare_data.py --data-dir <경로>      # 직접 지정

실행 로그는 reports/logs/prepare_data.log 에도 저장된다.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

from pilldet.clean import LabelCleaner, check_integrity  # noqa: E402
from pilldet.config import N_FOLDS, SEED, get_paths  # noqa: E402
from pilldet.data import list_train_images, load_annotations  # noqa: E402
from pilldet.export import build_class_map, export_coco, export_yolo, verify_yolo_roundtrip  # noqa: E402
from pilldet.split import fold_summary, make_folds  # noqa: E402


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("prepare_data")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, mode="w", encoding="utf-8")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=None, help="train_images, train_annotations가 있는 폴더")
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    paths = get_paths(REPO_DIR, args.data_dir)
    log = setup_logger(paths.out / "logs" / "prepare_data.log")
    t0 = time.time()
    log.info(f"데이터 경로: {paths.data_root}")

    # 1. 라벨 병합
    raw = load_annotations(paths.ann)
    train_files = list_train_images(paths.train_img)
    log.info(f"[1/6] 라벨 병합: 이미지 {raw.file_name.nunique()}장, 박스 {len(raw)}개, 클래스 {raw.category_id.nunique()}개")

    # 2~3. 분할 + 감사 + 수정
    cleaner = LabelCleaner(raw, paths.train_img)
    res = cleaner.run()
    log.info(f"[2/6] 라벨 감사: 테두리 보정값 {res.calib.round(1).tolist()}")
    for r in res.fixes.itertuples():
        log.info(f"      수정 {r.type:16s} {r.file_name[:40]:40s} {r.name[:14]:14s} {r.before} → {r.after}")
    for n in res.notes:
        log.info(f"      보류 {n}")
    res.fixes.to_csv(paths.out / "label_fixes.csv", index=False, encoding="utf-8-sig")

    # 4. 무결성 검사
    stats = check_integrity(res.clean, train_files)
    log.info(f"[3/6] 무결성 검사 통과: {stats}")

    # 5. 조합 단위 K-Fold
    folds = make_folds(res.clean, args.folds, args.seed)
    folds[["file_name", "combo", "camera_la", "fold"]].to_csv(paths.out / "folds.csv", index=False, encoding="utf-8-sig")
    log.info(f"[4/6] 조합 단위 {args.folds}-Fold (누수 0)\n{fold_summary(folds).to_string(index=False)}")

    # 6. 내보내기 + 검증
    class_map = build_class_map(res.clean)
    class_map.to_csv(paths.out / "class_map.csv", index=False, encoding="utf-8-sig")
    info = export_yolo(res.clean, folds, class_map, paths.train_img, paths.dataset)
    coco = export_coco(res.clean, class_map, train_files, paths.out / "annotations_clean.json")
    log.info(f"[5/6] 내보내기: YOLO 이미지 {info['images']}장({info['link_mode']}), 라벨 {info['labels']}개 / COCO {coco}")

    err = verify_yolo_roundtrip(paths.dataset, res.clean, class_map)
    assert err < 0.01, err
    log.info(f"[6/6] 왕복 검증 통과: 최대 좌표 오차 {err:.5f}px")
    log.info(f"완료 ({time.time() - t0:.1f}초) → {paths.dataset}")


if __name__ == "__main__":
    main()
