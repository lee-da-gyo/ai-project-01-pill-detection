"""핵심 함수 단위 테스트. 데이터 없이 돌아간다.

    python -m pytest tests -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pilldet.boxes import containment, iou_matrix, out_of_bounds, to_xyxy, xywh_to_yolo, yolo_to_xywh  # noqa: E402
from pilldet.data import camera_angle_of, combo_of, expected_codes  # noqa: E402
from pilldet.split import make_folds  # noqa: E402

FILE = "K-001900-016548-019607-029451_0_2_0_2_70_000_200.png"


# ---------- 파일명 파싱 ----------
def test_parse_file_name():
    assert combo_of(FILE) == "K-001900-016548-019607-029451"
    assert expected_codes(FILE) == [1900, 16548, 19607, 29451]
    assert camera_angle_of(FILE) == 70


# ---------- 박스 ----------
def test_iou_identical_and_disjoint():
    a = np.array([[0, 0, 10, 10]])
    assert iou_matrix(a, a)[0, 0] == pytest.approx(1.0)
    assert iou_matrix(a, np.array([[20, 20, 30, 30]]))[0, 0] == 0


def test_iou_half_overlap():
    # 겹침 50, 합집합 150 → 1/3
    assert iou_matrix([0, 0, 10, 10], [5, 0, 15, 10])[0, 0] == pytest.approx(1 / 3)


def test_iou_empty_inputs():
    assert iou_matrix(np.zeros((0, 4)), np.array([[0, 0, 1, 1]])).shape == (0, 1)


def test_containment_inner_box():
    assert containment([2, 2, 4, 4], [0, 0, 10, 10])[0, 0] == pytest.approx(1.0)


def test_out_of_bounds():
    xyxy = to_xyxy(pd.DataFrame({"x": [6567, 10], "y": [625, 10], "w": [311, 50], "h": [315, 50]}))
    assert out_of_bounds(xyxy, np.array([976, 976]), np.array([1280, 1280])).tolist() == [True, False]


@pytest.mark.parametrize("box", [(567, 625, 311, 315), (0, 0, 976, 1280), (100.5, 200.25, 50, 60)])
def test_yolo_roundtrip(box):
    back = yolo_to_xywh(*xywh_to_yolo(*box, 976, 1280), 976, 1280)
    assert np.allclose(back, box)


def test_yolo_normalized_range():
    cx, cy, w, h = xywh_to_yolo(567, 625, 311, 315, 976, 1280)
    assert all(0 <= v <= 1 for v in (cx, cy, w, h))


# ---------- 분할 ----------
def test_folds_keep_combo_together():
    """같은 조합의 사진은 반드시 같은 fold에 들어가야 한다 (누수 방지)."""
    rows = []
    for c in range(20):
        for angle in (70, 75, 90):
            rows.append({"file_name": f"K-{c:06d}-999999_0_2_0_2_{angle}_000_200.png", "category_id": c % 7,
                         "combo": f"K-{c:06d}-999999", "camera_la": angle})
    folds = make_folds(pd.DataFrame(rows), n_folds=5, seed=0)
    assert folds.groupby("combo")["fold"].nunique().max() == 1
    assert set(folds["fold"]) == set(range(5))


# ---------- 후처리 · 평가 · 제출 ----------
from pilldet.evaluate import coco_map  # noqa: E402
from pilldet.predict import limit_per_image  # noqa: E402
from pilldet.submission import to_submission, validate_submission  # noqa: E402

SPEC = {"columns": {k: k for k in ["annotation_id", "image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"]},
        "order": ["annotation_id", "image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"], "round": 2}


def _gt():
    return pd.DataFrame({"file_name": ["a.png", "a.png", "b.png"], "category_id": [1, 2, 1],
                         "x": [10, 300, 50], "y": [10, 300, 60], "w": [100, 120, 90], "h": [100, 110, 80],
                         "img_w": 976, "img_h": 1280})


def test_coco_map_perfect_and_wrong_class():
    gt = _gt()
    assert coco_map(gt.assign(score=1.0), gt)["mAP50_95"] == pytest.approx(1.0)
    wrong = gt.assign(score=1.0, category_id=[2, 1, 2])  # 위치는 맞고 클래스만 틀림
    assert coco_map(wrong, gt)["mAP50"] == pytest.approx(0.0)


def test_limit_per_image_keeps_top_unique():
    pred = pd.DataFrame({"file_name": ["a.png"] * 6, "category_id": [1, 1, 2, 3, 4, 5],
                         "x": 0, "y": 0, "w": 10, "h": 10, "score": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]})
    out = limit_per_image(pred, max_per_image=4)
    assert len(out) == 4 and out["category_id"].is_unique
    assert out["score"].tolist() == [0.9, 0.7, 0.6, 0.5]


def test_submission_format_and_validation():
    pred = pd.DataFrame({"file_name": ["2.png", "1.png"], "category_id": [1900, 3351],
                         "x": [1.234, 5.0], "y": [2.0, 6.0], "w": [10.0, 20.0], "h": [11.0, 21.0], "score": [0.5, 0.9]})
    sub = to_submission(pred, SPEC)
    assert list(sub.columns) == SPEC["order"]
    assert sub["image_id"].tolist() == [1, 2] and sub["annotation_id"].tolist() == [1, 2]
    sizes = {i: (976, 1280) for i in (1, 2, 3)}
    info = validate_submission(sub, SPEC, sizes, valid_categories={1900, 3351})
    assert info["rows"] == 2 and info["images_without_pred"] == 1
    with pytest.raises(ValueError):  # 테스트에 없는 이미지
        validate_submission(sub, SPEC, {1: (976, 1280)}, valid_categories={1900, 3351})
    out = sub.copy()
    out.loc[0, "bbox_x"] = 970.0  # 970 + 20 > 976 → 이미지 밖
    with pytest.raises(ValueError):
        validate_submission(out, SPEC, sizes, valid_categories={1900, 3351})
    nan = sub.copy()
    nan.loc[1, "score"] = float("nan")
    with pytest.raises(ValueError):
        validate_submission(nan, SPEC, sizes, valid_categories={1900, 3351})


def test_fold_of_parses_copy_paste_and_full():
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_experiment", Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.fold_of("data_fold0.yaml") == 0
    assert mod.fold_of("data_fold3_cp.yaml") == 3
    assert mod.fold_of("data_full.yaml") is None and mod.fold_of("data_full_cp.yaml") is None


# ---------- 2단계 결합 ----------
from pilldet.two_stage import combine  # noqa: E402


def test_combine_top_k_and_scores():
    boxes = pd.DataFrame({"file_name": ["a.png"], "category_id": [20], "x": [0.0], "y": [0.0], "w": [10.0], "h": [10.0],
                          "score": [0.8]})
    prob = np.array([[0.1, 0.6, 0.3]])
    ids = np.array([10, 20, 30])
    out = combine(boxes, prob, ids, top_k=2)
    assert out["category_id"].tolist() == [20, 30]
    assert out["score"].tolist() == pytest.approx([0.8 * 0.6, 0.8 * 0.3])
    # YOLO 1순위(20)에 비중 0.5를 섞으면 20의 확률이 0.5*0.6 + 0.5 = 0.8
    assert combine(boxes, prob, ids, top_k=1, yolo_weight=0.5)["score"].iloc[0] == pytest.approx(0.8 * 0.8)


# ---------- 앙상블 (WBF) ----------
from pilldet.ensemble import weighted_boxes_fusion  # noqa: E402


def _pred(x, score, cid=1, f="a.png"):
    return pd.DataFrame({"file_name": [f], "category_id": [cid], "x": [x], "y": [0.0], "w": [100.0], "h": [100.0], "score": [score]})


def test_wbf_fuses_overlapping_boxes_by_score():
    # 두 모델이 같은 알약을 x=0, x=10으로 그림 → 점수 가중 평균 위치, 두 모델 모두 찾았으므로 점수는 평균
    out = weighted_boxes_fusion([_pred(0, 0.9), _pred(10, 0.3)])
    assert len(out) == 1
    assert out["x"].iloc[0] == pytest.approx((0 * 0.9 + 10 * 0.3) / 1.2)
    assert out["score"].iloc[0] == pytest.approx((0.9 + 0.3) / 2)


def test_wbf_keeps_separate_classes_and_penalizes_single_model():
    out = weighted_boxes_fusion([_pred(0, 0.8, cid=1), _pred(0, 0.8, cid=2)])
    assert sorted(out["category_id"]) == [1, 2]
    # 한 모델만 찾은 박스는 점수가 모델 수만큼 줄어든다
    assert out["score"].tolist() == pytest.approx([0.4, 0.4])


def test_wbf_does_not_merge_distant_boxes():
    out = weighted_boxes_fusion([_pred(0, 0.9), _pred(500, 0.9)])
    assert len(out) == 2


def test_wbf_weighted_and_validation():
    # 가중치 2:1 → 좌표는 가중 점수(0.9·2, 0.3·1)로 평균, 점수 = 가중 점수 평균 × min(2,2) / 가중치 합 3
    out = weighted_boxes_fusion([_pred(0, 0.9), _pred(10, 0.3)], weights=[2, 1])
    assert out["x"].iloc[0] == pytest.approx((0 * 1.8 + 10 * 0.3) / 2.1)
    assert out["score"].iloc[0] == pytest.approx((1.8 + 0.3) / 2 * 2 / 3)
    with pytest.raises(ValueError):
        weighted_boxes_fusion([_pred(0, 0.9)], weights=[1, 1])
    with pytest.raises(ValueError):
        weighted_boxes_fusion([_pred(0, 0.9), _pred(0, 0.9)], weights=[1, 0])


def _load_script(name):
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_load_oof_rejects_incomplete_or_leaky_predictions(tmp_path):
    load_oof = _load_script("analyze_oof").load_oof
    folds = pd.DataFrame({"file_name": [f"{i}.png" for i in range(10)], "fold": [i % 5 for i in range(10)]})

    def write(k, files):
        d = tmp_path / ("exp" if k == 0 else f"exp_f{k}")
        d.mkdir(exist_ok=True)
        pd.DataFrame({"file_name": files, "category_id": 1, "x": 0, "y": 0, "w": 1, "h": 1, "score": 0.9}) \
            .to_csv(d / "val_predictions.csv", index=False)

    for k in range(5):
        write(k, list(folds.loc[folds["fold"] == k, "file_name"]))
    assert len(load_oof("exp", tmp_path, folds)) == 10          # 정상: 10장을 한 번씩

    write(0, ["0.png"])                                         # fold 0의 5.png 예측 누락
    with pytest.raises(SystemExit, match="예측이 하나도 없는"):
        load_oof("exp", tmp_path, folds)
    write(0, ["0.png", "5.png", "1.png"])                       # 다른 fold 이미지가 섞임 (누수)
    with pytest.raises(SystemExit, match="검증 이미지가 아닌"):
        load_oof("exp", tmp_path, folds)
    write(0, ["0.png", "5.png", "5.png"])                       # 같은 행 중복
    with pytest.raises(SystemExit, match="중복"):
        load_oof("exp", tmp_path, folds)


def test_snapshot_mismatches_catches_changed_and_default_keys(tmp_path):
    import yaml
    from pilldet.train import snapshot_mismatches
    cfg = {"name": "x", "model": "yolo11s.pt", "data": "data_fold0.yaml", "imgsz": 640}
    snap = {"model": "C:/w/yolo11s.pt", "data": "D:/ds/data_fold0.yaml", "imgsz": 640, "flipud": 0.0,
            "seed": 42, "deterministic": True, "plots": True, "device": "0", "project": "C:/runs"}
    (tmp_path / "args.yaml").write_text(yaml.safe_dump(snap), encoding="utf-8")
    assert snapshot_mismatches(cfg, tmp_path) == []                      # 경로·장치 차이는 무시
    assert any("imgsz" in d for d in snapshot_mismatches({**cfg, "imgsz": 960}, tmp_path))
    snap["flipud"] = 0.5                                                 # 설정 파일에 없는 키가 기본값과 다름
    (tmp_path / "args.yaml").write_text(yaml.safe_dump(snap), encoding="utf-8")
    assert any("flipud" in d for d in snapshot_mismatches(cfg, tmp_path))
    assert snapshot_mismatches(cfg, tmp_path / "none") != []             # 기록이 없으면 불일치


# ---------- 전처리 ----------
def test_preprocess_of_reads_dataset_suffix():
    from pilldet.preprocess import preprocess_of

    assert preprocess_of("data_fold0_clahe.yaml") == "clahe"
    assert preprocess_of("/abs/datasets/pill/data_full_clahe.yaml") == "clahe"
    assert preprocess_of("data_fold0.yaml") is None
    assert preprocess_of("data_fold0_cp.yaml") is None


def test_clahe_keeps_shape_and_hue_and_png_roundtrip(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from pilldet.preprocess import apply, load_bgr

    rng = np.random.default_rng(0)
    img = np.clip(rng.normal(120, 10, (64, 48, 3)), 0, 255).astype(np.uint8)  # 대비가 낮은 이미지
    out = apply(img, "clahe")
    assert out.shape == img.shape and out.dtype == np.uint8
    assert out.std() > img.std()  # 국소 대비가 커진다
    # 밝기(L) 채널만 바꾸므로 a·b(색) 채널은 거의 그대로다 (LAB↔BGR 반올림 오차만)
    lab_in, lab_out = cv2.cvtColor(img, cv2.COLOR_BGR2LAB), cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    assert np.abs(lab_in[..., 1:].astype(int) - lab_out[..., 1:].astype(int)).mean() < 2
    # 학습용으로 저장한 PNG와 예측 때 그때그때 전처리한 배열이 같아야 한다 (PNG 무손실)
    src = tmp_path / "원본.png"
    cv2.imencode(".png", img)[1].tofile(str(src))
    saved = tmp_path / "pre.png"
    cv2.imencode(".png", load_bgr(src, "clahe"))[1].tofile(str(saved))
    assert np.array_equal(load_bgr(saved), load_bgr(src, "clahe"))
    with pytest.raises(ValueError):
        apply(img, "unknown")
