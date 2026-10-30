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
from pilldet.evaluate import classify_errors, coco_map  # noqa: E402
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
    assert coco_map(gt.assign(score=1.0), gt)["per_iou"][0.95] == pytest.approx(1.0)


def test_classify_errors_types():
    gt = _gt()
    pred = pd.DataFrame({"file_name": ["a.png", "a.png", "a.png", "a.png", "b.png"],
                         "category_id": [1, 1, 1, 9, 1],
                         "x": [10, 12, 700, 300, 50], "y": [10, 10, 700, 300, 60], "w": [100, 100, 50, 120, 90],
                         "h": [100, 100, 50, 110, 80], "score": [0.9, 0.8, 0.7, 0.95, 0.1]})
    g, p = classify_errors(pred, gt, conf=0.25)
    assert g["type"].tolist() == ["정답", "분류 오류", "놓침"]  # b.png 예측은 신뢰도 미달
    assert g.loc[1, "pred_category_id"] == 9
    assert p["type"].tolist() == ["다른 약 위 오분류", "정답 매칭", "중복", "배경"]  # 점수 내림차순
    assert p["near_gt_category_id"].isna().tolist() == [False, False, False, True]  # 배경 예측은 어느 약에도 속하지 않는다


def test_classify_errors_one_prediction_matches_one_gt():
    # 겹친 두 알약 위에 예측이 하나뿐이면 정답 하나만 맞힌 것으로 센다 (예측 재사용 금지)
    gt = pd.DataFrame({"file_name": "a.png", "category_id": [1, 2, 3], "x": [0, 5, 300], "y": [0, 0, 300],
                       "w": 100, "h": 100, "img_w": 976, "img_h": 1280})
    pred = pd.DataFrame({"file_name": "a.png", "category_id": [1, 3], "x": [2, 350], "y": [0, 300], "w": 100, "h": 100,
                         "score": [0.9, 0.8]})
    g, p = classify_errors(pred, gt, conf=0.25)
    assert g["type"].tolist() == ["정답", "놓침", "위치 오류"]  # 예측 0은 IoU가 더 큰 정답 0에만 배정
    assert p["type"].tolist() == ["정답 매칭", "위치 부정확"]
    assert g.loc[2, "matched_iou"] == pytest.approx(50 / 150)


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


# ---------- 라벨 감사 D·E ----------
def _review_case(rows=None):
    """같은 좌표에 두 약(1, 2)이 라벨되고 다른 알약 하나는 라벨이 없는 이미지 + 두 라벨이 뒤바뀐 이미지."""
    from types import SimpleNamespace

    from pilldet.clean import REVIEW_COLUMNS, LabelCleaner

    raw = pd.DataFrame({"file_name": ["K-000001-000002_0_2_0_2_70_000_200.png"] * 2 + ["K-000003-000004_0_2_0_2_75_000_200.png"] * 2,
                        "category_id": [1, 2, 3, 4], "x": [10, 10, 10, 300], "y": [10, 10, 10, 300], "w": 100, "h": 100,
                        "img_w": 976, "img_h": 1280, "dl_name": ["a", "b", "c", "d"], "dl_name_en": ["a", "b", "c", "d"],
                        "source": "original"})
    box_a, box_b = "[10, 10, 100, 100]", "[300, 300, 100, 100]"
    if rows is None:
        rows = [["R001", "D_duplicate", raw.file_name[0], 2, 1, "relocate", box_a, box_a, "다른 각도에서 확인", "검수자", "2026-10-30", "confirmed"],
                ["R002", "E_swapped", raw.file_name[2], 3, 4, "swap", box_a, box_b, "다른 각도에서 확인", "검수자", "2026-10-30", "confirmed"]]
    review = pd.DataFrame(rows, columns=REVIEW_COLUMNS)
    cleaner = LabelCleaner(raw, Path("."), review=review)
    cleaner.segs = {raw.file_name[0]: SimpleNamespace(boxes=np.array([[10, 10, 110, 110], [500, 600, 600, 700]], float)),
                    raw.file_name[2]: SimpleNamespace(boxes=np.array([[10, 10, 110, 110], [300, 300, 400, 400]], float))}
    return raw, cleaner


def test_cross_class_overlaps_finds_duplicate_coordinates():
    from pilldet.clean import cross_class_overlaps

    raw, _ = _review_case()
    d = cross_class_overlaps(raw)
    assert d[["category_id", "other_category_id"]].values.tolist() == [[1, 2]]  # 뒤바뀐 라벨은 좌표로 잡히지 않는다


def test_apply_review_relocates_swaps_and_is_idempotent():
    from pilldet.clean import check_integrity, cross_class_overlaps

    raw, cleaner = _review_case()
    once = cleaner.apply_review(raw)
    moved = once[once.category_id == 2].iloc[0]
    assert [moved.x, moved.y, moved.w, moved.h] == [500, 600, 100, 100]  # 라벨 없는 알약으로 이동 (보정값 0)
    assert once.set_index("category_id").loc[3, "x"] == 300 and once.set_index("category_id").loc[4, "x"] == 10
    assert cross_class_overlaps(once).empty
    assert len(cleaner._fixes) == 3 and {f["review_id"] for f in cleaner._fixes} == {"R001", "R002"}
    # 같은 원본·같은 결정이면 결과와 수정 로그가 같다
    raw2, cleaner2 = _review_case()
    pd.testing.assert_frame_equal(cleaner2.apply_review(raw2), once)
    assert cleaner2._fixes == cleaner._fixes
    # 이미 고친 결과에 다시 적용해도 바뀌지 않는다 – swap도 원상복귀하지 않는다 (검수 때 본 좌표와 달라 건너뜀)
    again = cleaner.apply_review(once)
    pd.testing.assert_frame_equal(again, once)
    assert len(cleaner._fixes) == 3
    assert sum("검수 때 본 좌표와 다름" in n for n in cleaner._notes) == 2
    with pytest.raises(AssertionError, match="같은 좌표"):
        check_integrity(raw, sorted(raw.file_name.unique()))


def test_apply_review_skips_when_labels_differ_from_reviewed_boxes():
    """원본이 바뀌어 검수 때 본 좌표와 다르면 엉뚱한 박스를 고치지 않는다."""
    raw, cleaner = _review_case()
    shifted = raw.copy()
    shifted.loc[shifted.category_id == 4, "x"] = 320  # 1px 허용 오차보다 크게 다름
    out = cleaner.apply_review(shifted)
    assert out.set_index("category_id").loc[3, "x"] == 10  # 교환하지 않음
    assert any(n.startswith("R002 적용 안 함") for n in cleaner._notes)


def test_apply_review_holds_ambiguous_cases():
    """박스 수나 라벨 없는 알약 수가 결정과 맞지 않으면 고치지 않고 notes에 남긴다."""
    from types import SimpleNamespace

    raw, cleaner = _review_case()
    cleaner.segs[raw.file_name[0]] = SimpleNamespace(boxes=np.array([[10, 10, 110, 110], [500, 600, 600, 700], [700, 900, 800, 1000]], float))
    out = cleaner.apply_review(raw)  # 라벨 없는 알약 2개 → D 보류
    assert out.set_index("category_id").loc[2, "x"] == 10
    assert any("R001 보류 (라벨 없는 알약 2개)" in n for n in cleaner._notes)
    raw3, cleaner3 = _review_case()
    out = cleaner3.apply_review(raw3[raw3.category_id != 4])  # 교환할 박스 하나가 없음
    assert any("R002 적용 불가 (박스 1·0개)" in n for n in cleaner3._notes)


def test_review_schema_and_keep_overlap():
    from pilldet.clean import check_integrity

    raw, _ = _review_case()
    base = ["D_duplicate", raw.file_name[0], 2, 1]
    box = "[10, 10, 100, 100]"
    tail = ["확인", "검수자", "2026-10-30"]
    # pending은 적용하지 않는다
    _, c = _review_case([["R001", *base, "relocate", box, box, *tail, "pending"]])
    pd.testing.assert_frame_equal(c.apply_review(raw), raw)
    # 형식 오류는 조용히 넘어가지 않는다
    for rows, msg in [([["R001", *base, "move", box, box, *tail, "confirmed"]], "action"),
                      ([["R001", *base, "relocate", box, box, *tail, "done"]], "status"),
                      ([["R001", *base, "relocate", box, box, *tail, "confirmed"],
                        ["R001", "E_swapped", raw.file_name[2], 3, 4, "swap", box, box, *tail, "confirmed"]], "review_id"),
                      ([["R001", *base, "relocate", box, box, *tail, "confirmed"],
                        ["R002", "D_duplicate", raw.file_name[0], 1, 2, "relocate", box, box, *tail, "confirmed"]], "두 개"),
                      ([["R001", *base, "relocate", "[10, 10]", box, *tail, "confirmed"]], "형식")]:
        with pytest.raises(ValueError, match=msg):
            _review_case(rows)
    # 실제로 겹친 알약으로 확인(keep_overlap)한 쌍은 고치지 않고, 무결성 검사도 통과시킨다
    _, c = _review_case([["R001", *base, "keep_overlap", box, box, *tail, "confirmed"]])
    kept = c.apply_review(raw)
    pd.testing.assert_frame_equal(kept, raw)
    files = sorted(raw.file_name.unique())
    assert check_integrity(kept, files, c.review)["boxes"] == 4
    with pytest.raises(AssertionError, match="같은 좌표"):  # 결정 없이는 같은 겹침이 미해결 D 후보다
        check_integrity(kept, files)


def test_keep_overlap_only_exempts_reviewed_coordinates():
    """keep_overlap도 검수 때 본 좌표 그대로일 때만 겹침을 허용한다 – 두 박스가 함께 다른 곳으로 옮겨진 겹침은 새 후보다."""
    from pilldet.clean import check_integrity

    raw, _ = _review_case()
    box = "[10, 10, 100, 100]"
    _, c = _review_case([["R001", "D_duplicate", raw.file_name[0], 2, 1, "keep_overlap", box, box, "확인", "검수자", "2026-10-30", "confirmed"]])
    files = sorted(raw.file_name.unique())
    moved = raw.copy()
    moved.loc[moved.category_id.isin([1, 2]), ["x", "y"]] = 400  # 같은 두 약이 새 위치에서 다시 겹침
    pd.testing.assert_frame_equal(c.apply_review(moved), moved)
    assert any(n.startswith("R001 적용 안 함") for n in c._notes)
    with pytest.raises(AssertionError, match="같은 좌표"):
        check_integrity(moved, files, c.review)
    one_moved = raw.copy()
    one_moved.loc[one_moved.category_id == 2, "x"] = 13  # 아직 겹치지만(IoU ≥ 0.9) 한 박스가 허용 오차 밖
    with pytest.raises(AssertionError, match="같은 좌표"):
        check_integrity(one_moved, files, c.review)


def test_review_decision_matches_exact_pair_only():
    """한 약이 두 결정에 나와도 감사 후보에는 정확히 같은 쌍의 결정만 붙고, 같은 박스를 고친 결정은 관련 결정으로 구분한다."""
    from pilldet.clean import REVIEW_COLUMNS, UNREVIEWED, review_decision

    f, box = "K-000001-000002-000003_0_2_0_2_70_000_200.png", "[10, 10, 100, 100]"
    tail = ["확인", "검수자", "2026-10-30", "confirmed"]
    review = pd.DataFrame([["R001", "D_duplicate", f, 1, 2, "relocate", box, box, *tail],
                           ["R002", "E_swapped", f, 3, 1, "swap", box, box, *tail]], columns=REVIEW_COLUMNS)
    assert review_decision(review, f, 2, 1) == "confirmed: R001 D_duplicate relocate"  # 순서와 무관
    assert review_decision(review, f, 1, 3) == "confirmed: R002 E_swapped swap"
    assert review_decision(review, f, 1, 4) == "쌍 결정 없음 - 같은 박스를 고친 관련 결정: R001 relocate, R002 swap"
    assert review_decision(review, f, 2, 3) == UNREVIEWED  # 2는 relocate의 상대 박스라 바뀌지 않음
    assert review_decision(review, "other.png", 1, 2) == UNREVIEWED


def test_new_candidate_is_unreviewed_until_keep_label_is_recorded():
    """검수 기록이 없는 새 후보는 기각(라벨이 맞음)으로 단정하지 않고, keep_label 결정을 적어야 기각으로 표시된다."""
    from pilldet.clean import REVIEW_COLUMNS, UNREVIEWED, review_decision

    raw, _ = _review_case()
    f, box = raw.file_name[2], "[10, 10, 100, 100]"
    empty = pd.DataFrame(columns=REVIEW_COLUMNS)
    assert review_decision(empty, f, 3, 4) == UNREVIEWED
    rows = [["R009", "E_model_audit", f, 3, 4, "keep_label", box, "[300, 300, 100, 100]", "확인", "검수자", "2026-10-30", "confirmed"]]
    _, c = _review_case(rows)
    assert review_decision(c.review, f, 4, 3) == "confirmed: R009 E_model_audit keep_label"
    pd.testing.assert_frame_equal(c.apply_review(raw), raw)  # 라벨은 그대로


def test_console_messages_encode_in_cp949():
    """Windows 기본 콘솔(cp949)에서 print · 로그가 UnicodeEncodeError로 멈추지 않도록, 출력 문자열에 cp949 밖 문자가 없는지 본다."""
    import ast

    from pilldet.clean import REVIEW_COLUMNS, review_decision

    repo, bad = Path(__file__).resolve().parents[1], []
    for f in sorted([*(repo / "scripts").glob("*.py"), *(repo / "src" / "pilldet").glob("*.py")]):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            is_out = isinstance(node, ast.Call) and (getattr(node.func, "attr", None) or getattr(node.func, "id", "")) in                 {"print", "log", "info", "warning", "append"}
            if is_out or isinstance(node, ast.Return):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        try:
                            sub.value.encode("cp949")
                        except UnicodeEncodeError:
                            bad.append(f"{f.name}:{sub.lineno}")
    assert not bad, bad
    review = pd.DataFrame([["R001", "D_duplicate", "a.png", 1, 2, "relocate", "[0, 0, 1, 1]", "[0, 0, 1, 1]", "확인", "검수자",
                            "2026-10-30", "confirmed"]], columns=REVIEW_COLUMNS)
    for cid, other in [(1, 2), (1, 3), (2, 3)]:  # 직접 결정 · 관련 결정 · 미검토 문구 모두
        review_decision(review, "a.png", cid, other).encode("cp949")
