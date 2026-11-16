"""분석 스크립트의 핵심 함수 테스트: 조합 단위 부트스트랩 재표본, WBF 짝 비교 입력, 근사 메타 검증."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _box(f, cid=1, x=10.0, score=0.9):
    return {"file_name": f, "category_id": cid, "x": x, "y": 10.0, "w": 100.0, "h": 100.0, "score": score}


def test_resample_counts_repeated_combo_as_separate_images():
    cmp = _load("compare_oof")
    df = pd.DataFrame([_box("a.png"), _box("b.png"), _box("c.png")])
    # 같은 조합(a, b)이 두 번 뽑히고 c는 뽑히지 않았다
    out = cmp.resample(df, [(0, ["a.png", "b.png"]), (1, ["a.png", "b.png"])])
    assert sorted(out["file_name"]) == ["a.png#0", "a.png#1", "b.png#0", "b.png#1"]
    assert cmp.resample(df, []).empty


def _fold_setup(tmp_path):
    folds = pd.DataFrame({"file_name": [f"{i}.png" for i in range(5)], "fold": range(5), "combo": [f"c{i}" for i in range(5)]})
    for exp, shift in (("m1", 0.0), ("m2", 4.0)):
        for k in range(5):
            d = tmp_path / (exp if k == 0 else f"{exp}_f{k}")
            d.mkdir()
            pd.DataFrame([_box(f"{k}.png", x=10.0 + shift)]).to_csv(d / "val_predictions.csv", index=False)
    return folds


def test_load_side_fuses_joined_experiments(tmp_path):
    cmp = _load("compare_oof")
    folds = _fold_setup(tmp_path)
    single = cmp.load_side("m1", tmp_path, folds, None, 0.6)
    fused = cmp.load_side("m1+m2", tmp_path, folds, [1, 1], 0.6)
    assert len(single) == len(fused) == 5
    assert fused["x"].tolist() == pytest.approx([12.0] * 5)  # 같은 점수 두 박스의 평균 위치


def test_meta_validate_picks_config_without_held_out_fold(tmp_path):
    ana = _load("analyze_oof")
    folds = pd.DataFrame({"file_name": [f"{i}.png" for i in range(5)], "fold": range(5)})
    gt = pd.DataFrame([_box(f"{i}.png") for i in range(5)]).drop(columns="score").assign(img_w=976, img_h=1280)
    good = pd.DataFrame([_box(f"{i}.png") for i in range(5)])
    bad = pd.DataFrame([_box(f"{i}.png", x=60.0) for i in range(5)])  # IoU 0.33: 모든 임계값에서 틀림
    out = tmp_path / "meta.csv"
    ana.meta_validate({(0.6, "1:1"): bad, (0.7, "2:1"): good}, gt, folds, {1}, out)
    res = pd.read_csv(out)
    per_fold = res[res["held_out_fold"].astype(str).str.fullmatch(r"\d")]
    assert set(per_fold["weights"]) == {"2:1"}
    assert per_fold["held_out_learnable_mAP75_95"].tolist() == pytest.approx([1.0] * 5)
