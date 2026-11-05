"""경구약제 검출 데모 – 사진을 올리면 알약 위치·약 이름·신뢰도를 보여 준다.

현재 최종 후보와 같은 구성: 232장 전체로 학습한 YOLO11s + YOLO11m의 예측을 WBF(11s:11m = 2:1, IoU 0.6)로 합친다.
처음 보는 약도 학습한 약 이름 중 하나로 답할 수 있으므로(보고서 8.5절), 신뢰도가 낮거나 같은 알약에
이름이 둘 이상 붙으면 "사용자 확인 필요"로 표시한다. 이 표시는 오답을 모두 거르지 못한다.

실행
    streamlit run app/demo.py
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

from pilldet.boxes import iou_matrix, to_xyxy  # noqa: E402
from pilldet.config import get_paths  # noqa: E402
from pilldet.data import list_test_images  # noqa: E402
from pilldet.ensemble import weighted_boxes_fusion  # noqa: E402
from pilldet.predict import predict_images  # noqa: E402

MODELS = [REPO_DIR / "runs" / "final_yolo11s" / "weights" / "last.pt", REPO_DIR / "runs" / "final_yolo11m" / "weights" / "last.pt"]
WBF_WEIGHTS = [2, 1]  # 11s:11m, 5-Fold OOF 근사 메타 검증으로 고른 값 (README 7장)
WBF_IOU = 0.6
SHOW_CONF = 0.25  # 이 신뢰도 이상인 박스만 보여 준다 (Ultralytics 예측 기본값)
CHECK_CONF = 0.5  # 이보다 낮으면 사용자 확인 필요 (운영 임계값이 아니라 데모용 표시 기준)


def detect(image_path: Path, class_map: pd.DataFrame) -> pd.DataFrame:
    preds = [predict_images(w, [image_path], class_map) for w in MODELS]
    p = weighted_boxes_fusion(preds, WBF_WEIGHTS, iou_thr=WBF_IOU)
    return p[p.score >= SHOW_CONF].sort_values("score", ascending=False).reset_index(drop=True)


def group_pills(p: pd.DataFrame) -> list[pd.DataFrame]:
    """IoU ≥ 0.5로 겹친 박스를 한 알약으로 묶는다. 묶음의 첫 행이 가장 신뢰도 높은 이름이다."""
    if p.empty:
        return []
    iou = iou_matrix(to_xyxy(p), to_xyxy(p))
    used, groups = set(), []
    for a in range(len(p)):
        if a in used:
            continue
        g = [b for b in np.where(iou[a] >= 0.5)[0] if b not in used]
        used |= set(g)
        groups.append(p.iloc[g])
    return groups


def draw(img: Image.Image, groups: list[pd.DataFrame], names: dict) -> Image.Image:
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("malgun.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    for i, g in enumerate(groups, start=1):
        r = g.iloc[0]
        check = r.score < CHECK_CONF or len(g) > 1
        color = (230, 160, 0) if check else (0, 170, 80)
        d.rectangle([r.x, r.y, r.x + r.w, r.y + r.h], outline=color, width=6)
        label = f"{i}. {names[r.category_id]} {r.score:.2f}" + (" (확인 필요)" if check else "")
        tw = d.textlength(label, font=font)
        xy = (max(0, min(r.x, out.width - tw - 4)), max(r.y - 36, 0))  # 이름표가 사진 오른쪽 밖으로 나가지 않게
        d.rectangle(d.textbbox(xy, label, font=font), fill=color)
        d.text(xy, label, fill="white", font=font)
    return out


def main():
    st.set_page_config(page_title="경구약제 검출 데모", layout="wide")
    st.title("경구약제 이미지 객체 검출 데모")
    st.caption("YOLO11s + YOLO11m (232장 전체 학습) · WBF 앙상블 · 학습한 약 56종")
    if missing := [str(w.relative_to(REPO_DIR)) for w in MODELS if not w.exists()]:
        st.error(f"최종 모델 가중치가 없습니다: {missing}. README의 재현 명령으로 먼저 학습하세요.")
        return

    P = get_paths(REPO_DIR)
    class_map = pd.read_csv(P.out / "class_map.csv")
    names = dict(zip(class_map["category_id"], class_map["dl_name"]))
    names_en = dict(zip(class_map["category_id"], class_map["dl_name_en"]))

    up = st.file_uploader("알약 사진 (PNG·JPG)", type=["png", "jpg", "jpeg"])
    tests = list_test_images(P.test_img) if P.test_img.exists() else []
    sample = st.selectbox("또는 테스트 이미지 선택", ["-"] + [p.name for p in tests[:200]])
    if up is None and sample == "-":
        st.info("사진을 올리거나 테스트 이미지를 고르세요.")
        return
    if up is not None:
        tmp = Path(tempfile.mkdtemp()) / up.name
        tmp.write_bytes(up.getvalue())
        path = tmp
    else:
        path = P.test_img / sample

    with st.spinner("검출 중..."):
        groups = group_pills(detect(path, class_map))
    col1, col2 = st.columns([3, 2])
    col1.image(draw(Image.open(path), groups, names), width="stretch")
    with col2:
        st.subheader(f"찾은 알약 {len(groups)}개")
        rows = []
        for i, g in enumerate(groups, start=1):
            r = g.iloc[0]
            alt = ", ".join(f"{names[c]} {s:.2f}" for c, s in zip(g.category_id.iloc[1:], g.score.iloc[1:]))
            rows.append({"번호": i, "약 이름": names[r.category_id], "영문명": names_en[r.category_id], "약 코드": int(r.category_id),
                         "신뢰도": round(float(r.score), 3), "다른 후보": alt or "–",
                         "확인": "필요" if r.score < CHECK_CONF or len(g) > 1 else "–"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.warning("이 결과는 학습 데이터(단색 배경·고정 촬영)의 56종 안에서만 의미가 있습니다. 처음 보는 약도 높은 신뢰도로 "
                   "다른 약 이름을 붙일 수 있어(검증에서 처음 보는 약의 24%가 신뢰도 0.9 이상), 복약 판단에는 반드시 약 이름·각인을 직접 확인하세요.")


if __name__ == "__main__":
    main()
