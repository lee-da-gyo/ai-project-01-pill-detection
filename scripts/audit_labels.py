"""라벨 감사 후보 목록 - D(데이터 내부)와 E(모델 보조 사후 감사).

정제 규칙 A~C를 적용한 뒤, 사람 확인 결정(configs/label_review.csv)을 적용하기 전 라벨에서 후보를 찾는다.
- D: 한 이미지 안에서 다른 약 박스 쌍의 IoU ≥ 0.9. 모델 예측을 쓰지 않는다.
- E: 학습 가능한 약(2개 이상 fold에 있음)의 정답 박스에 IoU ≥ 0.5로 겹친 OOF 예측 중 신뢰도가 가장 높은 것이
     다른 약이고 그 신뢰도가 0.9 이상. 모델 예측으로 만든 후보라 라벨과 독립적인 근거가 아니며, 우선순위일 뿐이다.
     처음 보는 약은 모델이 그 이름을 낼 수 없어 항상 불일치하므로 후보에서 뺀다.
후보는 자동으로 고치지 않는다. 원본 사진과 같은 조합의 다른 각도 사진으로 확인한 뒤 configs/label_review.csv에 적는다.

후보는 OOF 예측뿐 아니라 원본 라벨·이미지(A~C 정제에 씀), fold 구성, 코드에 따라서도 달라진다. 그래서 결과 파일에
출처(실험 이름, WBF 가중치·IoU)와 세 묶음의 manifest 해시를 함께 적는다. manifest는 파일마다 (상대 경로, 크기, SHA-256)을
이어 붙여 다시 SHA-256을 낸 값의 앞 12자리라, 파일 경계·이름이 바뀌어도 값이 달라진다.
- 예측: OOF val_predictions.csv 10개
- 입력: 원본 라벨 JSON, 학습 이미지, folds.csv, class_map.csv, configs/label_review.csv
- 코드: src/pilldet/*.py, scripts/audit_labels.py, scripts/analyze_oof.py
세 해시가 같으면 같은 후보가 나온다.

사용법
    python scripts/audit_labels.py                                 # OOF: e0_baseline + e2_yolo11m WBF (2:1, IoU 0.7)
    python scripts/audit_labels.py --exps e0_baseline e2_yolo11m

산출물
    reports/label_audit_candidates.csv   (후보 + 사람 확인 결과)
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, str(REPO_DIR / "scripts"))

from analyze_oof import load_oof  # noqa: E402
from pilldet.boxes import iou_matrix, to_xyxy  # noqa: E402
from pilldet.clean import UNREVIEWED, LabelCleaner, cross_class_overlaps, review_decision, validate_review  # noqa: E402
from pilldet.config import N_FOLDS, get_paths  # noqa: E402
from pilldet.data import load_annotations  # noqa: E402
from pilldet.ensemble import weighted_boxes_fusion  # noqa: E402

E_SCORE = 0.9


TEXT_SUFFIXES = {".py", ".csv", ".json"}


def manifest_digest(groups: dict[Path, list[Path]]) -> str:
    """파일 묶음의 manifest 해시 (앞 12자리). {기준 폴더: 파일 목록}의 파일마다 "상대 경로<TAB>크기<TAB>sha256" 한 줄을
    정렬해 이어 붙인 뒤 SHA-256. 기준 폴더로부터의 상대 경로라 저장소·데이터 폴더 위치가 달라도 값이 같고,
    텍스트 파일은 줄바꿈을 LF로 맞춘 뒤 재므로 체크아웃한 운영체제(CRLF/LF)와도 무관하다."""
    lines = []
    for root, files in groups.items():
        for f in files:
            data = f.read_bytes()
            if f.suffix in TEXT_SUFFIXES:
                data = data.replace(b"\r\n", b"\n")
            lines.append(f"{f.relative_to(root).as_posix()}\t{len(data)}\t{hashlib.sha256(data).hexdigest()}")
    return hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()[:12]


def model_disagreements(pred: pd.DataFrame, gt: pd.DataFrame, learnable: set, min_score: float = E_SCORE) -> pd.DataFrame:
    rows = []
    for f, g in gt.groupby("file_name"):
        p = pred[pred.file_name == f]
        if p.empty:
            continue
        iou = iou_matrix(to_xyxy(g), to_xyxy(p))
        for i, r in enumerate(g.itertuples()):
            q = p[iou[i] >= 0.5]
            if q.empty or r.category_id not in learnable:
                continue
            top = q.loc[q.score.idxmax()]
            same = q[q.category_id == r.category_id].score.max() if (q.category_id == r.category_id).any() else 0.0
            if top.category_id != r.category_id and top.score >= min_score:
                rows.append({"file_name": f, "category_id": int(r.category_id), "other_category_id": int(top.category_id),
                             "score": round(float(top.score), 3), "same_class_score": round(float(same), 3)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exps", nargs="+", default=["e0_baseline", "e2_yolo11m"])
    ap.add_argument("--iou-thr", type=float, default=0.7)
    ap.add_argument("--weights", nargs="+", type=float, default=[2, 1], help="모델별 WBF 가중치 (현재 최종 후보 11s:11m = 2:1)")
    args = ap.parse_args()

    P = get_paths(REPO_DIR)
    raw = load_annotations(P.ann)
    before_review = LabelCleaner(raw, P.train_img).run().clean  # A~C만 적용한 라벨
    folds = pd.read_csv(P.out / "folds.csv")
    names = dict(pd.read_csv(P.out / "class_map.csv")[["category_id", "dl_name"]].values)
    n_folds_of = before_review.merge(folds[["file_name", "fold"]]).groupby("category_id")["fold"].nunique()
    learnable = set(n_folds_of[n_folds_of > 1].index)

    d = cross_class_overlaps(before_review).assign(rule="D_duplicate", evidence_auto=lambda x: "IoU " + x["iou"].astype(str))
    preds = [load_oof(e, P.out / "experiments", folds) for e in args.exps]
    pred = preds[0] if len(preds) == 1 else weighted_boxes_fusion(preds, args.weights, iou_thr=args.iou_thr)
    e = model_disagreements(pred, before_review, learnable)
    e = e.assign(rule="E_model_audit", evidence_auto=lambda x: "OOF 예측 " + x["other_category_id"].map(names) + " "
                 + x["score"].astype(str) + " (라벨 약 " + x["same_class_score"].astype(str) + ")")
    cand = pd.concat([d, e], ignore_index=True)

    review = validate_review(pd.read_csv(REPO_DIR / "configs" / "label_review.csv"))
    cand["name"] = cand.category_id.map(names)
    cand["other_name"] = cand.other_category_id.map(names)
    cand["review"] = [review_decision(review, r.file_name, r.category_id, r.other_category_id) for r in cand.itertuples()]
    pred_files = [P.out / "experiments" / (e if k == 0 else f"{e}_f{k}") / "val_predictions.csv"
                  for e in args.exps for k in range(N_FOLDS)]
    ann_files, img_files = sorted(P.ann.rglob("*.json")), sorted(P.train_img.glob("*.png"))
    meta_files = [P.out / "folds.csv", P.out / "class_map.csv", REPO_DIR / "configs" / "label_review.csv"]
    code_files = [*sorted((REPO_DIR / "src" / "pilldet").glob("*.py")), REPO_DIR / "scripts" / "audit_labels.py",
                  REPO_DIR / "scripts" / "analyze_oof.py"]
    data_root = P.ann.parent  # 원본 데이터 폴더는 저장소 밖에 있을 수 있다
    cand["audit_source"] = (f"OOF {' + '.join(args.exps)} · WBF {':'.join(f'{w:g}' for w in args.weights)}, IoU {args.iou_thr} · "
                            f"manifest 예측 {manifest_digest({REPO_DIR: pred_files})} (파일 {len(pred_files)}개) · "
                            f"입력 {manifest_digest({data_root: [*ann_files, *img_files], REPO_DIR: meta_files})} "
                            f"(라벨 {len(ann_files)} · 이미지 {len(img_files)} · folds · class_map · 검수 결정) · "
                            f"코드 {manifest_digest({REPO_DIR: code_files})} (파일 {len(code_files)}개)")
    cols = ["rule", "file_name", "category_id", "name", "other_category_id", "other_name", "evidence_auto", "review", "audit_source"]
    cand[cols].to_csv(P.out / "label_audit_candidates.csv", index=False, encoding="utf-8-sig")
    print(f"출처: {cand['audit_source'].iloc[0] if len(cand) else '-'} · E 기준 신뢰도 ≥ {E_SCORE}")
    print(cand[["rule", "file_name", "name", "other_name", "evidence_auto", "review"]].to_string(index=False))
    print(f"\n후보 D {len(d)}개 · E {len(e)}개 → reports/label_audit_candidates.csv")
    unreviewed = int((cand.review == UNREVIEWED).sum())
    if unreviewed:
        # 검수 기록이 없는 후보를 기각(라벨이 맞음)으로 단정하지 않는다. 사람이 확인한 뒤 결정을 적어야 통과한다.
        print(f"[경고] 미검토 후보 {unreviewed}개: 원본 사진과 다른 각도 사진으로 확인한 뒤 configs/label_review.csv에 결정"
              "(relocate · swap · keep_overlap · keep_label)을 적고 다시 실행한다.")
        sys.exit(1)


if __name__ == "__main__":
    main()
