"""라벨 감사(audit)와 규칙 기반 수정.

라벨 박스를 배경 차분 분할 결과(segment.py)와 대조해 오류를 찾고, 유형별 규칙으로 고친다.
모든 수정은 로그로 남겨 재현할 수 있게 한다.

| 유형 | 탐지 | 수정 |
|---|---|---|
| A. 범위 밖 박스 | 좌표가 이미지 밖 | 숫자 하나를 뺀 후보 중 라벨 없는 알약과 IoU가 가장 큰 값 (자릿수 입력 오류 가정) |
| B. 엉뚱한 위치 박스 | 박스 안 전경 비율 < fg_thresh, 누락 라벨이 없는 이미지, 원래 박스와 겹치지 않고 크기가 비슷한 라벨 없는 알약이 정확히 1개 | 그 알약 위치로 이동 |
| C. 누락 박스 | 파일명의 약 코드 중 라벨 없는 코드 1개 + 라벨 없는 알약 1개 | 해당 코드로 박스 추가 |
| D. 다른 약끼리 같은 좌표 | 한 이미지에서 다른 약 박스 쌍의 IoU ≥ 0.9 (후보만 자동 탐지) | 사람이 확인한 라벨을 라벨 없는 알약으로 이동 (실제 겹침이면 keep_overlap으로 기록) |
| E. 라벨 교환 | OOF 예측과의 고신뢰 불일치 (모델 보조 사후 감사, scripts/audit_labels.py) | 사람이 확인한 두 박스의 약 코드를 맞바꿈 (라벨이 맞으면 keep_label로 기록) |

A~D는 모델 예측을 쓰지 않는 데이터 내부 감사다. E는 모델 예측으로 후보를 만든 비독립 감사라서 성능 추정의 근거로 쓰지 않는다.
D·E는 알약이 실제로 겹칠 수도 있고 모델이 틀릴 수도 있으므로 자동으로 고치지 않는다.
원본 사진·다른 각도 사진으로 확인한 결정만 configs/label_review.csv(status=confirmed)에 적고, 그 결정만 적용한다.
결정에는 검수할 때 본 두 박스의 좌표(box, other_box)를 함께 적는다. 적용 직전 라벨이 그 좌표와 다르면(이미 적용됐거나 원본이 바뀜)
적용하지 않고 notes에 남긴다 → 같은 결정을 몇 번 적용해도 결과가 같고, 원본이 바뀌면 엉뚱한 박스를 고치지 않는다.
대응이 하나로 정해지지 않으면 자동 수정하지 않고 notes에 남긴다.
"""

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .boxes import containment, iou_matrix, out_of_bounds, to_xyxy
from .data import expected_codes, load_rgb
from .segment import Segmentation, segment_pills


DUP_IOU = 0.9  # D 후보 기준: 서로 다른 약 박스가 이만큼 겹치면 같은 알약을 두 번 라벨한 것으로 의심한다
BOX_TOL = 1.0  # 검수 때 본 좌표와 현재 라벨 좌표의 허용 오차 (px)
REVIEW_COLUMNS = ["review_id", "rule", "file_name", "category_id", "other_category_id", "action", "box", "other_box",
                  "evidence", "reviewer", "reviewed_at", "status"]
REVIEW_ACTIONS = {"relocate", "swap", "keep_overlap", "keep_label"}
# keep_overlap: 실제로 겹친 알약이라 고치지 않음 (D 후보 기각) / keep_label: 육안 확인 결과 라벨이 맞아 고치지 않음 (E 후보 기각)
UNREVIEWED = "미검토 (정확한 쌍 결정 없음)"
REVIEW_STATUS = {"confirmed", "pending"}
RULE_ACTIONS = {"D_duplicate": {"relocate", "keep_overlap"},  # 감사 규칙별로 허용되는 결정
                "E_swapped": {"swap"}, "E_model_audit": {"keep_label"}}


def pair_key(file_name: str, cid: int, other: int) -> tuple:
    """검수 결정과 감사 후보를 잇는 키: (파일, 작은 코드, 큰 코드). 두 약의 순서와 무관하게 정확히 같은 쌍만 맞는다."""
    return (file_name, *sorted([int(cid), int(other)]))


def review_by_pair(review: pd.DataFrame) -> dict:
    """pair_key → 결정 행. 한 약이 여러 결정에 나와도 다른 쌍의 결정이 섞이지 않는다."""
    return {pair_key(r.file_name, r.category_id, r.other_category_id): r for r in review.itertuples()}


def review_decision(review: pd.DataFrame, file_name: str, cid: int, other: int) -> str:
    """감사 후보 (파일, 라벨 약 cid, 상대 약 other)에 붙일 검수 결과.

    두 약이 정확히 같은 쌍의 결정만 그 후보의 결정으로 본다. 쌍 결정이 없어도 라벨 약 cid의 박스를 옮기거나 바꾼
    결정이 있으면 "관련 결정"으로 따로 적는다 (다른 쌍의 결정을 이 후보의 판정으로 붙이지 않는다).
    둘 다 없으면 UNREVIEWED다. 기각(라벨이 맞음)은 keep_label · keep_overlap 결정이 있을 때만 표시된다.
    """
    row = review_by_pair(review).get(pair_key(file_name, cid, other))
    if row is not None:
        return f"{row.status}: {row.review_id} {row.rule} {row.action}"
    moved = review[(review["file_name"] == file_name)
                   & ((review["category_id"] == cid) & review["action"].isin(["relocate", "swap"])
                      | (review["other_category_id"] == cid) & (review["action"] == "swap"))]
    if len(moved):
        return "쌍 결정 없음 - 같은 박스를 고친 관련 결정: " + ", ".join(f"{r.review_id} {r.action}" for r in moved.itertuples())
    return UNREVIEWED


def validate_review(review: pd.DataFrame) -> pd.DataFrame:
    """검수 결정 파일의 형식을 확인한다. 어긋나면 ValueError – 잘못된 결정이 조용히 적용되거나 무시되지 않게 한다."""
    missing = [c for c in REVIEW_COLUMNS if c not in review.columns]
    if missing:
        raise ValueError(f"검수 결정 파일에 열이 없다: {missing}")
    if review["review_id"].duplicated().any():
        raise ValueError(f"review_id 중복: {review.loc[review['review_id'].duplicated(), 'review_id'].tolist()}")
    pair = review.apply(lambda r: pair_key(r.file_name, r.category_id, r.other_category_id), axis=1)
    if pair.duplicated().any():
        raise ValueError(f"같은 박스 쌍에 결정이 두 개: {review.loc[pair.duplicated(), 'review_id'].tolist()}")
    if bad := sorted(set(review["action"]) - REVIEW_ACTIONS):
        raise ValueError(f"알 수 없는 action: {bad}")
    if bad := sorted(set(review["status"]) - REVIEW_STATUS):
        raise ValueError(f"알 수 없는 status: {bad}")
    for r in review.itertuples():
        if r.rule not in RULE_ACTIONS or r.action not in RULE_ACTIONS[r.rule]:
            raise ValueError(f"{r.review_id}: rule {r.rule}에 action {r.action}은 허용되지 않는다 ({RULE_ACTIONS})")
        if int(r.category_id) == int(r.other_category_id):
            raise ValueError(f"{r.review_id}: 두 약 코드가 같다 ({r.category_id})")
        for col in ("evidence", "reviewer", "reviewed_at"):
            if pd.isna(getattr(r, col)) or not str(getattr(r, col)).strip():
                raise ValueError(f"{r.review_id}: {col}이 비어 있다")
        try:
            date.fromisoformat(str(r.reviewed_at))
        except ValueError:
            raise ValueError(f"{r.review_id}: reviewed_at은 YYYY-MM-DD 형식이어야 한다 ({r.reviewed_at})") from None
        for col in ("box", "other_box"):
            v = getattr(r, col)
            try:
                b = np.asarray(json.loads(v), float)
            except (ValueError, TypeError):
                raise ValueError(f"{r.review_id}: {col}은 [x, y, w, h] 형식이어야 한다 ({v})") from None
            if b.shape != (4,) or not np.isfinite(b).all() or b[2] <= 0 or b[3] <= 0:
                raise ValueError(f"{r.review_id}: {col}은 [x, y, w, h] 형식이어야 한다 (유한한 숫자 4개, w·h > 0): {v}")
    return review


@dataclass
class CleanResult:
    clean: pd.DataFrame  # 정제된 박스 테이블
    fixes: pd.DataFrame  # 수정 로그
    audit: pd.DataFrame  # 원본 라벨 감사 결과 (best_iou, fg_ratio, out_of_bounds)
    calib: np.ndarray  # 분할 박스 → 라벨러 박스 테두리 보정값 [dx1, dy1, dx2, dy2]
    notes: list = field(default_factory=list)  # 자동 수정하지 않은 사례


class LabelCleaner:
    def __init__(self, raw: pd.DataFrame, img_dir: Path, review: pd.DataFrame | None = None, fg_thresh: float = 0.15,
                 match_iou: float = 0.1, contain_thresh: float = 0.5, calib_iou: float = 0.75,
                 size_ratio: tuple = (0.7, 1.4)):
        self.raw = raw.copy()
        self.review = validate_review(review) if review is not None else pd.DataFrame(columns=REVIEW_COLUMNS)
        self.img_dir = Path(img_dir)
        self.fg_thresh = fg_thresh
        self.match_iou = match_iou
        self.contain_thresh = contain_thresh
        self.calib_iou = calib_iou
        self.size_ratio = size_ratio
        self.segs: dict[str, Segmentation] = {}
        self.calib = np.zeros(4)
        self._fixes: list[dict] = []
        self._notes: list[str] = []
        self._names = raw.drop_duplicates("category_id").set_index("category_id")[["dl_name", "dl_name_en"]]

    # ---------- 분할과 감사 ----------
    def segment_all(self) -> dict[str, Segmentation]:
        for f in sorted(self.raw["file_name"].unique()):
            self.segs[f] = segment_pills(load_rgb(self.img_dir / f))
        return self.segs

    def audit(self, df: pd.DataFrame) -> pd.DataFrame:
        """박스마다 분할 후보와의 최대 IoU, 박스 안 전경 비율, 범위 밖 여부를 계산한다."""
        out = df.copy()
        xyxy = to_xyxy(out)
        files = out["file_name"].tolist()
        out["best_iou"] = [iou_matrix(xyxy[i], self.segs[f].boxes).max(initial=0) for i, f in enumerate(files)]
        out["fg_ratio"] = [self.segs[f].fg_ratio(xyxy[i]) for i, f in enumerate(files)]
        out["out_of_bounds"] = out_of_bounds(xyxy, out["img_w"].values, out["img_h"].values)
        return out

    def unmatched_segments(self, f: str, df: pd.DataFrame) -> np.ndarray:
        """어떤 라벨과도 겹치지 않고, 라벨 안에 포함되지도 않는 알약 후보 (= 라벨 없는 알약)."""
        seg = self.segs[f].boxes
        boxes = to_xyxy(df[df["file_name"] == f])
        if len(seg) == 0:
            return seg
        keep = (iou_matrix(seg, boxes).max(axis=1, initial=0) < self.match_iou) & \
               (containment(seg, boxes).max(axis=1, initial=0) < self.contain_thresh)
        return seg[keep]

    def calibrate(self, aud: pd.DataFrame) -> np.ndarray:
        """잘 맞는 쌍(IoU > calib_iou)에서 라벨 박스와 분할 박스의 테두리 차이 중앙값.

        라벨과 분할 영역은 헝가리안 매칭으로 1:1 대응시킨다 (두 라벨이 같은 분할 영역에 짝지어지는 것을 막는다).
        """
        diffs = []
        for f, g in aud[~aud["out_of_bounds"]].groupby("file_name"):
            seg = self.segs[f].boxes
            if len(seg) == 0:
                continue
            lab = to_xyxy(g)
            m = iou_matrix(lab, seg)
            for i, j in zip(*linear_sum_assignment(-m)):
                if m[i, j] > self.calib_iou:
                    diffs.append(lab[i] - seg[j])
        self.calib = np.median(np.array(diffs), axis=0)
        return self.calib

    def _calibrated(self, seg_box: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
        return np.clip(seg_box + self.calib, 0, [img_w, img_h, img_w, img_h])

    def _log(self, kind, f, cid, before, after, note, review_id=None):
        self._fixes.append({
            "type": kind, "file_name": f, "category_id": int(cid), "name": self._names.loc[cid, "dl_name"],
            "before": None if before is None else [round(float(v), 1) for v in before],
            "after": [round(float(v), 1) for v in after], "note": note, "review_id": review_id,
        })

    # ---------- 규칙 ----------
    def fix_out_of_bounds(self, df: pd.DataFrame, aud: pd.DataFrame) -> pd.DataFrame:
        """A. 자릿수 하나가 잘못 들어간 좌표를 복원한다."""
        df = df.copy()
        for idx in aud.index[aud["out_of_bounds"]]:
            r = df.loc[idx]
            free = self.unmatched_segments(r.file_name, df[df.index != idx])
            best = None
            for col, limit, size in (("x", r.img_w, r.w), ("y", r.img_h, r.h)):
                if 0 <= r[col] and r[col] + size <= limit:
                    continue
                digits = str(int(r[col]))
                for k in range(len(digits)):
                    cand = int(digits[:k] + digits[k + 1:]) if len(digits) > 1 else 0
                    if cand + size > limit:
                        continue
                    box = [cand, r.y, cand + r.w, r.y + r.h] if col == "x" else [r.x, cand, r.x + r.w, cand + r.h]
                    score = iou_matrix(box, free).max(initial=0)
                    if best is None or score > best[0]:
                        best = (score, col, cand)
            if best and best[0] > 0.5:
                before = [r.x, r.y, r.w, r.h]
                df.loc[idx, best[1]] = best[2]
                self._log("A_out_of_bounds", r.file_name, r.category_id, before, df.loc[idx, ["x", "y", "w", "h"]],
                          f"{best[1]}={int(r[best[1]])} → {best[2]} (라벨 없는 알약과 IoU {best[0]:.2f})")
            else:
                self._notes.append(f"A 자동 복원 실패: {r.file_name} / {r.dl_name}")
        return df

    def fix_misplaced(self, df: pd.DataFrame) -> pd.DataFrame:
        """B. 빈 배경에 그려진 박스를 같은 크기의 라벨 없는 알약 위치로 옮긴다."""
        df = df.copy()
        aud = self.audit(df)
        lo, hi = self.size_ratio
        for idx in aud.index[(aud["fg_ratio"] < self.fg_thresh) & ~aud["out_of_bounds"]]:
            r = df.loc[idx]
            labeled = set(df.loc[df["file_name"] == r.file_name, "category_id"])
            if set(expected_codes(r.file_name)) - labeled:
                continue  # 누락 라벨이 있는 이미지는 C 규칙이 처리한다
            free = self.unmatched_segments(r.file_name, df[df.index != idx])
            if len(free):
                # 같은 알약을 엉뚱한 곳에 그렸다면 원래 박스와 겹치지 않고 크기가 비슷해야 한다.
                # 반투명 캡슐처럼 분할이 일부만 잡힌 경우(작은 조각, 원래 박스와 겹침)를 걸러낸다.
                own = [r.x, r.y, r.x + r.w, r.y + r.h]
                rw = (free[:, 2] - free[:, 0] + self.calib[2] - self.calib[0]) / r.w
                rh = (free[:, 3] - free[:, 1] + self.calib[3] - self.calib[1]) / r.h
                ok = (iou_matrix(free, own)[:, 0] == 0) & (rw > lo) & (rw < hi) & (rh > lo) & (rh < hi)
                free = free[ok]
            if len(free) == 1:
                nb = self._calibrated(free[0], r.img_w, r.img_h)
                before = [r.x, r.y, r.w, r.h]
                df.loc[idx, ["x", "y", "w", "h"]] = [nb[0], nb[1], nb[2] - nb[0], nb[3] - nb[1]]
                df.loc[idx, "source"] = "relocated"
                self._log("B_misplaced", r.file_name, r.category_id, before, df.loc[idx, ["x", "y", "w", "h"]],
                          f"박스 안 전경 비율 {aud.loc[idx, 'fg_ratio']:.2f} → 라벨 없는 알약 위치로 이동")
            else:
                self._notes.append(f"B 보류 (분할 실패로 판단, 후보 {len(free)}개): {r.file_name} / {r.dl_name}")
        return df

    def fix_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """C. 파일명에 있는데 라벨이 없는 약 코드를 라벨 없는 알약에 추가한다."""
        new_rows = []
        for f, g in df.groupby("file_name"):
            miss = sorted(set(expected_codes(f)) - set(g["category_id"]))
            if not miss:
                continue
            free = self.unmatched_segments(f, g)
            if len(miss) == 1 and len(free) == 1:
                r0, cid = g.iloc[0], miss[0]
                nb = self._calibrated(free[0], r0.img_w, r0.img_h)
                xywh = [nb[0], nb[1], nb[2] - nb[0], nb[3] - nb[1]]
                new_rows.append({**r0.to_dict(), "category_id": cid,
                                 "dl_name": self._names.loc[cid, "dl_name"], "dl_name_en": self._names.loc[cid, "dl_name_en"],
                                 "x": xywh[0], "y": xywh[1], "w": xywh[2], "h": xywh[3], "source": "added"})
                self._log("C_missing", f, cid, None, xywh, "파일명에 있으나 라벨 없음 → 라벨 없는 알약에 추가")
            else:
                self._notes.append(f"C 자동 보완 불가 (누락 {miss}, 후보 {len(free)}개): {f}")
        return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    def apply_review(self, df: pd.DataFrame) -> pd.DataFrame:
        """D·E. 사람이 확인한 결정(configs/label_review.csv, status=confirmed)만 적용한다.

        두 박스의 현재 좌표가 검수 때 본 좌표(box, other_box)와 BOX_TOL 안에서 같을 때만 적용한다.
        이미 적용된 결과에 다시 적용하면 좌표가 달라 건너뛰므로(notes에 기록) 몇 번 실행해도 결과가 같다.
        """
        df = df.copy()
        for r in self.review[self.review["status"] == "confirmed"].itertuples():
            cid, other = int(r.category_id), int(r.other_category_id)
            a = df.index[(df["file_name"] == r.file_name) & (df["category_id"] == cid)]
            b = df.index[(df["file_name"] == r.file_name) & (df["category_id"] == other)]
            if len(a) != 1 or len(b) != 1:
                self._notes.append(f"{r.review_id} 적용 불가 (박스 {len(a)}·{len(b)}개): {r.file_name} / {cid}, {other}")
                continue
            a, b = a[0], b[0]
            ra = df.loc[a]
            xa, xb = df.loc[a, ["x", "y", "w", "h"]].to_numpy(float), df.loc[b, ["x", "y", "w", "h"]].to_numpy(float)
            if not (np.allclose(xa, json.loads(r.box), atol=BOX_TOL, rtol=0)
                    and np.allclose(xb, json.loads(r.other_box), atol=BOX_TOL, rtol=0)):
                self._notes.append(f"{r.review_id} 적용 안 함 (검수 때 본 좌표와 다름: 이미 적용됐거나 원본이 바뀜): {r.file_name}")
                continue
            if r.action in ("keep_overlap", "keep_label"):
                continue  # 고치지 않는 결정. keep_overlap은 check_integrity가 검수 때 본 좌표 그대로인 겹침만 허용한다
            if r.action == "relocate":
                free = self.unmatched_segments(r.file_name, df[df.index != a])
                if len(free) != 1:
                    self._notes.append(f"{r.review_id} 보류 (라벨 없는 알약 {len(free)}개): {r.file_name} / {cid}")
                    continue
                nb = self._calibrated(free[0], ra.img_w, ra.img_h)
                df.loc[a, ["x", "y", "w", "h"]] = [nb[0], nb[1], nb[2] - nb[0], nb[3] - nb[1]]
                df.loc[a, "source"] = "relocated"
                self._log(r.rule, r.file_name, cid, xa, df.loc[a, ["x", "y", "w", "h"]],
                          f"{self._names.loc[other, 'dl_name']}와 같은 좌표 → 라벨 없는 알약으로 이동 (육안 확인: {r.evidence})", r.review_id)
            else:  # swap
                df.loc[a, ["x", "y", "w", "h"]], df.loc[b, ["x", "y", "w", "h"]] = xb, xa
                df.loc[[a, b], "source"] = "swapped"
                self._log(r.rule, r.file_name, cid, xa, xb, f"{self._names.loc[other, 'dl_name']}와 라벨 교환 (육안 확인: {r.evidence})", r.review_id)
                self._log(r.rule, r.file_name, other, xb, xa, f"{self._names.loc[cid, 'dl_name']}와 라벨 교환 (육안 확인: {r.evidence})", r.review_id)
        return df

    # ---------- 전체 실행 ----------
    def run(self) -> CleanResult:
        if not self.segs:
            self.segment_all()
        aud = self.audit(self.raw)
        self.calibrate(aud)
        df = self.fix_out_of_bounds(self.raw, aud)
        df = self.fix_misplaced(df)
        df = self.fix_missing(df)
        df = self.apply_review(df)
        return CleanResult(df, pd.DataFrame(self._fixes), aud, self.calib, self._notes)


def cross_class_overlaps(df: pd.DataFrame, iou_thr: float = None) -> pd.DataFrame:
    """D 후보: 한 이미지 안에서 서로 다른 약 박스 쌍의 IoU ≥ iou_thr. 같은 알약을 두 약으로 라벨한 흔적이다."""
    iou_thr = DUP_IOU if iou_thr is None else iou_thr
    rows = []
    for f, g in df.groupby("file_name"):
        g = g.reset_index(drop=True)
        m = iou_matrix(to_xyxy(g), to_xyxy(g))
        for i, j in zip(*np.where(np.triu(m, 1) >= iou_thr)):
            rows.append({"file_name": f, "category_id": int(g.loc[i, "category_id"]),
                         "other_category_id": int(g.loc[j, "category_id"]), "iou": round(float(m[i, j]), 3)})
    return pd.DataFrame(rows, columns=["file_name", "category_id", "other_category_id", "iou"])


def kept_overlaps(review: pd.DataFrame | None) -> dict:
    """사람이 "실제로 겹친 알약"으로 확인한 D 후보: (file_name, 작은 코드, 큰 코드) → {코드: 검수 때 본 [x, y, w, h]}."""
    if review is None or review.empty:
        return {}
    k = review[(review["action"] == "keep_overlap") & (review["status"] == "confirmed")]
    return {pair_key(r.file_name, r.category_id, r.other_category_id):
            {int(r.category_id): json.loads(r.box), int(r.other_category_id): json.loads(r.other_box)} for r in k.itertuples()}


def _is_kept(clean: pd.DataFrame, f: str, cid: int, other: int, keep: dict) -> bool:
    """이 겹침이 keep_overlap 결정이 있고, 두 박스가 지금도 검수 때 본 좌표와 BOX_TOL 안에서 같은가."""
    boxes = keep.get(pair_key(f, cid, other))
    if boxes is None:
        return False
    for c in (cid, other):
        cur = clean.loc[(clean["file_name"] == f) & (clean["category_id"] == c), ["x", "y", "w", "h"]].to_numpy(float)
        if len(cur) != 1 or not np.allclose(cur[0], boxes[c], atol=BOX_TOL, rtol=0):
            return False
    return True


def check_integrity(clean: pd.DataFrame, train_files: list[str], review: pd.DataFrame | None = None) -> dict:
    """정제 결과의 무결성을 확인한다. 하나라도 어긋나면 AssertionError.

    다른 약끼리 같은 좌표(D 후보)는 사람이 keep_overlap으로 확인했고 두 박스가 검수 때 본 좌표 그대로인 겹침만 허용한다
    → "미해결 D 후보 0"을 검사한다. 같은 약 쌍이라도 좌표가 바뀐 겹침은 새 후보로 본다.
    """
    xyxy = to_xyxy(clean)
    per_img = clean.groupby("file_name")["category_id"].apply(sorted)
    assert set(per_img.index) == set(train_files), "라벨 없는 이미지가 있다"
    assert not out_of_bounds(xyxy, clean["img_w"].values, clean["img_h"].values).any(), "범위 밖 박스가 남아 있다"
    assert (clean["w"] > 0).all() and (clean["h"] > 0).all(), "크기 0 박스"
    assert not clean.duplicated(["file_name", "category_id"]).any(), "같은 이미지에 같은 클래스 중복"
    keep = kept_overlaps(review)
    unresolved = [r for r in cross_class_overlaps(clean).itertuples()
                  if not _is_kept(clean, r.file_name, r.category_id, r.other_category_id, keep)]
    assert not unresolved, f"다른 약 박스가 같은 좌표에 겹쳐 있다 (D 후보 미해결): {len(unresolved)}쌍"
    mismatch = [f for f, codes in per_img.items() if list(codes) != expected_codes(f)]
    assert not mismatch, f"파일명과 라벨 불일치: {mismatch}"
    return {"images": int(clean["file_name"].nunique()), "boxes": len(clean),
            "sources": clean["source"].value_counts().to_dict(),
            "boxes_per_image": clean.groupby("file_name").size().value_counts().to_dict()}
