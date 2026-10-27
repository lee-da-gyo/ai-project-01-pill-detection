"""라벨 감사(audit)와 규칙 기반 수정.

라벨 박스를 배경 차분 분할 결과(segment.py)와 대조해 오류를 찾고, 유형별 규칙으로 고친다.
모든 수정은 로그로 남겨 재현할 수 있게 한다.

| 유형 | 탐지 | 수정 |
|---|---|---|
| A. 범위 밖 박스 | 좌표가 이미지 밖 | 숫자 하나를 뺀 후보 중 라벨 없는 알약과 IoU가 가장 큰 값 (자릿수 입력 오류 가정) |
| B. 엉뚱한 위치 박스 | 박스 안 전경 비율 < fg_thresh, 누락 라벨이 없는 이미지, 원래 박스와 겹치지 않고 크기가 비슷한 라벨 없는 알약이 정확히 1개 | 그 알약 위치로 이동 |
| C. 누락 박스 | 파일명의 약 코드 중 라벨 없는 코드 1개 + 라벨 없는 알약 1개 | 해당 코드로 박스 추가 |

대응이 하나로 정해지지 않으면 자동 수정하지 않고 notes에 남긴다.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .boxes import containment, iou_matrix, out_of_bounds, to_xyxy
from .data import expected_codes, load_rgb
from .segment import Segmentation, segment_pills


@dataclass
class CleanResult:
    clean: pd.DataFrame  # 정제된 박스 테이블
    fixes: pd.DataFrame  # 수정 로그
    audit: pd.DataFrame  # 원본 라벨 감사 결과 (best_iou, fg_ratio, out_of_bounds)
    calib: np.ndarray  # 분할 박스 → 라벨러 박스 테두리 보정값 [dx1, dy1, dx2, dy2]
    notes: list = field(default_factory=list)  # 자동 수정하지 않은 사례


class LabelCleaner:
    def __init__(self, raw: pd.DataFrame, img_dir: Path, fg_thresh: float = 0.15,
                 match_iou: float = 0.1, contain_thresh: float = 0.5, calib_iou: float = 0.75,
                 size_ratio: tuple = (0.7, 1.4)):
        self.raw = raw.copy()
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

    def _log(self, kind, f, cid, before, after, note):
        self._fixes.append({
            "type": kind, "file_name": f, "category_id": int(cid), "name": self._names.loc[cid, "dl_name"],
            "before": None if before is None else [round(float(v), 1) for v in before],
            "after": [round(float(v), 1) for v in after], "note": note,
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

    # ---------- 전체 실행 ----------
    def run(self) -> CleanResult:
        if not self.segs:
            self.segment_all()
        aud = self.audit(self.raw)
        self.calibrate(aud)
        df = self.fix_out_of_bounds(self.raw, aud)
        df = self.fix_misplaced(df)
        df = self.fix_missing(df)
        return CleanResult(df, pd.DataFrame(self._fixes), aud, self.calib, self._notes)


def check_integrity(clean: pd.DataFrame, train_files: list[str]) -> dict:
    """정제 결과의 무결성을 확인한다. 하나라도 어긋나면 AssertionError."""
    xyxy = to_xyxy(clean)
    per_img = clean.groupby("file_name")["category_id"].apply(sorted)
    assert set(per_img.index) == set(train_files), "라벨 없는 이미지가 있다"
    assert not out_of_bounds(xyxy, clean["img_w"].values, clean["img_h"].values).any(), "범위 밖 박스가 남아 있다"
    assert (clean["w"] > 0).all() and (clean["h"] > 0).all(), "크기 0 박스"
    assert not clean.duplicated(["file_name", "category_id"]).any(), "같은 이미지에 같은 클래스 중복"
    mismatch = [f for f, codes in per_img.items() if list(codes) != expected_codes(f)]
    assert not mismatch, f"파일명과 라벨 불일치: {mismatch}"
    return {"images": int(clean["file_name"].nunique()), "boxes": len(clean),
            "sources": clean["source"].value_counts().to_dict(),
            "boxes_per_image": clean.groupby("file_name").size().value_counts().to_dict()}
