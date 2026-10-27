"""시각화 공통 함수."""

import platform
import subprocess
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def setup_korean_font() -> bool:
    """한글 폰트를 설정한다. 실패하면 False → 영문 약품명을 쓴다."""
    ok = False
    if platform.system() == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
        ok = True
    else:
        try:
            import koreanize_matplotlib  # noqa: F401
            ok = True
        except ImportError:
            try:  # Kaggle: Internet On 이면 설치된다
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "koreanize-matplotlib"], check=True)
                import koreanize_matplotlib  # noqa: F401
                ok = True
            except Exception as e:
                print("한글 폰트 설치 실패 → 영문 약품명 사용:", e)
    plt.rcParams["axes.unicode_minus"] = False
    return ok


def save_fig(fig, fig_dir: Path, name: str):
    """보고서에 쓸 수 있게 PNG로 저장한다."""
    fig.savefig(Path(fig_dir) / f"{name}.png", dpi=150, bbox_inches="tight")


def class_colors(category_ids) -> dict:
    cmap = plt.get_cmap("tab20")
    return {c: cmap(i % 20) for i, c in enumerate(sorted(category_ids))}


def draw_boxes(ax, img_path: Path, boxes: pd.DataFrame, name_col: str = "dl_name", colors: dict | None = None,
               highlight: set | None = None, title: str | None = None, max_label: int = 14):
    """이미지에 박스를 그린다.

    - 이미지 밖으로 벗어난 박스는 잘라서 빨간 점선으로, 완전히 밖이면 경고 문구만 표시한다.
    - highlight에 (file_name, category_id)가 있거나 source가 original이 아니면 노란 굵은 박스로 강조한다.
    """
    img = Image.open(img_path)
    W, H = img.size
    ax.imshow(img)
    highlight = highlight or set()
    for _, r in boxes.iterrows():
        x1, y1 = max(r.x, 0), max(r.y, 0)
        x2, y2 = min(r.x + r.w, W), min(r.y + r.h, H)
        name = str(r[name_col])[:max_label]
        if x2 <= x1 or y2 <= y1:
            ax.text(10, 40, f"범위 밖 박스: {name} x={r.x:.0f}", fontsize=7, color="white",
                    bbox=dict(fc="red", alpha=0.9, pad=1))
            continue
        invalid = (r.x < 0) or (r.y < 0) or (r.x + r.w > W) or (r.y + r.h > H)
        changed = r.get("source", "original") != "original" or (r.get("file_name"), r.category_id) in highlight
        if invalid:
            ec, lw, ls = "red", 2.5, "--"
        elif changed:
            ec, lw, ls = "yellow", 3, "-"
        else:
            ec, lw, ls = (colors or {}).get(r.category_id, "lime"), 2, "-"
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=ec, lw=lw, ls=ls))
        ax.text(x1, max(y1 - 6, 0), name, fontsize=6, color="black", bbox=dict(fc=ec, alpha=0.8, pad=1))
    ax.set_title(title or Path(img_path).name[:40], fontsize=8)
    ax.axis("off")
