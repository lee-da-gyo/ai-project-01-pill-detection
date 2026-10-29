"""보고서 HTML(docs/report/report.html)을 PDF로 만든다.

Edge(또는 Chrome)의 헤드리스 인쇄 엔진을 쓰므로 한글 폰트가 그대로 임베드되고, 그림은 저장소의 reports/figures/를 그대로 참조한다.

사용법
    python scripts/build_report.py                 # → docs/report/report.pdf
    python scripts/build_report.py --png 1 5 9     # 해당 쪽을 PNG로도 저장해 레이아웃을 확인한다 (pymupdf 필요)

검사: 쪽수, 폰트 임베드, 깨진 그림(로드 실패), 남은 [기입] 표시 개수를 출력한다.
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
SRC = REPO_DIR / "docs" / "report" / "report.html"
OUT = REPO_DIR / "docs" / "report" / "report.pdf"
BUILD = REPO_DIR / "docs" / "report" / "_build"  # 인쇄용으로 줄인 그림과 HTML (git 제외)
MAX_WIDTH, JPEG_QUALITY = 1600, 85  # A4 본문 폭(약 180mm)에서 220dpi 수준. 원본 PNG를 그대로 넣으면 PDF가 25MB를 넘는다
BROWSERS = [Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
            Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
            Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")]


def find_browser() -> Path:
    for b in [*BROWSERS, *(Path(p) for p in filter(None, [shutil.which("msedge"), shutil.which("google-chrome")]))]:
        if b.exists():
            return b
    raise SystemExit("Edge 또는 Chrome을 찾지 못했다")


def check_images(html: str) -> list[str]:
    """HTML이 참조하는 그림 파일이 모두 있는지 확인한다 (인쇄 엔진은 없는 그림을 조용히 빈칸으로 둔다)."""
    return [src for src in re.findall(r'<img[^>]+src="([^"]+)"', html) if not (SRC.parent / src).resolve().exists()]


def shrink_images(html: str) -> str:
    """그림을 JPEG(최대 폭 1600px)로 줄여 _build/에 두고, 그 사본을 가리키는 HTML을 반환한다. 원본 그림은 건드리지 않는다."""
    from PIL import Image

    BUILD.mkdir(exist_ok=True)

    def repl(m):
        src = (SRC.parent / m.group(1)).resolve()
        dst = BUILD / (src.stem + ".jpg")
        im = Image.open(src).convert("RGB")
        if im.width > MAX_WIDTH:
            im = im.resize((MAX_WIDTH, round(im.height * MAX_WIDTH / im.width)), Image.LANCZOS)
        im.save(dst, quality=JPEG_QUALITY, optimize=True)
        return f'src="{dst.name}"'

    html = re.sub(r'src="([^"]+\.(?:png|jpg))"', repl, html)
    return html.replace('href="report.css"', 'href="../report.css"')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--png", nargs="*", type=int, default=None, help="PNG로 저장할 쪽 번호 (1부터)")
    args = ap.parse_args()

    html = SRC.read_text(encoding="utf-8")
    if missing := check_images(html):
        raise SystemExit(f"없는 그림: {missing}")
    page = BUILD / "report.html"
    page.parent.mkdir(exist_ok=True)
    page.write_text(shrink_images(html), encoding="utf-8")
    OUT.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as prof:  # 사용 중인 브라우저 프로필과 충돌하지 않게 임시 프로필을 쓴다
        subprocess.run([str(find_browser()), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={prof}", f"--print-to-pdf={OUT}", page.as_uri()], check=True, timeout=180)
        # Windows의 Edge 실행 파일은 실제 인쇄 프로세스가 끝나기 전에 반환될 수 있어, 파일 크기가 멈출 때까지 기다린다
        size, deadline = -1, time.time() + 120
        while time.time() < deadline:
            time.sleep(1)
            if OUT.exists() and OUT.stat().st_size == size > 0:
                break
            size = OUT.stat().st_size if OUT.exists() else -1
        else:
            raise SystemExit("PDF가 만들어지지 않았다")

    import pymupdf

    doc = pymupdf.open(OUT)
    fonts = sorted({f[3].split("+")[-1] for page in doc for f in page.get_fonts() if f[3]})
    todo = sum(page.get_text().count("[기입]") + page.get_text().count("기입한다") for page in doc)
    print(f"{OUT.relative_to(REPO_DIR)}: {doc.page_count}쪽, {OUT.stat().st_size / 1e6:.1f}MB, 폰트 {fonts}, 남은 기입 표시 {todo}개")
    for n in [n for n in args.png or [] if 1 <= n <= doc.page_count]:
        out = OUT.with_name(f"page_{n:02d}.png")
        doc[n - 1].get_pixmap(dpi=80).save(out)
        print(f"  → {out.relative_to(REPO_DIR)}")


if __name__ == "__main__":
    sys.exit(main())
