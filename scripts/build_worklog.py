"""업무일지(docs/worklog/*.md)를 PDF 하나로 묶는다. → docs/worklog/worklog.pdf

사용법
    python scripts/build_worklog.py
"""

import html
import re
import subprocess
import tempfile
from pathlib import Path

from build_report import find_browser

REPO_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_DIR / "docs" / "worklog"
CSS = """body{font-family:'Malgun Gothic',sans-serif;font-size:10.5pt;line-height:1.6;color:#222}
h1{font-size:16pt;border-bottom:2px solid #1f5fa8;padding-bottom:4px;break-before:page;margin-top:0}
h1:first-of-type{break-before:auto}h2{font-size:12pt;color:#1f5fa8;margin:14px 0 4px}
ul{margin:4px 0 4px 20px;padding:0}ul ul{margin:2px 0 2px 18px}li{margin:2px 0}
table{border-collapse:collapse;margin:6px 0;font-size:9.5pt}td,th{border:1px solid #bbb;padding:3px 6px}
code{font-family:Consolas,monospace;font-size:9.5pt;background:#f1f3f5;padding:0 3px}pre{background:#f1f3f5;padding:6px;white-space:pre-wrap}
@page{size:A4;margin:16mm}"""


def inline(text: str) -> str:
    t = html.escape(text, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)


def to_html(md: str) -> str:
    """일지에 쓰는 문법만 옮긴다: 제목, 중첩 목록, 표, 코드 블록, 문단."""
    out, depth, in_table, in_code = [], 0, False, False

    def close_list(to=0):
        nonlocal depth
        while depth > to:
            out.append("</ul>")
            depth -= 1

    def close_table():
        nonlocal in_table
        if in_table:
            out.append("</table>")
            in_table = False

    for line in md.splitlines():
        if line.startswith("```"):
            close_list(), close_table()
            out.append("</pre>" if in_code else "<pre>")
            in_code = not in_code
            continue
        if in_code:
            out.append(html.escape(line, quote=False))
            continue
        m = re.match(r"(\s*)[-*] (.*)", line)
        if m:
            close_table()
            level = len(m.group(1)) // 2 + 1
            while depth < level:
                out.append("<ul>")
                depth += 1
            close_list(level)
            out.append(f"<li>{inline(m.group(2))}</li>")
            continue
        close_list()
        if line.startswith("|"):
            if re.fullmatch(r"[|\s:\-]+", line):
                continue
            if not in_table:
                out.append("<table>")
                in_table = True
            out.append("<tr>" + "".join(f"<td>{inline(c.strip())}</td>" for c in line.strip().strip("|").split("|")) + "</tr>")
            continue
        close_table()
        h = re.match(r"(#{1,3}) (.*)", line)
        if h:
            n = len(h.group(1))
            out.append(f"<h{n}>{inline(h.group(2))}</h{n}>")
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
    close_list(), close_table()
    return "\n".join(out)


def main():
    files = sorted(LOG_DIR.glob("*.md"))
    body = "\n".join(to_html(p.read_text(encoding="utf-8")) for p in files)
    out = LOG_DIR / "worklog.pdf"
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "worklog.html"
        page.write_text(f'<!doctype html><html lang="ko"><meta charset="utf-8"><title>업무일지</title><style>{CSS}</style><body>{body}</body></html>', encoding="utf-8")
        subprocess.run([str(find_browser()), "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={Path(tmp) / 'p'}",
                        f"--print-to-pdf={out}", page.as_uri()], check=True, timeout=180)
    print(out, f"{len(files)}개 일지, {out.stat().st_size / 1e6:.2f}MB")


if __name__ == "__main__":
    main()
