"""
Render report/eval_report.md to a styled HTML page suitable for "Save as PDF" from a browser.

Usage:
    uv run python report/build_pdf.py
    open report/eval_report.html

Then in the browser: Cmd+P → Save as PDF.
"""

import base64
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
MD_PATH = ROOT / "report" / "eval_report.md"
HTML_PATH = ROOT / "report" / "eval_report.html"

CSS = """
@page { size: letter; margin: 0.3in; }
* { box-sizing: border-box; }
body {
    font-family: -apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif;
    color: #1a1a1a;
    line-height: 1.3;
    max-width: 8in;
    margin: 0 auto;
    padding: 0.15in 0.25in;
    font-size: 9pt;
}
h1 { font-size: 15pt; margin: 0 0 0.03in 0; border-bottom: 1.5px solid #1a1a1a; padding-bottom: 0.02in; }
h2 { font-size: 10.5pt; margin-top: 0.08in; margin-bottom: 0.03in; color: #1a1a1a; page-break-after: avoid; }
h3 { font-size: 10pt; margin-top: 0.06in; margin-bottom: 0.02in; }
p, li { font-size: 8.5pt; margin: 0.03in 0; }
hr { border: none; border-top: 1px solid #ccc; margin: 0.05in 0; }
table { border-collapse: collapse; width: 100%; margin: 0.04in 0; font-size: 8pt; page-break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: 2px 5px; text-align: left; }
th { background: #f5f5f5; font-weight: 600; }
code { font-family: "SF Mono", Monaco, Consolas, monospace; font-size: 8pt; background: #f5f5f5; padding: 1px 3px; border-radius: 3px; }
em { color: #444; }
strong { color: #000; }
img { max-width: 100%; height: auto; }
.chart-row { display: table; width: 100%; table-layout: fixed; margin: 0.03in 0; border-spacing: 0.05in 0; }
.chart-row > * { display: table-cell; vertical-align: top; padding: 0; }
.chart-row img { width: 100%; max-height: 1.25in; object-fit: contain; display: block; }
.chart-row p { margin: 0; }
.subtitle { color: #666; font-size: 10.5pt; margin: 0 0 0.05in 0; }
.metadata { color: #999; font-size: 9pt; margin-bottom: 0.15in; }
"""


def _flush_pending(pending: list[str]) -> str:
    inner = "".join(f"<div>{img}</div>" for img in pending)
    return f'<div class="chart-row">{inner}</div>'


def md_to_html(md_text: str) -> str:
    # Strip YAML front matter
    if md_text.startswith("---"):
        end = md_text.find("---", 3)
        if end != -1:
            md_text = md_text[end + 3 :].lstrip()

    # Strip pandoc-style image width attributes
    md_text = re.sub(r"\)\{[^}]*\}", ")", md_text)

    html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

    # Group consecutive single-image paragraphs into 2-column chart-row divs
    img_p = re.compile(r'<p>\s*(<img[^>]+/?>)\s*</p>')
    parts = []
    pos = 0
    pending = []
    for m in img_p.finditer(html_body):
        if m.start() > pos:
            inter = html_body[pos:m.start()]
            if inter.strip():
                if pending:
                    parts.append(_flush_pending(pending))
                    pending = []
                parts.append(inter)
            elif pending:
                pass  # whitespace between images — keep accumulating
        pending.append(m.group(1))
        pos = m.end()
        if len(pending) == 2:
            parts.append(_flush_pending(pending))
            pending = []
    if pending:
        parts.append(_flush_pending(pending))
    parts.append(html_body[pos:])
    html_body = "".join(parts)

    # Embed PNGs as data URIs so the HTML is portable (works in browser print preview)
    def embed(match):
        src = match.group(1)
        path = (ROOT / "report" / src).resolve()
        if not path.exists():
            # Try relative to the markdown file's directory
            path = (MD_PATH.parent / src).resolve()
        if not path.exists():
            return match.group(0)
        b64 = base64.b64encode(path.read_bytes()).decode()
        return f'src="data:image/png;base64,{b64}"'

    html_body = re.sub(r'src="([^"]+\.png)"', embed, html_body)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AI Vendor Risk Evaluation</title>
<style>{CSS}</style>
</head>
<body>
<h1>AI Vendor Risk Evaluation</h1>
<p class="subtitle">OSS (Qwen2.5-1.5B on Modal) vs Frontier (GPT-4o-mini via OpenRouter) — AI vendor underwriting memo</p>
<p class="metadata">Date: 2026-05-22 · 70 evaluation rows · Judge: Claude Sonnet 4</p>
{html_body}
</body>
</html>
"""


def main():
    md_text = MD_PATH.read_text()
    html = md_to_html(md_text)
    HTML_PATH.write_text(html)
    print(f"Wrote {HTML_PATH}")
    print(f"Open in browser:  open {HTML_PATH}")
    print(f"Then Cmd+P → Save as PDF")


if __name__ == "__main__":
    main()
