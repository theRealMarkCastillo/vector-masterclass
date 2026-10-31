"""Static notebook export with local math assets and readable code wrapping."""

import json
import re
import shutil
from pathlib import Path

from nbconvert import HTMLExporter

from .config import file_hash, root


def export_html(notebook, output: Path):
    asset = root() / ".cache/render/tex-svg.js"
    record = json.loads((root() / "configs/render_asset.json").read_text())
    if not asset.exists() or file_hash(asset) != record["sha256"]:
        raise ValueError(
            "Local HTML math asset missing or changed; run scripts/download_render_assets.py"
        )
    shared = output.parent.parent / "_assets"
    shared.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(asset, shared / "tex-svg.js")
    exporter = HTMLExporter(template_name="lab", mathjax_url="", require_js_url="")
    html, _ = exporter.from_notebook_node(notebook)
    html = re.sub(
        r"<!-- Load mathjax -->.*?<!-- End of mathjax configuration -->", "", html, flags=re.S
    )
    html = re.sub(r'<script src="">\s*</script>', "", html)
    html = html.replace(
        "<title>Notebook</title>",
        f"<title>{notebook.metadata.get('course', {}).get('id', 'Course')} · Vector Masterclass</title>",
    )
    math = r"""
    <script>window.MathJax = {tex: {inlineMath: [['$', '$'], ['\\(', '\\)']], displayMath: [['$$', '$$'], ['\\[', '\\]']]}, svg: {fontCache: 'local'}};</script>
    <script defer src="../_assets/tex-svg.js"></script>
    <style>.jp-InputArea pre, .jp-OutputArea-output pre {white-space: pre-wrap; overflow-wrap: anywhere;} .jp-RenderedHTMLCommon {line-height: 1.65;} body {font-size: 15px;}</style>
    """
    html = html.replace("</head>", math + "</head>")
    output.write_text(html)
