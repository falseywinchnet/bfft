#!/usr/bin/env python3
"""Render an operator-frame fragment with the preserved local viewer shell."""
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fragment", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--shell", type=Path,
        default=Path("ML_experiment/operator_frame_battery.html"),
    )
    args = parser.parse_args()

    shell = args.shell.read_text()
    match = re.fullmatch(
        r'(.*?<iframe\b[^>]*\bsrcdoc=")(.+?)("></iframe>\s*</body>\s*</html>\s*)',
        shell,
        flags=re.DOTALL,
    )
    if match is None:
        raise RuntimeError(f"could not locate srcdoc shell in {args.shell}")
    inner = html.unescape(match.group(2))
    marker = '<div id="curvature-superset-viz">'
    start = inner.find(marker)
    end = inner.rfind("</body>")
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError("viewer shell does not contain the expected fragment root")

    fragment = args.fragment.read_text()
    if not fragment.startswith(marker):
        raise RuntimeError("fragment does not use the operator-frame root")
    rendered_inner = inner[:start] + fragment + "\n" + inner[end:]
    rendered = match.group(1) + html.escape(rendered_inner, quote=True) + match.group(3)
    args.output.write_text(rendered)
    print(f"{args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
