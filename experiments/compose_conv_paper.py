"""Compose the modular CONV paper into one compact archival TeX target."""

from __future__ import annotations

from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "output" / "pdf"
MAIN = PDF_DIR / "eikonal_interpolation.tex"
CONSERVATIVE_CORE = PDF_DIR / "conv_conservative_core.tex"
SYNTHETIC = PDF_DIR / "conv_synthetic_evaluation.tex"
CONSERVATIVE_EVALUATION = PDF_DIR / "conv_conservative_evaluation.tex"
CONVSTAR = PDF_DIR / "convstar_insert.tex"
WITNESSED = PDF_DIR / "conv_witnessed_transport.tex"
WARP_APPENDIX = PDF_DIR / "conv_warp_appendix.tex"
TARGET = PDF_DIR / "conv_paper_composed.tex"


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one {label}, found {count}")
    return text.replace(old, new, 1)


def _synthetic_body() -> str:
    text = SYNTHETIC.read_text()
    text = _replace_once(
        text,
        "\\input{conv_conservative_evaluation.tex}",
        "% BEGIN CONSERVATIVE MULTIRESOLUTION EVIDENCE\n"
        + CONSERVATIVE_EVALUATION.read_text().rstrip()
        + "\n% END CONSERVATIVE MULTIRESOLUTION EVIDENCE",
        label="conservative evaluation input",
    )
    text = _replace_once(
        text,
        "\\clearpage\n\\raggedbottom\n\\section{Synthetic Evaluation}",
        "\\raggedbottom\n\\section{Synthetic Evaluation}",
        label="synthetic opening page break",
    )
    text = _replace_once(
        text,
        "\\clearpage\n\\subsection{Technical synthetic fields}",
        "\\subsection{Technical synthetic fields}",
        label="technical-plate page break",
    )
    return text.rstrip() + "\n"


def _convstar_body() -> str:
    """Return the insert in the document's native IEEE two-column flow."""
    text = CONVSTAR.read_text()
    text = _replace_once(
        text,
        "\\onecolumn\n\\begin{multicols}{2}\n",
        "",
        label="CONV* one-column wrapper opening",
    )
    text = _replace_once(
        text,
        "\\end{multicols}\n",
        "",
        label="CONV* multicols wrapper closing",
    )
    text = _replace_once(
        text,
        "\n\\twocolumn\n",
        "\n",
        label="CONV* two-column restoration",
    )
    return text


def export_fused(text: str) -> Path:
    """Export one fused TeX file with a flat, relocatable figure directory."""
    assets = PDF_DIR / "conv_paper_assets"
    assets.mkdir(exist_ok=True)
    warp_root = "../support_geometry"
    text = text.replace(r"\providecommand{\WarpFigureRoot}{../support_geometry}",
                        r"\providecommand{\WarpFigureRoot}{conv_paper_assets}")
    def relocate(match):
        path = match.group(2)
        source = PDF_DIR / path.replace(r"\WarpFigureRoot", warp_root)
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = assets / source.name
        shutil.copy2(source, destination)
        return match.group(1) + "{conv_paper_assets/" + source.name + "}"
    text = re.sub(r"(\\includegraphics(?:\[[^\]]*\])?)\{([^}]+)\}", relocate, text)
    if re.search(r"\\(?:input|include)\{", text):
        raise RuntimeError("unfused TeX input remains")
    target = PDF_DIR / "conv_paper_fused.tex"
    target.write_text(text)
    return target


def compose() -> Path:
    text = MAIN.read_text()
    text = _replace_once(
        text,
        "\\usepackage[hidelinks]{hyperref}\n",
        "\\usepackage[hidelinks]{hyperref}\n"
        "\\setlength{\\textfloatsep}{8pt plus 2pt minus 2pt}\n"
        "\\setlength{\\floatsep}{7pt plus 2pt minus 2pt}\n"
        "\\setlength{\\intextsep}{7pt plus 2pt minus 2pt}\n",
        label="compact float-spacing insertion point",
    )
    text = _replace_once(
        text,
        "\\input{conv_witnessed_transport.tex}",
        "% BEGIN WITNESSED EIKONAL TRANSPORT\n"
        + WITNESSED.read_text().rstrip()
        + "\n% END WITNESSED EIKONAL TRANSPORT",
        label="witnessed transport input",
    )
    text = _replace_once(
        text,
        "\\input{conv_conservative_core.tex}",
        "% BEGIN CONSERVATIVE MULTIRESOLUTION CORE\n"
        + CONSERVATIVE_CORE.read_text().rstrip()
        + "\n% END CONSERVATIVE MULTIRESOLUTION CORE",
        label="conservative core input",
    )
    text = _replace_once(
        text,
        "\\input{conv_synthetic_evaluation.tex}",
        "% BEGIN COMPOSED SYNTHETIC EVIDENCE\n"
        + _synthetic_body()
        + "% END COMPOSED SYNTHETIC EVIDENCE",
        label="synthetic input",
    )
    text = _replace_once(
        text,
        "\\input{convstar_insert.tex}",
        "% BEGIN COMPOSED CONVSTAR\n"
        + _convstar_body().rstrip()
        + "\n% END COMPOSED CONVSTAR",
        label="CONV* input",
    )
    text = _replace_once(
        text,
        "\\input{conv_warp_appendix.tex}",
        "% BEGIN COMPOSED DIRECT WARP APPENDIX\n"
        + WARP_APPENDIX.read_text().rstrip()
        + "\n% END COMPOSED DIRECT WARP APPENDIX",
        label="direct warp appendix input",
    )
    text = _replace_once(
        text,
        "\\enlargethispage{3\\baselineskip}\n",
        "\\enlargethispage{2\\baselineskip}\n",
        label="legacy inversion page enlargement",
    )
    text = _replace_once(
        text,
        "\\clearpage\n\\section{Inversion in Conventional Resampling}",
        "\\section{Inversion in Conventional Resampling}",
        label="legacy inversion page break",
    )
    text = _replace_once(
        text,
        "\\clearpage\n\\begin{thebibliography}{99}",
        "\\begin{thebibliography}{99}",
        label="bibliography page break",
    )
    TARGET.write_text(text)
    export_fused(text)
    return TARGET


if __name__ == "__main__":
    print(compose())
