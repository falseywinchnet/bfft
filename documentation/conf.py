project = "BFFT"
author = "BFFT contributors"
release = "0.1.0"
version = "0.1.0"

extensions = [
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]
html_theme = "furo"
html_title = "BFFT documentation"

myst_heading_anchors = 3
