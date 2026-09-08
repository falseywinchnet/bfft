from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DEMO = Path(__file__).resolve().parent
for directory in (ROOT, DEMO):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from backend import (  # noqa: E402
    lanczos3_resize,
    linear_resize,
    polyphase_fir_resize,
)
from image_demo import main  # noqa: E402
from experiments.conv_distilled_core import distilled_conv_resize  # noqa: E402


METHODS = {
    "CONV": distilled_conv_resize,
    "Polyphase FIR-8": lambda value, shape: polyphase_fir_resize(
        value, shape, radius=8
    ),
    "Lanczos-sinc (3-lobe)": lanczos3_resize,
    "Bilinear": linear_resize,
}


if __name__ == "__main__":
    main(METHODS)
