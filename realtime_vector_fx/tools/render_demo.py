"""Package rvfx_demo's dependency-free PPM frames as a looping GIF."""

from pathlib import Path
import sys
from PIL import Image


source = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/rvfx_demo_frames")
destination = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/rvfx_demo.gif")
frames = [Image.open(path).convert("RGB") for path in sorted(source.glob("frame_*.ppm"))]
if not frames:
    raise SystemExit(f"no PPM frames in {source}")
frames[0].save(destination, save_all=True, append_images=frames[1:], duration=33,
               loop=0, optimize=False)
print(destination)
