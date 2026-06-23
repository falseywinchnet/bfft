"""Build hook that compiles the BFFT shared library from source at install time.

No prebuilt binaries ship in the distribution. We compile src/bfft.cpp and
src/bodft.cpp into a single shared object and place it inside the bfft package
directory, where the ctypes loader (bfft/_core.py) finds it at runtime. The
compiler invocation mirrors the project Makefile recipe.
"""

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
SOURCES = ["src/bfft.cpp", "src/bodft.cpp"]
INCLUDE = "include"


def _shared_lib_suffix() -> str:
    # ctypes.CDLL loads a plain ".so" fine on macOS too, and the project Makefile
    # already produces "libbfft.so" there, so we keep one name across Unix.
    if sys.platform == "win32":
        return ".dll"
    return ".so"


def _compile_shared_library(out_path: Path) -> None:
    cxx = os.environ.get("CXX", "c++")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [cxx, "-O3", "-std=c++17", "-fPIC", "-shared"]
    cmd += ["-I", str(ROOT / INCLUDE)]
    cmd += [str(ROOT / s) for s in SOURCES]
    cmd += ["-o", str(out_path)]
    if sys.platform != "win32":
        cmd += ["-lm"]
    extra = os.environ.get("BFFT_CXXFLAGS")
    if extra:
        cmd += extra.split()
    print("bfft: compiling native library:\n  " + " ".join(cmd), file=sys.stderr)
    subprocess.check_call(cmd)


class build_py_with_native(build_py):
    def run(self) -> None:
        super().run()
        # Place the compiled lib alongside the package modules in the build tree.
        target_dir = Path(self.build_lib) / "bfft"
        target_dir.mkdir(parents=True, exist_ok=True)
        out = target_dir / ("_libbfft" + _shared_lib_suffix())
        _compile_shared_library(out)


setup(cmdclass={"build_py": build_py_with_native})
