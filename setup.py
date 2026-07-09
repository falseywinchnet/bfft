"""Build hook that compiles the BFFT shared library from source at install time.

No prebuilt binaries ship in the distribution. We compile src/bfft.cpp and
src/bodft.cpp into a single shared object and place it inside the bfft package
directory, where the ctypes loader (bfft/_core.py) finds it at runtime. The
compiler invocation mirrors the project Makefile recipe.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
SOURCES = ["src/bfft.cpp", "src/bodft.cpp", "src/fct.cpp", "src/stft.cpp"]
INCLUDE = "include"


def _shared_lib_suffix() -> str:
    # ctypes.CDLL loads a plain ".so" fine on macOS too, and the project Makefile
    # already produces "libbfft.so" there, so we keep one name across Unix.
    if sys.platform == "win32":
        return ".dll"
    return ".so"


def _env_off(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _compiler_accepts(cxx: str, flags: list) -> bool:
    """Return True if the compiler accepts ``flags`` on a trivial source file."""
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "probe.cpp"
        src.write_text("int main() { return 0; }\n")
        try:
            subprocess.run(
                [cxx, *flags, "-c", str(src), "-o", str(Path(d) / "probe.o")],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return True
        except (subprocess.CalledProcessError, OSError):
            return False


def _optimization_flags(cxx: str) -> list:
    """Pick the strongest optimization flags the host compiler accepts.

    Because the library is compiled on the install machine (no prebuilt binaries
    are distributed), tuning for the local CPU is safe and is the default. Set
    BFFT_NO_NATIVE=1 to skip CPU-native codegen and BFFT_NO_FAST_MATH=1 to keep
    strict IEEE math.
    """
    flags = ["-O3", "-DNDEBUG"]

    if not _env_off("BFFT_NO_NATIVE"):
        # x86 / older clang / gcc use -march=native; Apple-silicon clang wants
        # -mcpu=native instead. Probe and take whichever the compiler accepts.
        if _compiler_accepts(cxx, ["-march=native"]):
            flags.append("-march=native")
        elif _compiler_accepts(cxx, ["-mcpu=native"]):
            flags.append("-mcpu=native")

    return flags


def _compile_shared_library(out_path: Path) -> None:
    cxx = os.environ.get("CXX", "c++")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [cxx, *_optimization_flags(cxx), "-std=c++17", "-fPIC", "-shared"]
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
