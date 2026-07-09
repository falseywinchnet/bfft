"""Validate the C++ DIP/PGHI port against the Python reference, bisected."""
import ctypes
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/Downloads"))
import active_delta_center5_cpp_reference as ref  # noqa: E402

L = ref.DEFAULT_L  # 8192

# --- test signal: chirps + tones + noise (distinct spectral magnitudes) ---
rng = np.random.default_rng(1234)
n = np.arange(L)
x = (0.7 * np.cos(2 * np.pi * 300 * n / L)
     + 0.4 * np.cos(2 * np.pi * 1200 * n / L + 0.7)
     + 0.25 * np.cos(2 * np.pi * (500 + 0.03 * n) * n / L)
     + 0.05 * rng.standard_normal(L))
x -= x.mean()
x *= 0.95 / (np.max(np.abs(x)) + 1e-12)
x = np.ascontiguousarray(x, dtype=np.float64)

lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "libiqwaterfall.dylib"))
dp = ctypes.POINTER(ctypes.c_double)
lib.iqw_dip_run.argtypes = [dp, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                            ctypes.c_int, ctypes.c_int, ctypes.c_double,
                            ctypes.c_int, ctypes.c_int, dp, dp]
lib.iqw_dip_seed.argtypes = [dp, ctypes.c_int, dp]
lib.iqw_dip_magnitudes.argtypes = [dp, ctypes.c_int, dp, dp]
lib.iqw_dip_loss0.restype = ctypes.c_double
lib.iqw_dip_loss0.argtypes = [dp, ctypes.c_int]

xp = x.ctypes.data_as(dp)


def rel(a, b):
    return np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-30)


print("=" * 60)
# 1) magnitudes -----------------------------------------------------------
solver = ref.ActiveDeltaCenter5Fast1(x)
Ib, Is = solver.Ib, solver.Is
yb_c = np.empty(Ib * ref.NB); ys_c = np.empty(Is * ref.NS)
lib.iqw_dip_magnitudes(xp, L, yb_c.ctypes.data_as(dp), ys_c.ctypes.data_as(dp))
print(f"[magnitudes] yb maxerr {np.max(np.abs(yb_c.reshape(Ib,ref.NB)-solver.yb)):.2e}"
      f"  ys maxerr {np.max(np.abs(ys_c.reshape(Is,ref.NS)-solver.ys)):.2e}")

# 2) PGHI seed u0 ---------------------------------------------------------
u0_py = ref.pghi_record_seed_from_magnitudes(L, ref.NB, ref.HB, solver.yb)
u0_c = np.empty(L)
lib.iqw_dip_seed(xp, L, u0_c.ctypes.data_as(dp))
print(f"[seed u0]  maxerr {np.max(np.abs(u0_c-u0_py)):.2e}  rel {rel(u0_c,u0_py):.2e}")

# 3) loss0 at seed --------------------------------------------------------
Z0_py, _ = solver.seed()
loss0_py, _ = solver.loss_grad(Z0_py)
loss0_c = lib.iqw_dip_loss0(xp, L)
print(f"[loss0]    py {loss0_py:.8e}  c {loss0_c:.8e}  rel {abs(loss0_c-loss0_py)/abs(loss0_py):.2e}")

# 4) full one-shot output u -----------------------------------------------
u_py, info = solver.run()
u_c = np.empty(L)
loss0_out = ctypes.c_double(0.0)
lib.iqw_dip_run(xp, L, None, 0, 1, 2.5e-4, 1, 0, u_c.ctypes.data_as(dp),
                ctypes.byref(loss0_out))
print(f"[run u]    maxerr {np.max(np.abs(u_c-u_py)):.2e}  rel {rel(u_c,u_py):.2e}")
print(f"[run u]    ||u_py||={np.linalg.norm(u_py):.4f} ||u_c||={np.linalg.norm(u_c):.4f}")
print("=" * 60)

ok = (rel(u_c, u_py) < 1e-9 and rel(u0_c, u0_py) < 1e-9)
print("VALIDATION", "PASS" if ok else "FAIL")
