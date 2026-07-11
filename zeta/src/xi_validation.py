"""Independent direct-xi oracle and Mellin-identity validation."""
from __future__ import annotations

import csv
from pathlib import Path
import mpmath as mp
from suzuki_kernel import mellin_from_arithmetic, theta_omega_direct, kernel_invariants


def validate_mellin(output: Path, omegas=(2, 1.5, 1.25, 1.1), dps=80):
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with mp.workdps(dps):
        for omega in omegas:
            points = [1j * (omega + 1), 1 + 1j * (omega + 1),
                      3 + 1j * (omega + 1.5)]
            inv = kernel_invariants(omega)
            for z in points:
                arithmetic = mellin_from_arithmetic(omega, z)
                direct = theta_omega_direct(omega, z)
                error = abs(arithmetic - direct) / (1 + abs(direct))
                rows.append({
                    "omega": omega, "z_real": float(mp.re(z)), "z_imag": float(mp.im(z)),
                    "mellin_real": mp.nstr(mp.re(arithmetic), 35),
                    "mellin_imag": mp.nstr(mp.im(arithmetic), 35),
                    "theta_real": mp.nstr(mp.re(direct), 35),
                    "theta_imag": mp.nstr(mp.im(direct), 35),
                    "relative_error": mp.nstr(error, 12),
                    "passed_1e-25": bool(error < mp.mpf("1e-25")),
                    **inv,
                })
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    return rows
