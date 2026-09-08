"""Run the rank-five Mellin-current probe and a rank-one internal control."""

from __future__ import annotations

import json
from pathlib import Path

from mellin_current import EllipticCurve, current_report


RECORD = EllipticCurve(
    conductor=19_047_851,
    a1=0,
    a2=0,
    a3=1,
    a4=-79,
    a6=342,
    name="rank5_conductor_19047851",
)

# A small rank-one control already familiar from the integral model
# y^2+y=x^3-x.  It is not used as external data or as a search seed.
CONTROL = EllipticCurve(
    conductor=37,
    a1=0,
    a2=0,
    a3=1,
    a4=-1,
    a6=0,
    name="rank1_conductor_37",
)


def main() -> None:
    record = current_report(RECORD, coefficient_limit=30_000)
    control = current_report(CONTROL, coefficient_limit=2_000)
    output = {"record": record, "control": control}
    target = Path("/tmp/elliptic_mellin_current.json")
    target.write_text(json.dumps(output, indent=2))
    for key, report in output.items():
        print(key, report["curve"])
        print("  moments", report["moments"])
        print("  cancellation", report["cancellation_ratios"])
        print("  sign changes", report["current_sign_changes"])
    print("wrote", target)


if __name__ == "__main__":
    main()

