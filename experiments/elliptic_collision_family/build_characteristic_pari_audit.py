"""Build an exact PARI audit for the first norm-conic characteristics.

This keeps the existing theoretical ordering: on each constructed slice,
only continued-fraction convergents to the discriminant wall are considered,
and they remain ordered by integral discriminant.  PARI computes a global
minimal conductor first and runs the exact rank descent only below the live
rank-six target.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from run_rank_six_wall_descent import CURRENT_SLICES, descent


TARGET = 5_187_563_742


def gp_program(maximum_denominator: int, per_slice: int) -> str:
    entries = []
    for slice_name, current_slice in CURRENT_SLICES.items():
        fibers = descent(
            maximum_denominator,
            certify_count=0,
            current_slice=current_slice,
        )
        for ordinal, fiber in enumerate(fibers[:per_slice], start=1):
            label = f"{slice_name}_{ordinal}_t{current_slice.t}_c{fiber.c}"
            label = label.replace("/", "o").replace("-", "m")
            entries.append((label, (0, 0, fiber.a3, fiber.a4, 0)))
    names = ",".join(f'"{name}"' for name, _ in entries)
    models = ",".join(str(list(model)).replace(" ", "") for _, model in entries)
    return (
        f"target={TARGET};names=[{names}];models=[{models}];"
        "hits=0;for(i=1,#models,E=ellinit(models[i]);N=ellglobalred(E)[1];"
        "if(N<target,hits++;r=ellrank(E);print(names[i],\" conductor=\",N,"
        "\" root=\",ellrootno(E),\" rank=\",r[1..2],\" model=\",models[i])));"
        'print("subtarget_characteristics=",hits);quit;\n'
    )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--maximum-denominator", type=int, default=100)
    parser.add_argument("--per-slice", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(
        gp_program(args.maximum_denominator, args.per_slice),
        encoding="utf-8",
    )
    print(args.out)


if __name__ == "__main__":
    main()
