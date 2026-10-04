"""Fixed algebraic compilation of two tensor quintics and a bilinear weight.

Scalar reference accepting floats or exact Fractions. No source analysis,
collocation, clipping, subdivision, or state selection is performed here.
Controls use [y][x] order. This is a coefficient reference, not a WASM kernel.
"""

from fractions import Fraction


def compile_blend(forward, reverse, beta):
    """Return 7x7 controls for (1-beta)*forward + beta*reverse exactly.

    Exactness is algebraic; floating-point inputs retain rounding error.
    Both order fields must already be represented by 6x6 controls.
    beta contains its four bilinear corner values, in [y][x] order.
    """
    if any(len(p) != 6 or any(len(row) != 6 for row in p)
           for p in (forward, reverse)):
        raise ValueError("Order controls must each be 6x6")
    if len(beta) != 2 or any(len(row) != 2 for row in beta):
        raise ValueError("Blend controls must be 2x2")
    result = [[0 for _ in range(7)] for _ in range(7)]
    for j in range(7):
        for i in range(7):
            for s in range(2):
                for r in range(2):
                    y, x = j-s, i-r
                    # These bounds depend only on the fixed coefficient layout.
                    if not (0 <= y < 6 and 0 <= x < 6):
                        continue
                    weight = Fraction((j if s else 6-j)*(i if r else 6-i), 36)
                    b = beta[s][r]
                    result[j][i] += weight*((1-b)*forward[y][x]+b*reverse[y][x])
    return result
