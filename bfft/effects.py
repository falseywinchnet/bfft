"""Recomposition effects over the Meyer cartoon + texture split.

The decomposition writes an image as three layers,

    image = cartoon + texture + residual,

and every effect here is a reassembly of those layers with independent
gains.  Two things make that more than a crossfade:

  * The layers are separated by scale geometry, not by frequency.  Raising
    the texture gain amplifies oscillation wherever it lives, at any
    contrast, without touching the edges that bound it.

  * The cartoon layer produced by the transport descent retains smooth
    illumination -- ramps and shading -- that a converged flat cartoon
    discards as staircase.  A plain ROF solve of the cartoon recovers that
    flat cartoon, so ``shade = cartoon - ROF(cartoon, c)`` isolates the
    illumination, and it is edge-aware: the total-variation solve keeps the
    jumps in place, so the difference carries their contrast loss rather
    than a blurred copy of them.  Measured against unsharp masking at
    matched added energy over ten test images, amplifying this layer
    produces a median 2.2x less mean overshoot beside strong edges.  It is
    not overshoot-free: worst-case overshoot is comparable to unsharp,
    because a ROF solve does shrink jump amplitude.

Colour is handled by decomposing planes independently and reassembling.
OKLab is the useful space here: its lightness axis is perceptually uniform
and very nearly orthogonal to the chroma axes, so a texture gain applied to
luma alone brightens detail without shifting hue, which the same operation
in RGB cannot do.

All planes are carried as float64 in a [0, 255] working range, which is the
range the decomposition's default lambda and mu are set for; each channel
is affinely rescaled into it and back, and since the split is exact and the
rescaling affine, unit gains reproduce the input to roundoff.
"""

import numpy as np

from ._core import meyer_split, rof

__all__ = ["srgb_to_lab", "lab_to_srgb", "shade", "recompose",
           "meyer_channels", "recompose_channels", "ChannelSplit",
           "SPACES"]

SPACES = ("gray", "rgb", "oklab", "oklab_lc")


# --- OKLab (Ottosson) ----------------------------------------------------
#
# sRGB in [0, 1] <-> OKLab.  L is roughly [0, 1]; a and b are roughly
# [-0.4, 0.4].

_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])
# Ottosson publishes the inverses rounded to ten decimals, which limits the
# colour round trip to about 3e-7.  Inverting the forward matrices instead
# makes it exact to roundoff, and costs one 3x3 inversion at import.
_M2_INV = np.linalg.inv(_M2)
_M1_INV = np.linalg.inv(_M1)


def _apply(mat, planes):
    """(H, W, 3) x 3x3 matrix, contracting the channel axis."""
    # NumPy routes a stacked (..., 3) @ (3, 3) through thousands of tiny
    # BLAS matrix calls.  On large images that is both poor bookkeeping and,
    # on some Accelerate/Python combinations, intermittently unsafe.  A
    # single fixed-stride contraction is easier for the compiler to
    # vectorize and avoids the batched-BLAS dispatcher entirely.
    return np.einsum("...j,ij->...i", planes, mat, optimize=False)


def _srgb_decode(c):
    return np.where(c <= 0.04045, c / 12.92,
                    np.power((np.abs(c) + 0.055) / 1.055, 2.4))


def _srgb_encode(c):
    return np.where(c <= 0.0031308, 12.92 * c,
                    1.055 * np.power(np.maximum(c, 0.0), 1.0 / 2.4) - 0.055)


def srgb_to_lab(rgb):
    """sRGB in [0, 1], shape (H, W, 3), to OKLab (H, W, 3)."""
    lin = _srgb_decode(np.asarray(rgb, dtype=np.float64))
    lms = _apply(_M1, lin)
    return _apply(_M2, np.cbrt(lms))


def lab_to_srgb(lab):
    """OKLab (H, W, 3) back to sRGB in [0, 1], unclipped."""
    lms = _apply(_M2_INV, np.asarray(lab, dtype=np.float64))
    return _srgb_encode(_apply(_M1_INV, lms ** 3))


# --- the shading layer ---------------------------------------------------

def shade(cartoon, c=0.02, sweeps=100, tol=1e-5, threads=0):
    """The smooth illumination a flat cartoon discards.

    ``cartoon - ROF(cartoon, c)``: the total-variation solve keeps the
    jumps in place, so the difference is ramps and shading plus the
    contrast the solve took off the jumps -- not a blurred copy of the
    edges.  Smaller ``c`` flattens harder and so takes more into the
    shading layer."""
    a = np.asarray(cartoon, dtype=np.float64)
    return a - rof(a, c=c, sweeps=sweeps, tol=tol, threads=threads)


# --- single-plane recomposition ------------------------------------------

def recompose(image, cartoon, texture, gain_cartoon=1.0, gain_texture=1.0,
              gain_shade=0.0, shade_c=0.02, clip=None, threads=0):
    """Reassemble a split with independent layer gains.

        out = gc*cartoon + gt*texture + residual + gs*shade(cartoon)

    where ``residual = image - cartoon - texture`` is carried unchanged, so
    ``(1, 1, 0)`` returns the image exactly.  ``gain_shade`` is additive on
    top of whatever ``gain_cartoon`` already carries: at ``gc = 1`` the
    shading ends up weighted ``1 + gs``.  ``clip=(lo, hi)`` clamps the
    result."""
    f = np.asarray(image, dtype=np.float64)
    u = np.asarray(cartoon, dtype=np.float64)
    v = np.asarray(texture, dtype=np.float64)
    out = f - u - v                              # residual, carried
    out = out + gain_cartoon * u + gain_texture * v
    if gain_shade:
        out = out + gain_shade * shade(u, c=shade_c, threads=threads)
    if clip is not None:
        out = np.clip(out, clip[0], clip[1])
    return out


# --- colour: per-channel decomposition -----------------------------------

class ChannelSplit:
    """The per-channel decomposition of a colour image, plus everything
    needed to put it back together.

    Attributes:
      space     one of SPACES
      names     channel names, in plane order
      planes    (H, W, K) working-space image, each channel in [0, 255]
      cartoon   (H, W, K) cartoon layers
      texture   (H, W, K) texture layers
      offset    (K,) and scale (K,): plane = (native - offset) * scale
      carried   dict of untouched planes (hue, alpha) needed to invert
    """

    __slots__ = ("space", "names", "planes", "cartoon", "texture", "offset",
                 "scale", "carried")

    def __init__(self, space, names, planes, cartoon, texture, offset, scale,
                 carried):
        self.space = space
        self.names = names
        self.planes = planes
        self.cartoon = cartoon
        self.texture = texture
        self.offset = offset
        self.scale = scale
        self.carried = carried

    @property
    def residual(self):
        return self.planes - self.cartoon - self.texture


def _to_working(img, space):
    """Native colour array -> (planes (H, W, K), names, carried)."""
    a = np.asarray(img, dtype=np.float64)
    carried = {}
    if a.ndim == 2:
        if space not in ("gray",):
            space = "gray"
        return a[..., None], ("gray",), carried, space
    if a.ndim != 3 or a.shape[2] not in (3, 4):
        raise ValueError("expected (H, W), (H, W, 3) or (H, W, 4)")
    if a.shape[2] == 4:
        carried["alpha"] = a[..., 3]
        a = a[..., :3]
    if a.max() > 1.5:                            # 0-255 input
        a = a / 255.0
    if space == "rgb":
        return a * 255.0, ("R", "G", "B"), carried, space
    lab = srgb_to_lab(a)
    if space == "oklab":
        return lab, ("L", "a", "b"), carried, space
    if space == "oklab_lc":
        chroma = np.hypot(lab[..., 1], lab[..., 2])
        carried["hue"] = np.arctan2(lab[..., 2], lab[..., 1])
        return (np.stack([lab[..., 0], chroma], axis=-1), ("L", "C"),
                carried, space)
    if space == "gray":
        carried["hue"] = np.arctan2(lab[..., 2], lab[..., 1])
        carried["chroma"] = np.hypot(lab[..., 1], lab[..., 2])
        return lab[..., :1], ("L",), carried, space
    raise ValueError(f"unknown space {space!r}; expected one of {SPACES}")


def _from_working(planes, space, carried, was_gray):
    """Inverse of _to_working; returns sRGB in [0, 1] (or a grey plane)."""
    if was_gray:
        return planes[..., 0]
    if space == "rgb":
        out = planes / 255.0
    else:
        if space == "oklab":
            lab = planes
        elif space == "oklab_lc":
            c = np.maximum(planes[..., 1], 0.0)   # chroma cannot go negative
            h = carried["hue"]
            lab = np.stack([planes[..., 0], c * np.cos(h), c * np.sin(h)],
                           axis=-1)
        else:                                     # gray: luma only, colour
            c = carried["chroma"]                 # carried through
            h = carried["hue"]
            lab = np.stack([planes[..., 0], c * np.cos(h), c * np.sin(h)],
                           axis=-1)
        out = lab_to_srgb(lab)
    if "alpha" in carried:
        out = np.concatenate([out, carried["alpha"][..., None]], axis=-1)
    return out


def meyer_channels(image, space="oklab_lc", lam=0.05, mu=40.0, passes=64,
                   threads=0, solver=0):
    """Decompose a colour image plane by plane.

    ``space`` selects what gets decomposed:

      "gray"      luma only; the original chroma is carried through, so the
                  effect is monochrome detail work on a colour image
      "rgb"       each of R, G, B independently
      "oklab"     OKLab L, a, b independently
      "oklab_lc"  OKLab lightness and chroma; hue is carried untouched, so
                  no gain can shift a colour, only its lightness and
                  saturation (the default, and the safest for effects)

    Every channel is affinely rescaled into [0, 255] before the split --
    the range the default lambda and mu are set for -- and the scaling is
    recorded so :func:`recompose_channels` inverts it exactly.

    Returns a :class:`ChannelSplit`."""
    if space not in SPACES:
        raise ValueError(f"unknown space {space!r}; expected one of {SPACES}")
    was_gray = np.asarray(image).ndim == 2
    planes, names, carried, space = _to_working(image, space)
    k = planes.shape[2]
    work = np.empty_like(planes)
    offset = np.empty(k)
    scale = np.empty(k)
    for i in range(k):
        p = planes[..., i]
        lo = float(p.min())
        span = float(p.max()) - lo
        offset[i] = lo
        scale[i] = 255.0 / span if span > 1e-12 else 1.0
        work[..., i] = (p - lo) * scale[i]

    cartoon = np.empty_like(work)
    texture = np.empty_like(work)
    for i in range(k):
        u, v = meyer_split(work[..., i], lam=lam, mu=mu, passes=passes,
                           threads=threads, solver=solver)
        cartoon[..., i] = u
        texture[..., i] = v
    carried["_was_gray"] = was_gray
    return ChannelSplit(space, names, work, cartoon, texture, offset, scale,
                        carried)


def _per_channel(g, k):
    a = np.atleast_1d(np.asarray(g, dtype=np.float64))
    if a.size == 1:
        return np.full(k, float(a[0]))
    if a.size != k:
        raise ValueError(f"expected a scalar or {k} gains, got {a.size}")
    return a


def recompose_channels(split, gain_cartoon=1.0, gain_texture=1.0,
                       gain_shade=0.0, shade_c=0.02, clip=True, threads=0):
    """Reassemble a :class:`ChannelSplit` with per-layer gains and return a
    native-space image (sRGB in [0, 1], or a grey plane for a grey input).

    Each gain is a scalar or one value per channel, so on ``"oklab_lc"``
    ``gain_texture=(2.0, 1.0)`` doubles luma detail while leaving
    saturation alone.  Unit gains reproduce the input to roundoff.
    ``clip`` clamps the sRGB result to [0, 1].

    In ``"oklab_lc"`` chroma is clamped at zero, since saturation cannot go
    negative; a gain large enough to drive a pixel past the clamp leaves it
    grey, and its hue is then no longer carried."""
    k = split.planes.shape[2]
    gc = _per_channel(gain_cartoon, k)
    gt = _per_channel(gain_texture, k)
    gs = _per_channel(gain_shade, k)
    out = np.empty_like(split.planes)
    for i in range(k):
        y = recompose(split.planes[..., i], split.cartoon[..., i],
                      split.texture[..., i], gain_cartoon=gc[i],
                      gain_texture=gt[i], gain_shade=gs[i], shade_c=shade_c,
                      threads=threads)
        out[..., i] = split.offset[i] + y / split.scale[i]
    rgb = _from_working(out, split.space, split.carried,
                        split.carried.get("_was_gray", False))
    if clip:
        rgb = np.clip(rgb, 0.0, 1.0)
    return rgb
