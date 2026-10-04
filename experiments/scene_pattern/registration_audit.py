"""Region-level registration evidence, independent of object names or transforms."""
import numpy as np
from scipy import ndimage
from .core import srgb_to_lab

def contrast_error(reference,moving,mask):
    if not np.any(mask):return None
    a=srgb_to_lab(reference)[...,0];b=srgb_to_lab(moving)[...,0]
    a=a-ndimage.gaussian_filter(a,3);b=b-ndimage.gaussian_filter(b,3)
    return float(np.mean(np.abs(a[mask]-b[mask])))

def field_evidence(delta,measured_support,inspection):
    """A smooth/no-fold field does not imply correspondence support."""
    inside=inspection&measured_support;outside=inspection&~measured_support
    magnitude=np.linalg.norm(delta,axis=-1)
    return dict(inspected_pixels=int(inspection.sum()),measured_pixels=int(inside.sum()),extrapolated_pixels=int(outside.sum()),extrapolated_max_displacement=float(magnitude[outside].max()) if np.any(outside) else 0.,extrapolated_median_displacement=float(np.median(magnitude[outside])) if np.any(outside) else 0.)

def nonregressing_regions(checks,tolerance=.02):
    """A gain in one region cannot conceal a regression or unknown in another."""
    return bool(checks) and all(c['before'] is not None and c['after'] is not None and np.isfinite(c['before']) and np.isfinite(c['after']) and c['after']<=c['before']*(1+tolerance) for c in checks)
