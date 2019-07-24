import numpy as np
from scipy.interpolate import interp1d

__all__ = ["interpolateFlux", "interpolateMask"]


def interpolateFlux(fromWavelength, fromFlux, toWavelength, fill=0.0):
    """Interpolate a flux-like spectrum

    Basic linear interpolation, suitable for fluxes and flux-like (e.g., maybe
    variances) quantities.

    Parameters
    ----------
    fromWavelength : array-like of `float`
        Source wavelength array.
    fromFlux : array-like of `float`
        Source flux(-like) array.
    toWavelength : array-like of `float`
        Target wavelength array.
    fill : `float`, optional
        Fill value.

    Returns
    -------
    toFlux : `numpy.ndarray` of `float`
        Target flux-(like) array.
    """
    with np.errstate(invalid="ignore"):
        return interp1d(fromWavelength, fromFlux, kind="linear", bounds_error=False,
                        fill_value=fill, copy=True, assume_sorted=True)(toWavelength)


def interpolateMask(fromWavelength, fromMask, toWavelength, fill=0):
    """Interpolate a mask spectrum

    Nearest-neighbour interpolation, suitable for masks.

    Parameters
    ----------
    fromWavelength : array-like of `float`
        Source wavelength array.
    fromMask : array-like of `float`
        Source mask array.
    toWavelength : array-like of `float`
        Target wavelength array.
    fill : `float`, optional
        Fill value.

    Returns
    -------
    toMask : `numpy.ndarray` of `float`
        Target mask array.
    """
    def impl(kind):
        with np.errstate(invalid="ignore"):
            return interp1d(fromWavelength, fromMask, kind=kind, bounds_error=False,
                            fill_value=fill, copy=True, assume_sorted=True
                            )(toWavelength).astype(fromMask.dtype)

    try:
        # kind="previous" and kind="next" requires scipy v1.1.0 or later
        return np.bitwise_or(impl("previous"), impl("next"))
    except NotImplementedError:
        # Grow the bad pixels by 1
        array = impl("nearest")
        result = array.copy()
        result[1:] |= array[:-1]
        result[:-1] |= array[1:]
        return result
