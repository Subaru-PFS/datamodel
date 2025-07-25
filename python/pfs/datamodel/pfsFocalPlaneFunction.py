from abc import ABC, abstractmethod
from typing import Dict

import astropy.io.fits
import numpy as np

from .utils import subclasses

__all__ = (
    "PfsFocalPlaneFunction",
    "PfsConstantFocalPlaneFunction",
    "PfsOversampledSpline",
    "PfsBlockedOversampledSpline",
    "PfsPolynomialPerFiber",
    "PfsFluxCalib",
)


class PfsFocalPlaneFunction(ABC):
    """Base class for spectrum that is a function of position on the focal plane

    This has a variety of uses in calibrating the observed spectra:
    * Flux from the sky as a function of wavelength
    * Intensity of sky emission lines
    * Normalisation as a function of wavelength

    The implementations in the datamodel package handle I/O only. For
    fully-functional implementations that include fitting and evaluation of the
    functions, see the drp_stella package.

    Subclasses must implement the ``__init__``, ``fromFits`` and ``toFits``
    methods.
    """

    _classIdentifier = "pfs_focalPlaneFunction_class"
    """Header keyword to identify the sub-class"""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "This abstract base class cannot be instantiated directly"
        )

    @classmethod
    def readFits(cls, filename: str):
        """Read from FITS file

        Parameters
        ----------
        filename : `str`
            Filename of FITS file.

        Returns
        -------
        self : ``cls``
            Constructed instance, from FITS file.
        """
        subs = {ss.__name__: ss for ss in subclasses(cls)}
        with astropy.io.fits.open(filename, memmap=False) as fits:
            if cls._classIdentifier not in fits[0].header:
                raise RuntimeError(f"Header keyword {cls._classIdentifier} not found")
            name = fits[0].header.get(cls._classIdentifier)
            damdVer = fits[0].header.get("DAMD_VER", None)
            if damdVer is None:
                # Backwards compatibility for files written before I/O moved to datamodel package
                name = "Pfs" + name
            if name not in subs:
                raise RuntimeError(f"Unrecognised {cls._classIdentifier} value: {name}")
            return subs[name].fromFits(fits)

    @classmethod
    @abstractmethod
    def fromFits(cls, fits: astropy.io.fits.HDUList) -> "PfsFocalPlaneFunction":
        """Implementation of reading from a FITS file in memory

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file in memory.

        Returns
        -------
        self : `PfsFocalPlaneFunction`
            Function read from FITS file.
        """
        raise NotImplementedError("Subclasses must override this method")

    def writeFits(self, filename: str):
        """Write to FITS file

        Parameters
        ----------
        filename : `str`
            Filename of FITS file.
        """
        header = astropy.io.fits.Header()
        header["HIERARCH " + self._classIdentifier] = type(self).__name__
        header["DAMD_VER"] = (1, "PfsFocalPlaneFunction datamodel version")
        fits = astropy.io.fits.HDUList(
            [astropy.io.fits.PrimaryHDU(header=header), *self.toFits()]
        )
        with open(filename, "wb") as fd:
            fits.writeto(fd)

    @abstractmethod
    def toFits(self) -> astropy.io.fits.HDUList:
        """Implementation of writing to FITS file

        Returns
        -------
        fits : `astropy.io.fits.HDUList`
            FITS file representation.
        """
        raise NotImplementedError("Subclasses must override this method")


class PfsConstantFocalPlaneFunction(PfsFocalPlaneFunction):
    """Constant function over the focal plane

    This implementation is something of a placeholder, as it simply returns a
    constant vector as a function of wavelength. No attention is paid to the
    position of the fibers on the focal plane.

    Parameters
    ----------
    wavelength : `numpy.ndarray` of `float`, shape ``(N,)``
        Wavelengths for each value.
    value : `numpy.ndarray` of `float`, shape ``(N,)``
        Value at each wavelength.
    mask : `numpy.ndarray` of `bool`, shape ``(N,)``
        Indicate whether values should be ignored.
    variance : `numpy.ndarray` of `float`, shape ``(N,)``
        Variance in value at each wavelength.
    """

    def __init__(self, wavelength, value, mask, variance):
        self.wavelength = wavelength
        self.value = value
        self.mask = mask
        self.variance = variance

    @classmethod
    def fromFits(cls, fits: astropy.io.fits.HDUList) -> "PfsConstantFocalPlaneFunction":
        """Construct from FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file from which to read.

        Returns
        -------
        self : cls
            Constructed focal plane function.
        """
        wavelength = fits["WAVELENGTH"].data
        value = fits["VALUE"].data
        mask = fits["MASK"].data.astype(bool)
        variance = fits["VARIANCE"].data
        return cls(wavelength, value, mask, variance)

    def toFits(self) -> astropy.io.fits.HDUList:
        """Write to FITS file

        Returns
        -------
        fits : `astropy.io.fits.HDUList`
            FITS file to write.
        """
        return astropy.io.fits.HDUList(
            hdus=[
                astropy.io.fits.ImageHDU(self.wavelength, name="WAVELENGTH"),
                astropy.io.fits.ImageHDU(self.value, name="VALUE"),
                astropy.io.fits.ImageHDU(self.mask.astype(np.uint8), name="MASK"),
                astropy.io.fits.ImageHDU(self.variance, name="VARIANCE"),
            ],
        )


class PfsOversampledSpline(PfsFocalPlaneFunction):
    """An oversampled spline in the wavelength dimension, without regard to
    focal plane position

    Parameters
    ----------
    knots : `numpy.ndarray` of `float`
        Spline knots.
    coeffs : `numpy.ndarray` of `float`
        Spline coefficients.
    splineOrder : `int`
        Order of spline.
    wavelength : `numpy.ndarray` of `float`, shape ``(N,)``
        Wavelength array for variance estimation.
    variance : `numpy.ndarray` of `float`, shape ``(N,)``
        Variance array.
    """

    def __init__(
        self,
        knots: np.ndarray,
        coeffs: np.ndarray,
        splineOrder: int,
        wavelength: np.ndarray,
        variance: np.ndarray,
        defaultValue: float,
    ):
        self.knots = knots
        self.coeffs = coeffs
        self.splineOrder = splineOrder
        self.wavelength = wavelength
        self.variance = variance
        self.defaultValue = defaultValue

    @classmethod
    def fromFits(cls, fits: astropy.io.fits.HDUList) -> "PfsOversampledSpline":
        """Construct from FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file from which to read.

        Returns
        -------
        self : cls
            Constructed focal plane function.
        """
        hdu = fits["SPLINE"]
        splineOrder = hdu.header["ORDER"]
        defaultValue = float(hdu.header["DEFAULT"])
        knots = hdu.data["knots"][0]
        coeffs = hdu.data["coeffs"][0]
        wavelength = hdu.data["wavelength"][0]
        variance = hdu.data["variance"][0]
        return cls(knots, coeffs, splineOrder, wavelength, variance, defaultValue)

    def toFits(self) -> astropy.io.fits.HDUList:
        """Write to FITS file

        Returns
        -------
        fits : `astropy.io.fits.HDUList`
            FITS file to write.
        """
        header = astropy.io.fits.Header()
        header["ORDER"] = self.splineOrder
        header["DEFAULT"] = (
            self.defaultValue
            if np.isfinite(self.defaultValue)
            else str(self.defaultValue)
        )
        numKnots = len(self.knots)
        numCoeffs = len(self.coeffs)
        numVar = len(self.variance)
        table = astropy.io.fits.BinTableHDU.from_columns(
            [
                astropy.io.fits.Column(
                    "knots", format=f"{numKnots}D", array=[self.knots]
                ),
                astropy.io.fits.Column(
                    "coeffs", format=f"{numCoeffs}D", array=[self.coeffs]
                ),
                astropy.io.fits.Column(
                    "wavelength", format=f"{numVar}D", array=[self.wavelength]
                ),
                astropy.io.fits.Column(
                    "variance", format=f"{numVar}D", array=[self.variance]
                ),
            ],
            header=header,
            name="SPLINE",
        )
        return astropy.io.fits.HDUList(hdus=[table])


class PfsBlockedOversampledSpline(PfsFocalPlaneFunction):
    """Oversampled splines defined in blocks of fiberId

    Parameters
    ----------
    splines : `dict` [`float`: `OversampledSpline`]
        Splines for each block index.
    """

    def __init__(self, splines: Dict[float, PfsOversampledSpline]):
        self.splines = splines
        self.fiberId = np.sort(np.array(list(splines.keys())))

    @classmethod
    def fromFits(cls, fits: astropy.io.fits.HDUList) -> "PfsBlockedOversampledSpline":
        """Construct from FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file from which to read.

        Returns
        -------
        self : cls
            Constructed focal plane function.
        """
        hdu = fits["BLOCKSPLINE"]
        splines = {
            row["fiberId"]: PfsOversampledSpline(
                row["knots"],
                row["coeffs"],
                row["splineOrder"],
                row["wavelength"],
                row["variance"],
                row["defaultValue"],
            )
            for row in hdu.data
        }
        return cls(splines)

    def toFits(self) -> astropy.io.fits.HDUList:
        """Write to FITS file

        Returns
        -------
        fits : `astropy.io.fits.HDUList`
            FITS file to write.
        """
        header = astropy.io.fits.Header()
        fiberId = list(self.splines.keys())
        splineOrder = [sp.splineOrder for sp in self.splines.values()]
        defaultValue = [sp.defaultValue for sp in self.splines.values()]
        knots = np.array([sp.knots for sp in self.splines.values()], dtype=object)
        coeffs = np.array([sp.coeffs for sp in self.splines.values()], dtype=object)
        wavelength = np.array(
            [sp.wavelength for sp in self.splines.values()], dtype=object
        )
        variance = np.array([sp.variance for sp in self.splines.values()], dtype=object)
        table = astropy.io.fits.BinTableHDU.from_columns(
            [
                astropy.io.fits.Column("fiberId", format="F", array=fiberId),
                astropy.io.fits.Column("splineOrder", format="J", array=splineOrder),
                astropy.io.fits.Column("defaultValue", format="D", array=defaultValue),
                astropy.io.fits.Column("knots", format="PD()", array=knots),
                astropy.io.fits.Column("coeffs", format="PD()", array=coeffs),
                astropy.io.fits.Column("wavelength", format="PD()", array=wavelength),
                astropy.io.fits.Column("variance", format="PD()", array=variance),
            ],
            header=header,
            name="BLOCKSPLINE",
        )
        return astropy.io.fits.HDUList(hdus=[table])


class PfsPolynomialPerFiber(PfsFocalPlaneFunction):
    """A polynomial in wavelength for each fiber independently.

    Parameters
    ----------
    coeffs : `dict` [`int`: `numpy.ndarray` of `float`]
        Polynomial coefficients, indexed by fiberId.
    rms : `dict` [`int`: `float`]
        RMS of residuals from fit, indexed by fiberId.
    minWavelength : `float`
        Minimum wavelength, for normalising the polynomial inputs.
    maxWavelength : `float`
        Maximum wavelength, for normalising the polynomial inputs.
    """

    def __init__(
        self,
        coeffs: Dict[int, np.ndarray],
        rms: Dict[int, float],
        minWavelength: float,
        maxWavelength: float,
    ):
        assert set(coeffs.keys()) == set(rms.keys())
        self.coeffs = coeffs
        self.rms = rms
        self.minWavelength = minWavelength
        self.maxWavelength = maxWavelength

    @property
    def fiberId(self):
        """Fiber identifiers with a corresponding polynomial"""
        return np.array(sorted(self.coeffs.keys()), dtype=int)

    @classmethod
    def fromFits(cls, fits: astropy.io.fits.HDUList) -> "PfsPolynomialPerFiber":
        """Construct from FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file from which to read.

        Returns
        -------
        self : cls
            Constructed focal plane function.
        """
        hdu = fits["POLYPERFIBER"]
        minWavelength = hdu.header["MIN_WL"]
        maxWavelength = hdu.header["MAX_WL"]
        fiberId = hdu.data["fiberId"]
        coeffs = dict(zip(fiberId, hdu.data["coeffs"]))
        rms = dict(zip(fiberId, hdu.data["rms"]))

        return cls(coeffs, rms, minWavelength, maxWavelength)

    def toFits(self) -> astropy.io.fits.HDUList:
        """Write to FITS file

        Returns
        -------
        fits : `astropy.io.fits.HDUList`
            FITS file to write.
        """
        fiberId = self.fiberId

        header = astropy.io.fits.Header()
        header["MIN_WL"] = self.minWavelength
        header["MAX_WL"] = self.maxWavelength
        coeffs = [self.coeffs[ff] for ff in fiberId]
        rms = [self.rms[ff] for ff in fiberId]

        return astropy.io.fits.HDUList(
            hdus=[
                astropy.io.fits.BinTableHDU.from_columns(
                    [
                        astropy.io.fits.Column("fiberId", format="J", array=fiberId),
                        astropy.io.fits.Column("coeffs", format="PD()", array=coeffs),
                        astropy.io.fits.Column("rms", format="D", array=rms),
                    ],
                    header=header,
                    name="POLYPERFIBER",
                ),
            ],
        )


class PfsFluxCalib(PfsFocalPlaneFunction):
    r"""Flux calibration vector such that pfsMerged divided by fluxCalib
    will be the calibrated spectra.

    This is the product of a ConstantFocalPlaneFunction ``h(\lambda)``
    multiplied by the exponential of a trivariate polynomial
    ``g(x, y, \lambda)``, where ``(x, y)`` is the fiber position.
    ``h(\lambda)`` represents the average shape of flux calibration vectors
    up to ``exp g(x, y, \lambda)``. ``exp g(x, y, \lambda)``, which is
    expected to be almost independent of ``\lambda``, represents the overall
    height of a flux calibration vector at ``(x, y)``. The height varies from
    fiber to fiber (or, according to ``(x, y)``) because of imperfect fiber
    positioning. ``g(x, y, \lambda)`` indeed depends slightly on ``\lambda``
    because seeing depends on wavelength.

    Parameters
    ----------
    polyParams : `numpy.ndarray` of `float`
        Parameters used by ``NormalizedPolynomialND`` in ``drp_stella``.
        These parameters define ``g(x, y, \lambda)``.
    polyMin : `numpy.ndarray` of `float`, shape ``(3,)``
        Vertex of the rectangular-parallelepipedal domain of the polynomial
        at which ``(x, y, \lambda)`` are minimal.
    polyMax : `numpy.ndarray` of `float`, shape ``(3,)``
        Vertex of the rectangular-parallelepipedal domain of the polynomial
        at which ``(x, y, \lambda)`` are maximal.
    constantFocalPlaneFunction : `PfsConstantFocalPlaneFunction`
        ``h(\lambda)`` as explaned above.
    polyNewNorm : `bool`, optional
        Use the new normalization scheme? This allows compatibility with
        files written before the normalization scheme was changed (2025 June).
    """

    def __init__(
        self,
        polyParams: np.ndarray,
        polyMin: np.ndarray,
        polyMax: np.ndarray,
        constantFocalPlaneFunction: PfsConstantFocalPlaneFunction,
        polyNewNorm: bool = True,
    ) -> None:
        self.polyParams = polyParams
        self.polyMin = polyMin
        self.polyMax = polyMax
        self.polyNewNorm = polyNewNorm
        self.constantFocalPlaneFunction = constantFocalPlaneFunction

    @classmethod
    def fromFits(cls, fits: astropy.io.fits.HDUList) -> "PfsFluxCalib":
        """Construct from FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file from which to read.

        Returns
        -------
        self : cls
            Constructed focal plane function.
        """
        constantFocalPlaneFunction = PfsConstantFocalPlaneFunction.fromFits(fits)

        catalog = fits["POLYNOMIAL"].data
        polyParams = catalog["params"][0].astype(np.float64)
        polyMin = catalog["min"][0].astype(np.float64)
        polyMax = catalog["max"][0].astype(np.float64)
        polyNewNorm = fits["POLYNOMIAL"].header.get("NEWNORM", False)  # If NEWNORM isn't there, it's old

        return cls(polyParams, polyMin, polyMax, constantFocalPlaneFunction, polyNewNorm)

    def toFits(self) -> astropy.io.fits.HDUList:
        """Write to FITS file

        Returns
        -------
        fits : `astropy.io.fits.HDUList`
            FITS file to write.
        """
        catalog = np.empty(
            shape=(1,),
            dtype=[
                ("params", float, self.polyParams.shape),
                ("min", float, self.polyMin.shape),
                ("max", float, self.polyMax.shape),
            ],
        )
        catalog["params"][0, :] = self.polyParams
        catalog["min"][0, :] = self.polyMin
        catalog["max"][0, :] = self.polyMax

        header = astropy.io.fits.Header()
        header["NEWNORM"] = (self.polyNewNorm, "Use new normalization scheme?")

        return astropy.io.fits.HDUList(
            hdus=[
                *self.constantFocalPlaneFunction.toFits(),
                astropy.io.fits.BinTableHDU(catalog, name="POLYNOMIAL", header=header),
            ]
        )
