"""Each fiber's fine structure in throughput as a function of cobra position (theta, phi)

On top of the smooth model (`PfsThetaPhiModel`), each fiber's throughput has its own fine structure as its
cobra moves (about 0.2% rms, on scales of a few degrees), stable over months.  It is described as a Gaussian
process per fiber, trained on the sky residuals from the smooth model at converged cobra positions:  the file
holds each fiber's training points (theta_i, phi_i) and weights w_i (the GP's (K + N)^-1 y), so the layer
anywhere is::

    f(theta, phi) = sum_i w_i sum_c AMP_c^2 M32(r_ic)
    r_ic = sqrt((dtheta_i/LTHETA_c)^2 + ((phi - phi_i)/LPHI_c)^2)
    dtheta_i = ((theta - theta_i + 180) mod 360) - 180
    M32(r) = (1 + sqrt(3) r) exp(-sqrt(3) r)

and the factor applied to a spectrum's norm is 1 + f;  1 for a cobraHome exposure and for a fiber with no
layer.  The layer is the sky's correction:  it is applied to all light, although quartz light shares only
about a quarter of it (an error of ~0.07% rms for quartz, none at home).

The layer is trained per arm, and refers to the smooth model it was fitted against (by its hash).

Angles are in degrees, measured from the CCW hard stops as in the cobra calibration, the FPS and the
pfsConfig:  theta from tht0 and phi from phiIn (as for `PfsThetaPhiModel`).
"""
import enum
from typing import Any, Dict, Optional

import numpy as np
import astropy.io.fits

from .pfsTable import Column, PfsTableWithSeparateStrings
from .utils import astropyHeaderFromDict, astropyHeaderToDict

__all__ = ("ThetaPhiFineFlag", "PfsThetaPhiFine", "ThetaPhiFineQa")

SQRT3 = np.sqrt(3)


class ThetaPhiFineFlag(enum.IntFlag):
    """Flags for each fiber's layer"""

    NO_LAYER = 0x1
    """No layer:  too few points, or the validity test failed;  the correction is 1"""
    STALE = 0x2
    """The repeat correlation against the previous run is low:  a possible break (advisory)"""
    FEW_POINTS = 0x4
    """Few training points (advisory)"""


def matern32(r: np.ndarray) -> np.ndarray:
    """The Matern 3/2 correlation, (1 + sqrt(3) r) exp(-sqrt(3) r)"""
    return (1 + SQRT3*r)*np.exp(-SQRT3*r)


class PfsThetaPhiFine:
    """Each fiber's fine structure in throughput across its patrol region, for one arm

    Parameters
    ----------
    fiberId : `numpy.ndarray` of `int`, shape ``(numFibers,)``
        Fiber identifiers.
    pointStart : `numpy.ndarray` of `int`, shape ``(numFibers,)``
        Index of each fiber's first training point.
    numPoints : `numpy.ndarray` of `int`, shape ``(numFibers,)``
        Number of each fiber's training points (contiguous from ``pointStart``).
    amplitude : `numpy.ndarray` of `float`, shape ``(numFibers, numTerms)``
        Amplitude (rms, a fraction) of each kernel term.
    lTheta, lPhi : `numpy.ndarray` of `float`, shape ``(numFibers, numTerms)``
        Length scales (degrees) of each kernel term in theta and phi.
    flags : `numpy.ndarray` of `int`, shape ``(numFibers,)``
        `ThetaPhiFineFlag` bits.
    repeatCorr : `numpy.ndarray` of `float`, shape ``(numFibers,)``
        The validity test:  the repeat correlation against the previous run (NaN if not measured).
    varRemoved : `numpy.ndarray` of `float`, shape ``(numFibers,)``
        Share of the variance removed, leaving one night out (NaN if not measured).
    theta, phi : `numpy.ndarray` of `float`, shape ``(totalPoints,)``
        The training points' cobra angles (degrees).
    weight : `numpy.ndarray` of `float`, shape ``(totalPoints,)``
        The training points' weights (1/fraction).
    errDistance, errScale : `numpy.ndarray` of `float`, shape ``(numErr,)``
        The error model:  the layer's error is ``sqrt(sum_c AMP_c^2)`` times ``errScale`` interpolated at the
        distance to the nearest training point (in units of the first kernel term's length scales) in
        ``errDistance``.
    metadata : `dict` (`str`: POD), optional
        Keyword-value pairs for the header (arm, band, provenance, ``SMOOTH_HASH``, ...).
    """

    def __init__(
        self,
        fiberId: np.ndarray,
        pointStart: np.ndarray,
        numPoints: np.ndarray,
        amplitude: np.ndarray,
        lTheta: np.ndarray,
        lPhi: np.ndarray,
        flags: np.ndarray,
        repeatCorr: np.ndarray,
        varRemoved: np.ndarray,
        theta: np.ndarray,
        phi: np.ndarray,
        weight: np.ndarray,
        errDistance: np.ndarray,
        errScale: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.fiberId = np.asarray(fiberId, dtype=int)
        self.pointStart = np.asarray(pointStart, dtype=int)
        self.numPoints = np.asarray(numPoints, dtype=int)
        self.amplitude = np.atleast_2d(np.asarray(amplitude, dtype=float))
        self.lTheta = np.atleast_2d(np.asarray(lTheta, dtype=float))
        self.lPhi = np.atleast_2d(np.asarray(lPhi, dtype=float))
        self.flags = np.asarray(flags, dtype=int)
        self.repeatCorr = np.asarray(repeatCorr, dtype=float)
        self.varRemoved = np.asarray(varRemoved, dtype=float)
        self.theta = np.asarray(theta, dtype=float)
        self.phi = np.asarray(phi, dtype=float)
        self.weight = np.asarray(weight, dtype=float)
        self.errDistance = np.asarray(errDistance, dtype=float)
        self.errScale = np.asarray(errScale, dtype=float)
        self.metadata = dict(metadata) if metadata else {}
        self._index = {int(fid): ii for ii, fid in enumerate(self.fiberId)}
        self.validate()

    @property
    def numTerms(self) -> int:
        """Number of terms in the kernel"""
        return self.amplitude.shape[1]

    @property
    def arm(self) -> Optional[str]:
        """The arm the layer was trained for"""
        return self.metadata.get("ARM")

    def validate(self):
        """Check that the arrays have consistent shapes"""
        num = len(self.fiberId)
        for name in ("pointStart", "numPoints", "flags", "repeatCorr", "varRemoved"):
            assert getattr(self, name).shape == (num,), name
        for name in ("amplitude", "lTheta", "lPhi"):
            assert getattr(self, name).shape == (num, self.numTerms), name
        total = len(self.theta)
        assert self.phi.shape == (total,) and self.weight.shape == (total,)
        assert np.all(self.pointStart + self.numPoints <= total)
        assert self.errDistance.shape == self.errScale.shape

    def __len__(self) -> int:
        """Return the number of fibers"""
        return len(self.fiberId)

    def __contains__(self, fiberId: int) -> bool:
        """Is the fiber present?"""
        return int(fiberId) in self._index

    def _points(self, index: int) -> slice:
        """The training points of the fiber with this index"""
        return slice(self.pointStart[index], self.pointStart[index] + self.numPoints[index])

    def _kernel(self, index: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """The kernel between (theta, phi) and the fiber's training points;  shape (len(theta), numPoints)"""
        pts = self._points(index)
        dTheta = (theta[:, None] - self.theta[pts][None, :] + 180) % 360 - 180
        dPhi = phi[:, None] - self.phi[pts][None, :]
        kernel = np.zeros(dTheta.shape)
        for cc in range(self.numTerms):
            rr = np.hypot(dTheta/self.lTheta[index, cc], dPhi/self.lPhi[index, cc])
            kernel += self.amplitude[index, cc]**2*matern32(rr)
        return kernel

    def evaluate(self, fiberId: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Return the layer f at (theta, phi)

        Parameters
        ----------
        fiberId : array_like of `int`
            Fiber identifiers.
        theta, phi : array_like of `float`
            Cobra angles (degrees);  broadcast with ``fiberId``.

        Returns
        -------
        f : `numpy.ndarray` of `float`
            The layer (a fraction);  0 for fibers with no layer (NO_LAYER, or not present).
        """
        fiberId, theta, phi = np.broadcast_arrays(np.asarray(fiberId), np.asarray(theta, dtype=float),
                                                  np.asarray(phi, dtype=float))
        result = np.zeros(theta.shape)
        for fid in np.unique(fiberId):
            index = self._index.get(int(fid))
            if index is None or self.flags[index] & ThetaPhiFineFlag.NO_LAYER or self.numPoints[index] == 0:
                continue
            select = fiberId == fid
            kernel = self._kernel(index, theta[select], phi[select])
            result[select] = kernel @ self.weight[self._points(index)]
        return result

    def getCorrection(
        self, fiberId: np.ndarray, theta: np.ndarray, phi: np.ndarray, isCobraHome: np.ndarray
    ) -> np.ndarray:
        """Return the factor by which to multiply the norm of a spectrum (``corrFine``)

        Parameters
        ----------
        fiberId : array_like of `int`
            Fiber identifiers.
        theta, phi : array_like of `float`
            Cobra angles (degrees), as in the pfsConfig;  broadcast with ``fiberId``.
        isCobraHome : array_like of `bool`
            Is each fiber's cobra in the cobraHome state?  (Decided by the caller from the pfsConfig's
            ``cobraCommand``, as for `PfsThetaPhiModel.getCorrection`.)

        Returns
        -------
        factor : `numpy.ndarray` of `float`
            1 + f;  1 for cobraHome exposures and fibers with no layer.
        """
        fiberId, theta, phi, isCobraHome = np.broadcast_arrays(
            np.asarray(fiberId), np.asarray(theta, dtype=float), np.asarray(phi, dtype=float),
            np.asarray(isCobraHome, dtype=bool)
        )
        return np.where(isCobraHome, 1.0, 1.0 + self.evaluate(fiberId, theta, phi))

    def getError(self, fiberId: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Return the approximate error in f at (theta, phi)

        From the distance to the fiber's nearest training point, in units of the first kernel term's length
        scales, via the error model (``errDistance``, ``errScale``).

        Parameters
        ----------
        fiberId : array_like of `int`
            Fiber identifiers.
        theta, phi : array_like of `float`
            Cobra angles (degrees);  broadcast with ``fiberId``.

        Returns
        -------
        error : `numpy.ndarray` of `float`
            The error (a fraction);  NaN for fibers with no layer.
        """
        fiberId, theta, phi = np.broadcast_arrays(np.asarray(fiberId), np.asarray(theta, dtype=float),
                                                  np.asarray(phi, dtype=float))
        result = np.full(theta.shape, np.nan)
        for fid in np.unique(fiberId):
            index = self._index.get(int(fid))
            if index is None or self.flags[index] & ThetaPhiFineFlag.NO_LAYER or self.numPoints[index] == 0:
                continue
            select = fiberId == fid
            pts = self._points(index)
            dTheta = (theta[select][:, None] - self.theta[pts][None, :] + 180) % 360 - 180
            dPhi = phi[select][:, None] - self.phi[pts][None, :]
            distance = np.min(np.hypot(dTheta/self.lTheta[index, 0], dPhi/self.lPhi[index, 0]), axis=1)
            total = np.sqrt(np.sum(self.amplitude[index]**2))
            result[select] = total*np.interp(distance, self.errDistance, self.errScale)
        return result

    def checkSmoothModel(self, model) -> None:
        """Check that this layer was fitted against a given smooth model

        Parameters
        ----------
        model : `pfs.datamodel.PfsThetaPhiModel`
            The smooth model.

        Raises
        ------
        RuntimeError
            If the layer records a different model.
        """
        expected = self.metadata.get("SMOOTH_HASH")
        if expected is None or int(expected) != model.hash:
            raise RuntimeError(f"thetaPhiFine was fitted against thetaPhiModel {expected}, not {model.hash}")

    @classmethod
    def _readImpl(cls, fits: astropy.io.fits.HDUList) -> Dict[str, Any]:
        """Implementation for reading from FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            Opened FITS file.

        Returns
        -------
        kwargs : ``dict``
            Keyword arguments for constructing PfsThetaPhiFine.
        """
        header = fits[0].header
        version = header["DAMD_VER"]
        if version != 1:
            raise RuntimeError(f"Cannot read PfsThetaPhiFine with DAMD_VER={version}")
        if header.get("KERNEL") != "matern32sum":
            raise RuntimeError(f"Unknown kernel {header.get('KERNEL')}")
        numTerms = int(header["NTERM"])
        numErr = int(header.get("ERR_N", 0))
        metadata = astropyHeaderToDict(header)
        for key in ("SIMPLE", "BITPIX", "NAXIS", "EXTEND", "DAMD_VER", "KERNEL", "NTERM", "ANGLE_UNIT",
                    "THETA0", "PHI0", "THETA_WRAP", "ERR_N"):
            metadata.pop(key, None)
        errDistance = np.array([metadata.pop(f"ERR_D{ii}") for ii in range(numErr)], dtype=float)
        errScale = np.array([metadata.pop(f"ERR_G{ii}") for ii in range(numErr)], dtype=float)

        fibers = fits["FIBERS"].data
        points = fits["POINTS"].data

        def terms(name):
            return np.array([fibers[f"{name}_{cc + 1}"] for cc in range(numTerms)], dtype=float).T

        return dict(
            fiberId=fibers["FIBERID"].astype(int),
            pointStart=fibers["POINT0"].astype(int),
            numPoints=fibers["NPOINT"].astype(int),
            amplitude=terms("AMP"),
            lTheta=terms("LTHETA"),
            lPhi=terms("LPHI"),
            flags=fibers["FLAGS"].astype(int),
            repeatCorr=fibers["REPEAT_CORR"].astype(float),
            varRemoved=fibers["VAR_REMOVED"].astype(float),
            theta=points["THETA"].astype(float),
            phi=points["PHI"].astype(float),
            weight=points["WEIGHT"].astype(float),
            errDistance=errDistance,
            errScale=errScale,
            metadata=metadata,
        )

    @classmethod
    def readFits(cls, filename: str) -> "PfsThetaPhiFine":
        """Read from FITS file

        This API is intended for use by the LSST data butler, which handles
        translating the desired identity into a filename.

        Parameters
        ----------
        filename : `str`
            Filename of FITS file.

        Returns
        -------
        self : ``cls``
            Constructed instance, from FITS file.
        """
        with astropy.io.fits.open(filename) as fd:
            data = cls._readImpl(fd)
        return cls(**data)

    def _writeImpl(self, fits: astropy.io.fits.HDUList):
        """Implementation for writing to FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            List of FITS HDUs.
        """
        # NOTE: When making any changes to this method that modify the output
        # format, increment the DAMD_VER header value and record the change in
        # the versions.txt file.
        from astropy.io.fits import BinTableHDU, Column as FitsColumn, Header, PrimaryHDU

        header = astropyHeaderFromDict(self.metadata)
        header["DAMD_VER"] = (1, "PfsThetaPhiFine datamodel version")
        header["KERNEL"] = ("matern32sum", "f = sum_i w_i sum_c AMP_c^2 M32(r_ic)")
        header["NTERM"] = (self.numTerms, "Number of kernel terms")
        header["HIERARCH ANGLE_UNIT"] = ("deg", "Units of theta and phi")
        header["THETA0"] = ("tht0", "theta is measured from the CCW theta hard stop")
        header["PHI0"] = ("phiIn", "phi is measured from the CCW phi hard stop")
        header["HIERARCH THETA_WRAP"] = (360, "theta differences are wrapped into [-180, 180)")
        header["ERR_N"] = (len(self.errDistance), "Number of points in the error model")
        for ii, (dd, gg) in enumerate(zip(self.errDistance, self.errScale)):
            header[f"ERR_D{ii}"] = (float(dd), "Error model:  distance (fine-scale units)")
            header[f"ERR_G{ii}"] = (float(gg), "Error model:  error/total amplitude")
        fits.append(PrimaryHDU(header=header))

        inherit = Header()
        inherit["INHERIT"] = True
        columns = [
            FitsColumn(name="FIBERID", format="J", array=self.fiberId),
            FitsColumn(name="POINT0", format="J", array=self.pointStart),
            FitsColumn(name="NPOINT", format="J", array=self.numPoints),
        ]
        for name, values, unit in (("AMP", self.amplitude, None), ("LTHETA", self.lTheta, "deg"),
                                   ("LPHI", self.lPhi, "deg")):
            for cc in range(self.numTerms):
                column = FitsColumn(name=f"{name}_{cc + 1}", format="E", unit=unit, array=values[:, cc])
                columns.append(column)
        columns += [
            FitsColumn(name="FLAGS", format="J", array=self.flags),
            FitsColumn(name="REPEAT_CORR", format="E", array=self.repeatCorr),
            FitsColumn(name="VAR_REMOVED", format="E", array=self.varRemoved),
        ]
        fits.append(BinTableHDU.from_columns(columns, name="FIBERS", header=inherit))
        fits.append(BinTableHDU.from_columns([
            FitsColumn(name="THETA", format="E", unit="deg", array=self.theta),
            FitsColumn(name="PHI", format="E", unit="deg", array=self.phi),
            FitsColumn(name="WEIGHT", format="E", array=self.weight),
        ], name="POINTS", header=inherit))

    def writeFits(self, filename: str):
        """Write to FITS file

        This API is intended for use by the LSST data butler, which handles
        translating the desired identity into a filename.

        Parameters
        ----------
        filename : `str`
            Filename of FITS file.
        """
        fits = astropy.io.fits.HDUList()
        self._writeImpl(fits)
        with open(filename, "wb") as fd:
            fits.writeto(fd)


class ThetaPhiFineQa(PfsTableWithSeparateStrings):
    """QA for a fine layer (`PfsThetaPhiFine`):  its training data and held-out predictions

    One row per point.  Nothing in the pipeline reads it;  it is enough to refit the layer (the weights
    follow from ``y``, ``sigma`` and the kernel) and to redo its validation.
    """

    schema = [
        Column("fiberId", np.int32, "Fiber identifier", -1),
        Column("visit", np.int32, "Exposure identifier", -1),
        Column("night", str, "Night of the exposure (ISO-8601 date of the evening)", ""),
        Column("source", str, "Kind of light:  sky or quartz", ""),
        Column("theta", np.float32, "Cobra theta angle (degrees)", np.nan),
        Column("phi", np.float32, "Cobra phi angle (degrees)", np.nan),
        Column("y", np.float32, "Residual from the smooth model (fraction)", np.nan),
        Column("sigma", np.float32, "Error in y (fraction)", np.nan),
        Column("used", bool, "Used in the training set?", False),
        Column("looPred", np.float32, "Prediction leaving the point's night out (fraction)", np.nan),
    ]
    fitsExtName = "THETAPHIFINEQA"
