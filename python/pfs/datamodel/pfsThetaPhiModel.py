"""The smooth model of each fiber's throughput as a function of cobra position (theta, phi)

A fiber's throughput depends on where its cobra puts it.  For each fiber the model is multiplicative::

    flux(theta, phi) = h (1 + s(theta, phi))        [ x (1 + delta_run) for a cobraHome exposure]

where h is a nuisance level (per night), and the deviation s is zero at the cobra's home position::

    s(theta, phi) = sum_k c_k [b_k(theta, phi) - b_k(thetaHome, phiHome)]

So 1 + s is the throughput at (theta, phi) relative to the smooth surface at home, and 1 + delta the
throughput of a cobraHome exposure relative to that surface.  The fiberNorms are measured at home, so a
spectrum taken with the cobra anywhere else is corrected by (1 + alpha s)/(1 + delta), where alpha (per arm)
is the share of s that the sky sees.  A cobraHome exposure needs no correction.

The basis functions b_k are products of cos, sin or 1 of integer multiples of theta and phi (and optionally a
ramp in theta);  the BASIS table in the file defines them, so the model can be evaluated from datamodel.txt
alone.

Angles are in degrees, measured from the CCW hard stops as in the cobra calibration, the FPS and the
pfsConfig:  theta from tht0 (so home, the CW stop, is at 360 + (tht1 - tht0) mod 360) and phi from phiIn
(home is 0).
"""
import enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import astropy.io.fits

from .utils import astropyHeaderFromDict, astropyHeaderToDict, createHash

__all__ = ("ThetaPhiModelFlag", "makeThetaPhiBasis", "evaluateThetaPhiBasis", "PfsThetaPhiModel")


class ThetaPhiModelFlag(enum.IntFlag):
    """Quality flags for each fiber's model"""

    NO_MODEL = 0x1
    """No model was fitted (no usable data);  the coefficients are NaN"""
    FIT_FAILED = 0x2
    """The fit is degenerate or absurd (e.g. |delta| or |s| far too large)"""
    BROKEN_COBRA = 0x4
    """BROKENCOBRA in at least one input visit"""
    POOR_COVERAGE = 0x8
    """The scans covered too little of phi, or had too few points"""
    EDGE = 0x10
    """The cobra is shadowed at the edge of the PFI (e.g. by the AG cameras)"""


BasisFunction = Tuple[int, str, int, str]
"""A basis function:  (theta harmonic, theta function, phi harmonic, phi function)

The functions are ``cos``, ``sin``, ``one`` (harmonic 0) or ``lin``:  a ramp, (harmonic theta)/360, in theta
measured from tht0 (not periodic, so it can tell theta from theta + 360)
"""


def makeThetaPhiBasis(order: int, pureHarmonics: int = 1, thetaRamp: bool = False) -> List[BasisFunction]:
    """Return the standard basis for a model of the given order

    The cross terms have theta and phi harmonics summing to at most ``order``;  the pure theta and phi terms
    go up to ``max(1, pureHarmonics)``;  a ramp in theta is added last if ``thetaRamp``.  There's no constant
    term, as it can't contribute to a deviation that is zero at home.

    Parameters
    ----------
    order : `int`
        Highest total harmonic of the cross terms.
    pureHarmonics : `int`, optional
        Highest harmonic of the terms in theta alone and phi alone.
    thetaRamp : `bool`, optional
        Include a ramp in theta?

    Returns
    -------
    basis : `list` of `tuple`
        The basis functions (see `BasisFunction`).
    """
    basis = [(1, "sin", 0, "one"), (1, "cos", 0, "one"), (0, "one", 1, "cos"), (0, "one", 1, "sin")]
    for it in range(1, order + 1):
        for ip in range(1, order + 1):
            if it + ip <= order:
                basis += [(it, "cos", ip, "cos"), (it, "cos", ip, "sin"), (it, "sin", ip, "cos"),
                          (it, "sin", ip, "sin")]
    for k in range(2, pureHarmonics + 1):
        basis += [(k, "cos", 0, "one"), (k, "sin", 0, "one"), (0, "one", k, "cos"), (0, "one", k, "sin")]
    if thetaRamp:
        basis.append((1, "lin", 0, "one"))
    return basis


def evaluateThetaPhiBasis(basis: Sequence[BasisFunction], theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Evaluate the basis functions at (theta, phi)

    Parameters
    ----------
    basis : sequence of `tuple`
        The basis functions (see `BasisFunction`).
    theta, phi : array_like of `float`
        Cobra angles (degrees);  broadcast together.

    Returns
    -------
    values : `numpy.ndarray` of `float`
        The functions' values;  the last dimension runs over the basis.
    """
    funcs = dict(cos=np.cos, sin=np.sin, one=np.ones_like, lin=lambda x: x/(2*np.pi))
    theta, phi = np.broadcast_arrays(np.deg2rad(np.asarray(theta, dtype=float)),
                                     np.deg2rad(np.asarray(phi, dtype=float)))
    return np.stack([funcs[tf](it*theta)*funcs[pf](ip*phi) for it, tf, ip, pf in basis], axis=-1)


class PfsThetaPhiModel:
    """The smooth (theta, phi) throughput model of every fiber in the PFI

    One model serves the whole PFI and every arm:  it is measured with quartz light in one band
    (``BAND_LO``, ``BAND_HI`` in the metadata), and is achromatic to 0.06-0.10% from 420 to 870 nm.  The
    share of it that the sky sees, alpha, is measured per arm.

    Parameters
    ----------
    fiberId : `numpy.ndarray` of `int`, shape ``(numFibers,)``
        Fiber identifiers.
    basis : `list` of `tuple`
        The basis functions (see `BasisFunction`).
    coeffs : `numpy.ndarray` of `float`, shape ``(numFibers, numCoeffs)``
        Coefficients of each fiber's model (NaN if there's no model).
    homeTheta, homePhi : `numpy.ndarray` of `float`, shape ``(numFibers,)``
        The cobra angles (degrees) at home, where s is zero.
    flags : `numpy.ndarray` of `int`, shape ``(numFibers,)``
        `ThetaPhiModelFlag` bits.
    numPoints : `numpy.ndarray` of `int`, shape ``(numFibers,)``
        Number of points used in each fiber's fit.
    phiMin, phiMax : `numpy.ndarray` of `float`, shape ``(numFibers,)``
        Range of phi (degrees) covered by each fiber's data.
    thetaCoverage : `numpy.ndarray` of `float`, shape ``(numFibers,)``
        Fraction of the theta circle covered by each fiber's data.
    rms : `numpy.ndarray` of `float`, shape ``(numFibers,)``
        Robust rms of each fiber's fit residuals (a fraction).
    modelError : `numpy.ndarray` of `float`, shape ``(numFibers,)``
        Error in each fiber's s (a fraction);  0 if not measured.
    runs : `list` of `str`
        Labels of the runs with a measured delta, in order of their start.
    runStart : `list` of `str`
        Date (ISO-8601) of the first night of each run.
    homeOffsets : `numpy.ndarray` of `float`, shape ``(numFibers, numRuns)``
        delta for each fiber and run (NaN if not measured).
    alpha : `dict` mapping `str` to `float`
        The sky's share of s, indexed by arm.
    alphaSample : `dict` mapping `str` to `str`
        What each arm's alpha was measured from (runs and selection).
    metadata : `dict` (`str`: POD), optional
        Keyword-value pairs for the header (provenance, fit settings, band, calibration date).
    """

    def __init__(
        self,
        fiberId: np.ndarray,
        basis: Sequence[BasisFunction],
        coeffs: np.ndarray,
        homeTheta: np.ndarray,
        homePhi: np.ndarray,
        flags: np.ndarray,
        numPoints: np.ndarray,
        phiMin: np.ndarray,
        phiMax: np.ndarray,
        thetaCoverage: np.ndarray,
        rms: np.ndarray,
        modelError: np.ndarray,
        runs: Sequence[str],
        runStart: Sequence[str],
        homeOffsets: np.ndarray,
        alpha: Dict[str, float],
        alphaSample: Dict[str, str],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.fiberId = np.asarray(fiberId, dtype=int)
        self.basis = [(int(it), str(tf), int(ip), str(pf)) for it, tf, ip, pf in basis]
        self.coeffs = np.asarray(coeffs, dtype=float)
        self.homeTheta = np.asarray(homeTheta, dtype=float)
        self.homePhi = np.asarray(homePhi, dtype=float)
        self.flags = np.asarray(flags, dtype=int)
        self.numPoints = np.asarray(numPoints, dtype=int)
        self.phiMin = np.asarray(phiMin, dtype=float)
        self.phiMax = np.asarray(phiMax, dtype=float)
        self.thetaCoverage = np.asarray(thetaCoverage, dtype=float)
        self.rms = np.asarray(rms, dtype=float)
        self.modelError = np.asarray(modelError, dtype=float)
        self.runs = list(runs)
        self.runStart = list(runStart)
        self.homeOffsets = np.asarray(homeOffsets, dtype=float).reshape(len(self.fiberId), len(self.runs))
        self.alpha = dict(alpha)
        self.alphaSample = dict(alphaSample)
        self.metadata = dict(metadata) if metadata else {}
        self._index = {int(fid): ii for ii, fid in enumerate(self.fiberId)}
        self.validate()

    def validate(self):
        """Check that the arrays have consistent shapes"""
        num = len(self.fiberId)
        assert self.coeffs.shape == (num, len(self.basis))
        for name in ("homeTheta", "homePhi", "flags", "numPoints", "phiMin", "phiMax", "thetaCoverage", "rms",
                     "modelError"):
            assert getattr(self, name).shape == (num,), name
        assert len(self.runStart) == len(self.runs)
        assert self.homeOffsets.shape == (num, len(self.runs))
        assert set(self.alphaSample) <= set(self.alpha)

    def __len__(self) -> int:
        """Return the number of fibers"""
        return len(self.fiberId)

    def __contains__(self, fiberId: int) -> bool:
        """Is the fiber present?"""
        return int(fiberId) in self._index

    @property
    def hash(self) -> int:
        """A hash of the model's content

        Recorded by the products fitted against this model (e.g. `PfsThetaPhiFine`), so that a mismatched pair
        can be detected.  Not ``__hash__``, whose seed varies between runs.
        """
        return createHash(
            self.fiberId.tobytes(), self.basis, self.coeffs.tobytes(), self.homeTheta.tobytes(),
            self.homePhi.tobytes(), self.runs, self.homeOffsets.tobytes(), sorted(self.alpha.items()),
        )

    def _indices(self, fiberId: np.ndarray) -> np.ndarray:
        """Return the index of each fiberId, or -1 if it isn't present"""
        fiberId = np.asarray(fiberId)
        index = [self._index.get(int(ff), -1) for ff in fiberId.ravel()]
        return np.array(index, dtype=int).reshape(fiberId.shape)

    def evaluate(self, fiberId: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Return the deviation s at (theta, phi):  the throughput relative to the surface at home, minus 1

        Parameters
        ----------
        fiberId : array_like of `int`
            Fiber identifiers.
        theta, phi : array_like of `float`
            Cobra angles (degrees);  broadcast with ``fiberId``.

        Returns
        -------
        s : `numpy.ndarray` of `float`
            The deviation;  zero at home, and NaN for unknown fibers or fibers with no model.
        """
        fiberId, theta, phi = np.broadcast_arrays(np.asarray(fiberId), np.asarray(theta, dtype=float),
                                                  np.asarray(phi, dtype=float))
        index = self._indices(fiberId)
        known = index >= 0
        result = np.full(theta.shape, np.nan)
        if np.any(known):
            ii = index[known]
            values = evaluateThetaPhiBasis(self.basis, theta[known], phi[known])
            home = evaluateThetaPhiBasis(self.basis, self.homeTheta[ii], self.homePhi[ii])
            result[known] = np.sum((values - home)*self.coeffs[ii], axis=-1)
        return result

    def getHomeAngles(self, fiberId: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return each fiber's home angles

        Parameters
        ----------
        fiberId : array_like of `int`
            Fiber identifiers.

        Returns
        -------
        homeTheta, homePhi : `numpy.ndarray` of `float`
            The cobra angles (degrees) at home;  NaN for unknown fibers.
        """
        index = self._indices(fiberId)
        known = index >= 0
        safe = np.clip(index, 0, None)
        return (np.where(known, self.homeTheta[safe], np.nan), np.where(known, self.homePhi[safe], np.nan))

    def getRun(self, date: str) -> Optional[str]:
        """Return the run whose delta applies on a date:  the latest to start on or before it

        Parameters
        ----------
        date : `str`
            Date (ISO-8601) of the observation.

        Returns
        -------
        run : `str` or `None`
            The run's label;  `None` if no run started by then.
        """
        started = [(start, run) for start, run in zip(self.runStart, self.runs) if start[:10] <= date[:10]]
        return max(started)[1] if started else None

    def getHomeOffset(self, fiberId: np.ndarray, run: str) -> np.ndarray:
        """Return delta for a run:  a cobraHome exposure's throughput relative to the surface at home, minus 1

        Parameters
        ----------
        fiberId : array_like of `int`
            Fiber identifiers.
        run : `str`
            Label of the run (see `getRun`).

        Returns
        -------
        delta : `numpy.ndarray` of `float`
            NaN if not measured, or for unknown fibers.
        """
        index = self._indices(fiberId)
        if run not in self.runs:
            return np.full(index.shape, np.nan)
        kk = self.runs.index(run)
        return np.where(index >= 0, self.homeOffsets[np.clip(index, 0, None), kk], np.nan)

    def getCorrection(
        self,
        fiberId: np.ndarray,
        theta: np.ndarray,
        phi: np.ndarray,
        arm: str,
        date: str,
        isCobraHome: np.ndarray,
    ) -> np.ndarray:
        """Return the factor by which to multiply the norm of a spectrum (``corrThetaPhi``)

        The fiberNorms are measured at home, so the factor is (1 + alpha s)/(1 + delta) for a fiber anywhere
        else, and 1 for a cobraHome exposure (both arms against their stops).

        Parameters
        ----------
        fiberId : array_like of `int`
            Fiber identifiers.
        theta, phi : array_like of `float`
            Cobra angles (degrees), as in the pfsConfig;  broadcast with ``fiberId``.
        arm : `str`
            Arm of the spectra (selects alpha).
        date : `str`
            Date (ISO-8601) of the observation (selects delta;  see `getRun`).
        isCobraHome : array_like of `bool`
            Is each fiber's cobra in the cobraHome state?  Decided by the caller from the pfsConfig's
            ``cobraCommand`` (not from the angles, which mislabel the cobraHome state).

        Returns
        -------
        factor : `numpy.ndarray` of `float`
            The correction factor;  NaN where there's no usable model (see `isGood`) or no delta.
        """
        if arm not in self.alpha:
            raise KeyError(f"No alpha for arm {arm}: {sorted(self.alpha)}")
        fiberId, theta, phi, isCobraHome = np.broadcast_arrays(
            np.asarray(fiberId), np.asarray(theta, dtype=float), np.asarray(phi, dtype=float),
            np.asarray(isCobraHome, dtype=bool)
        )
        run = self.getRun(date)
        delta = self.getHomeOffset(fiberId, run) if run is not None else np.full(theta.shape, np.nan)
        factor = (1 + self.alpha[arm]*self.evaluate(fiberId, theta, phi))/(1 + delta)
        factor = np.where(self.isGood(fiberId), factor, np.nan)
        return np.where(isCobraHome, 1.0, factor)

    def isGood(self, fiberId: np.ndarray, badFlags: Optional[int] = None) -> np.ndarray:
        """Does each fiber have a usable model?

        Parameters
        ----------
        fiberId : array_like of `int`
            Fiber identifiers.
        badFlags : `int`, optional
            `ThetaPhiModelFlag` bits that make a model unusable;  default NO_MODEL | FIT_FAILED.

        Returns
        -------
        good : `numpy.ndarray` of `bool`
        """
        if badFlags is None:
            badFlags = ThetaPhiModelFlag.NO_MODEL | ThetaPhiModelFlag.FIT_FAILED
        index = self._indices(fiberId)
        return (index >= 0) & ((self.flags[np.clip(index, 0, None)] & int(badFlags)) == 0)

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
            Keyword arguments for constructing PfsThetaPhiModel.
        """
        version = fits[0].header["DAMD_VER"]
        if version != 1:
            raise RuntimeError(f"Cannot read PfsThetaPhiModel with DAMD_VER={version}")
        metadata = astropyHeaderToDict(fits[0].header)
        for key in ("SIMPLE", "BITPIX", "NAXIS", "EXTEND", "DAMD_VER", "MODEL", "ANGLE_UNIT", "THETA0",
                    "PHI0", "PFS.HASH.THETAPHIMODEL"):
            metadata.pop(key, None)

        table = fits["BASIS"].data
        basis = [(int(it), tf.strip(), int(ip), pf.strip()) for it, tf, ip, pf in
                 zip(table["THETA_HARMONIC"], table["THETA_FUNC"], table["PHI_HARMONIC"], table["PHI_FUNC"])]
        # The BASIS table defines the model;  ORDER and PURE_HARMONICS describe it, so must agree
        if "ORDER" in metadata and "PURE_HARMONICS" in metadata:
            expected = makeThetaPhiBasis(metadata["ORDER"], metadata["PURE_HARMONICS"],
                                         bool(metadata.get("THETA_RAMP", False)))
            if basis != expected:
                raise RuntimeError(f"BASIS table isn't the basis for ORDER={metadata['ORDER']}, "
                                   f"PURE_HARMONICS={metadata['PURE_HARMONICS']}")

        fibers = fits["FIBERS"].data
        runs = fits["RUNS"].data
        alpha = fits["ALPHA"].data
        return dict(
            fiberId=fibers["FIBERID"].astype(int),
            basis=basis,
            coeffs=fits["COEFFS"].data.astype(float),
            homeTheta=fibers["HOME_THETA"].astype(float),
            homePhi=fibers["HOME_PHI"].astype(float),
            flags=fibers["FLAGS"].astype(int),
            numPoints=fibers["NUM_POINTS"].astype(int),
            phiMin=fibers["PHI_MIN"].astype(float),
            phiMax=fibers["PHI_MAX"].astype(float),
            thetaCoverage=fibers["THETA_COVERAGE"].astype(float),
            rms=fibers["RMS"].astype(float),
            modelError=fibers["MODEL_ERR"].astype(float),
            runs=[str(rr).strip() for rr in runs["RUN"]],
            runStart=[str(ss).strip() for ss in runs["START"]],
            homeOffsets=fits["HOMEOFFSETS"].data.astype(float),
            alpha={str(aa).strip(): float(vv) for aa, vv in zip(alpha["ARM"], alpha["ALPHA"])},
            alphaSample={str(aa).strip(): str(ss).strip() for aa, ss in zip(alpha["ARM"], alpha["SAMPLE"])},
            metadata=metadata,
        )

    @classmethod
    def readFits(cls, filename: str) -> "PfsThetaPhiModel":
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
        from astropy.io.fits import BinTableHDU, Column, Header, ImageHDU, PrimaryHDU

        header = astropyHeaderFromDict(self.metadata)
        header["DAMD_VER"] = (1, "PfsThetaPhiModel datamodel version")
        header["HIERARCH PFS.HASH.THETAPHIMODEL"] = (self.hash, "Hash of this thetaPhiModel")
        header["MODEL"] = ("doubleFourier", "s = sum_k c_k [b_k(theta, phi) - b_k(home)]")
        header["HIERARCH ANGLE_UNIT"] = ("deg", "Units of theta and phi")
        header["THETA0"] = ("tht0", "theta is measured from the CCW theta hard stop")
        header["PHI0"] = ("phiIn", "phi is measured from the CCW phi hard stop")
        fits.append(PrimaryHDU(header=header))

        inherit = Header()
        inherit["INHERIT"] = True
        fits.append(ImageHDU(self.coeffs.astype(np.float64), name="COEFFS", header=inherit))

        thetaN, thetaF, phiN, phiF = zip(*self.basis)
        fits.append(BinTableHDU.from_columns([
            Column(name="THETA_HARMONIC", format="J", array=np.array(thetaN)),
            Column(name="THETA_FUNC", format="3A", array=np.array(thetaF)),
            Column(name="PHI_HARMONIC", format="J", array=np.array(phiN)),
            Column(name="PHI_FUNC", format="3A", array=np.array(phiF)),
        ], name="BASIS", header=inherit))

        fits.append(BinTableHDU.from_columns([
            Column(name="FIBERID", format="J", array=self.fiberId),
            Column(name="HOME_THETA", format="D", unit="deg", array=self.homeTheta),
            Column(name="HOME_PHI", format="D", unit="deg", array=self.homePhi),
            Column(name="FLAGS", format="J", array=self.flags),
            Column(name="NUM_POINTS", format="J", array=self.numPoints),
            Column(name="PHI_MIN", format="D", unit="deg", array=self.phiMin),
            Column(name="PHI_MAX", format="D", unit="deg", array=self.phiMax),
            Column(name="THETA_COVERAGE", format="D", array=self.thetaCoverage),
            Column(name="RMS", format="D", array=self.rms),
            Column(name="MODEL_ERR", format="D", array=self.modelError),
        ], name="FIBERS", header=inherit))

        width = max([len(rr) for rr in self.runs] + [1])
        fits.append(BinTableHDU.from_columns([
            Column(name="RUN", format=f"{width}A", array=np.array(self.runs, dtype=str)),
            Column(name="START", format="10A", array=np.array([ss[:10] for ss in self.runStart], dtype=str)),
        ], name="RUNS", header=inherit))
        fits.append(ImageHDU(self.homeOffsets.astype(np.float64), name="HOMEOFFSETS", header=inherit))

        arms = sorted(self.alpha)
        width = max([len(self.alphaSample.get(aa, "")) for aa in arms] + [1])
        fits.append(BinTableHDU.from_columns([
            Column(name="ARM", format="1A", array=np.array(arms, dtype=str)),
            Column(name="ALPHA", format="D", array=np.array([self.alpha[aa] for aa in arms])),
            Column(name="SAMPLE", format=f"{width}A",
                   array=np.array([self.alphaSample.get(aa, "") for aa in arms], dtype=str)),
        ], name="ALPHA", header=inherit))

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
