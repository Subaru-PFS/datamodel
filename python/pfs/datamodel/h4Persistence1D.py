"""\
h4Persistence1D.py
==================
Persistence model --- "middle-decay" variant for PFS H4RG infrared detectors.

Physics
-------
When light illuminates the detector during [t0, t0 + texp_quartz], trapped
charge accumulates without decaying until a "switch time"

    t_mid = t0 + f_mid * texp_quartz     (0 <= f_mid <= 1, default 1.0)

After t_mid, all existing trapped charges begin to decay, and newly captured
charges also decay immediately (standard-model behaviour).

For trap component i (fraction f_i, decay time tau_i, delay fraction f_mid_i):

    Let t_delay = f_mid_i * texp_quartz   (no-decay duration)
        t_mid   = t0 + t_delay
        t_end   = t0 + texp_quartz
        dt_2nd  = (1 - f_mid_i) * texp_quartz   (decay-active duration)

    Before illumination  (t < t0):
        Q_i(t) = 0

    Phase 1 --- linear accumulation  (t0 <= t <= t_mid):
        Q_i(t) = f_i * flux * (t - t0)

    Phase 2 --- mixed decay + accumulation  (t_mid < t <= t_end):
        Q_i(t) = f_i * flux * t_delay * exp(-(t - t_mid) / tau_i)
                + f_i * flux * tau_i * (1 - exp(-(t - t_mid) / tau_i))

    After illumination  (t > t_end):
        Q_i(t) = Q_i(t_end) * exp(-(t - t_end) / tau_i)

        where Q_i(t_end) = f_i * flux *
              [t_delay * exp(-dt_2nd / tau_i) + tau_i * (1 - exp(-dt_2nd / tau_i))]

Boundary cases:
    f_mid = 0  ->  standard model (persistence.py): decay starts immediately
    f_mid = 1  ->  delayed model (persistence_delayed.py): linear accumulation only

Persistence during a dark exposure starting at t (t > t_end), duration texp_dark:
    P_i(t) = Q_i(t) - Q_i(t + texp_dark)

All times are in seconds; flux in electrons/s (or any consistent unit).
"""

from collections.abc import Iterable
from dataclasses import dataclass
import io
import textwrap
from typing import TypedDict

import astropy.io.fits
import numpy as np

from .identity import Identity

type FloatArray0D = np.ndarray[tuple[()], np.dtype[np.floating]]
type FloatArray1D = np.ndarray[tuple[int], np.dtype[np.floating]]
type FloatArray2D = np.ndarray[tuple[int, int], np.dtype[np.floating]]
type FloatArray3D = np.ndarray[tuple[int, int, int], np.dtype[np.floating]]

type IntArray1D = np.ndarray[tuple[int], np.dtype[np.integer]]


@dataclass(frozen=True)
class TrapComponent:
    """Single trap component characterised by a trapping fraction, decay time,
    and delay fraction.

    Parameters
    ----------
    fraction : `float`
        Fraction of incoming flux trapped by this component.  Must be in [0, 1].
    tau : `float`
        Exponential decay time constant in seconds.  Must be > 0.
    fMid : `float`
        Fractional time into the exposure at which decay begins.
        0 = decay starts immediately (standard model);
        1 = decay starts at exposure end (delayed model);
        0.5 = decay starts halfway through.
        Must be in [0, 1].
    fractionErr : `float`
        1-sigma error of ``fraction``.
        This parameter is not used effectively for now.
    """

    fraction: float
    tau: float
    fMid: float
    fractionErr: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.fraction <= 1.0):
            raise ValueError(f"fraction must be in [0, 1], got {self.fraction}")
        if self.tau <= 0:
            raise ValueError(f"tau must be > 0, got {self.tau}")
        if not (0.0 <= self.fMid <= 1.0):
            raise ValueError(f"fMid must be in [0, 1], got {self.fMid}")

    @property
    def label(self) -> str:
        """Brief description of this component intended for a label in a plot."""
        return f"tau={self.tau:.1f}s, fMid={self.fMid:.1f}"

    def charge(
        self,
        t: FloatArray1D,
        t0: float,
        tExpQuartz: float,
        flux: float,
    ) -> FloatArray1D:
        """Compute trapped charge at every time in *t*.

        Parameters
        ----------
        t : `np.ndarray`, shape (N,)
            Evaluation times [s].
        t0 : `float`
            Illumination start time [s].
        tExpQuartz : `float`
            Illumination duration [s].
        flux : `float`
            Constant photon flux during illumination [electrons/s].

        Returns
        -------
        `np.ndarray`, shape (N,)
            Trapped charge [electrons].
        """
        tDelay = self.fMid * tExpQuartz  # duration with no decay
        tMid = t0 + tDelay
        tEnd = t0 + tExpQuartz
        dt2nd = tExpQuartz - tDelay  # = (1 - fMid) * tExpQuartz

        result = np.zeros_like(t, dtype=float)

        # Phase 1: linear accumulation, no decay
        mask1 = (t >= t0) & (t <= tMid)
        if np.any(mask1):
            result[mask1] = self.fraction * flux * (t[mask1] - t0)

        # Phase 2: accumulated charge decays + new charge decays immediately
        mask2 = (t > tMid) & (t <= tEnd)
        if np.any(mask2):
            dt2 = t[mask2] - tMid
            exp2 = np.exp(-dt2 / self.tau)
            q0 = self.fraction * flux * tDelay  # charge at tMid
            result[mask2] = q0 * exp2 + self.fraction * flux * self.tau * (1.0 - exp2)

        # After illumination: Q(tEnd) * exp decay
        maskAfter = t > tEnd
        if np.any(maskAfter):
            exp2nd = np.exp(-dt2nd / self.tau)
            q0 = self.fraction * flux * tDelay
            qAtEnd = q0 * exp2nd + self.fraction * flux * self.tau * (1.0 - exp2nd)
            result[maskAfter] = qAtEnd * np.exp(-(t[maskAfter] - tEnd) / self.tau)

        return result


class SpatialProfile:
    """Spatial profile of persistence.

    Parameters
    ----------
    fiberId : `IntArray1D`
        fiber ID.
    profile : `np.ndarray` of shape (nFibers, nWavelen, nComponents).
        Spatial profile. The last dimension ``nComponents`` is optional.
        If the last dimension does not exist, the array is broadcast.
    """

    def __init__(self, fiberId: IntArray1D, profile: FloatArray2D | FloatArray3D) -> None:
        self.fiberId = np.asarray(fiberId, dtype=np.int32)
        self.profile = np.asarray(profile, dtype=float)
        if self.profile.ndim == 2:
            self.profile = self.profile.reshape((*self.profile.shape, 1))

        if self.fiberId.shape != self.profile.shape[:-2]:
            raise ValueError(
                f"# of fibers {self.fiberId.shape} and # of profiles {self.profile.shape[:-2]} differ"
            )

    def select(
        self,
        fiberId: np.ndarray,
    ) -> np.ndarray:
        """Get spatial profiles with the first dimension arranged
        in the same order as ``fiberId``

        Parameters
        ----------
        fiberId : `IntArray1D`
            fiber ID.

        Returns
        -------
        profile : `np.ndarray` of shape (nFibers, nWavelen, nComponents).
            Spatial profile. The last dimension ``nComponents`` may be 1.
        """

        index = {fid: i for i, fid in enumerate(self.fiberId)}

        try:
            indices = np.array([index[fid] for fid in fiberId], dtype=int)
        except KeyError as exc:
            raise RuntimeError(f"No persistence shape for fiberId={exc.args[0]}") from exc

        spatialProfile = self.profile[indices, ...]  # shape (nFibers, nWavelen, nComponents)

        # This reproduces the notebook behaviour.
        return np.clip(spatialProfile, 0.0, None)


class H4Persistence1DBasicModel:
    """H4RG detector persistence model --- middle-decay variant.

    Trapped charges accumulate linearly during the first ``fMid`` fraction
    of the exposure.  After that switch time, existing trapped charges begin
    exponential decay, and newly captured charges also decay immediately.

    Parameters
    ----------
    components : `list` [`TrapComponent`]
        Trap components in any order.  Each component carries its own ``fMid``.
    spatialProfile : `SpatialProfile`
        Spatial profile (as opposed to temporal profile determined by
        ``components``) of a persistence model.
    name : `str`, optional
        H4Persistence1DBasicModel identifier used in plot titles and summary output.
    """

    def __init__(
        self,
        components: Iterable[TrapComponent],
        spatialProfile: SpatialProfile,
        name: str = "H4RG",
    ) -> None:
        if not components:
            raise ValueError("components must not be empty")
        self._components: list[TrapComponent] = list(components)
        self.spatialProfile = spatialProfile
        self.name = name

    @classmethod
    def fromFits(cls, fits: astropy.io.fits.HDUList) -> "H4Persistence1DBasicModel":
        """Construct from a FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            The FITS file.

        Returns
        -------
        instance : `H4Persistence1DBasicModel`
            Constructed instance.
        """
        name = fits[0].header.get("MODELNAM", "H4")

        components = [
            TrapComponent(
                fraction=float(row["fraction"]),
                tau=float(row["tau"]),
                fMid=float(row["f_mid"]),
                fractionErr=float(row["fraction_error"]),
            )
            for row in fits["TEMPORALPROF"].data
        ]

        spatialProfile = SpatialProfile(
            fiberId=fits["FIBERID"].data,
            profile=fits["SPATIALPROF"].data,
        )

        return cls(
            components,
            spatialProfile=spatialProfile,
            name=name,
        )

    def toFits(self) -> astropy.io.fits.HDUList:
        """Convert self to a FITS file

        Returns
        -------
        fits : `astropy.io.fits.HDUList`
            The FITS file.
        """

        fitsDescription = """
        This file contains parameters of H4RG persistence curves.
        """

        class ColumnDef(TypedDict):
            """
            Keys
            ----
            name : `str`
                Name of a column in a FITS BinTable.
            type : `tuple` [`type`] | `tuple` [`type`, `int`]
                Column type. Array size may optionally follow.
            unit : `str`
                Physical unit.
            propertyName : `str`
                Property name in the parent object.
            doc : `str`
                Documentation.
            """

            name: str
            type: tuple[type] | tuple[type, int]
            unit: str
            propertyName: str
            doc: str

        columns: list[ColumnDef] = [
            {
                "name": "tau",
                "type": (float,),
                "unit": "s",
                "propertyName": "taus",
                "doc": "Lifetime of exponential decay",
            },
            {
                "name": "f_mid",
                "type": (float,),
                "unit": "",
                "propertyName": "fMids",
                "doc": "Fractional time into exp. when decay begins.",
            },
            {
                "name": "fraction",
                "type": (float,),
                "unit": "",
                "propertyName": "fractions",
                "doc": "Fract. of incoming flux trapped by this comp.",
            },
            {
                "name": "fraction_error",
                "type": (float,),
                "unit": "",
                "propertyName": "fractionErrs",
                "doc": "Error of fraction.",
            },
        ]

        comments = textwrap.dedent(fitsDescription).strip("\n").split("\n")
        columnToDef = {columndef["name"]: columndef for columndef in columns}

        # Primary HDU (nothing)
        hdu = astropy.io.fits.PrimaryHDU()
        header = hdu.header
        for line in comments:
            header.add_comment(line)
        header["MODELNAM"] = self.name
        hdulist = astropy.io.fits.HDUList([hdu])

        # Temporal profile
        table = np.empty(
            shape=(self.nComponents,),
            dtype=[(columndef["name"], *columndef["type"]) for columndef in columns],
        )
        for columndef in columns:
            table[columndef["name"]] = getattr(self, columndef["propertyName"])

        hdu = astropy.io.fits.BinTableHDU(name="TEMPORALPROF", data=table)
        header = hdu.header
        for i in range(len(hdu.data.columns)):
            typekey = f"TTYPE{i + 1}"
            columndef = columnToDef[header[typekey]]
            header.set(typekey, comment=columndef["doc"])
            unitkey = f"TUNIT{i + 1}"
            header.set(unitkey, value=columndef["unit"], after=typekey)

        hdulist.append(hdu)

        # Spatial profile
        hdu = astropy.io.fits.ImageHDU(
            name="SPATIALPROF",
            data=self.spatialProfile.profile.astype(np.float32),
        )
        hdulist.append(hdu)

        hdu = astropy.io.fits.ImageHDU(
            name="FIBERID",
            data=self.spatialProfile.fiberId.astype(np.int32),
        )
        hdulist.append(hdu)

        return hdulist

    def writeFits(self, path: str) -> None:
        """Write self to a FITS file

        Parameters
        ----------
        path : `str`
            Path to the output FITS file.
        """
        temp = io.BytesIO()
        self.toFits().writeto(temp, checksum=True)
        with open(path, "wb") as f:
            f.write(temp.getvalue())

    @classmethod
    def readFits(cls, path: str) -> "H4Persistence1DBasicModel":
        """Read from a FITS file

        Parameters
        ----------
        path : `str`
            Path to the FITS file.

        Returns
        -------
        instance : `H4Persistence1DBasicModel`
            Constructed instance.
        """
        with astropy.io.fits.open(path) as fits:
            return cls.fromFits(fits)

    @property
    def components(self) -> list[TrapComponent]:
        """Read-only list of trap components."""
        return list(self._components)

    @property
    def nComponents(self) -> int:
        """Number of trap components."""
        return len(self._components)

    @property
    def totalFraction(self) -> float:
        """Sum of all trapping fractions."""
        return sum(c.fraction for c in self._components)

    @property
    def taus(self) -> FloatArray1D:
        """Array of decay time constants [s]."""
        return np.array([c.tau for c in self._components])

    @property
    def fractions(self) -> FloatArray1D:
        """Array of trapping fractions."""
        return np.array([c.fraction for c in self._components])

    @property
    def fMids(self) -> FloatArray1D:
        """Array of delay fractions (one per component)."""
        return np.array([c.fMid for c in self._components])

    @property
    def fractionErrs(self) -> FloatArray1D:
        """Array of 1-sigma errors of trapping fractions"""
        return np.array([c.fractionErr for c in self._components])

    def __repr__(self) -> str:
        components = ", ".join(f"(f={c.fraction}, tau={c.tau}s, fMid={c.fMid})" for c in self._components)
        return f"H4Persistence1DBasicModel(name={self.name!r}, components=[{components}])"

    def __len__(self) -> int:
        return self.nComponents


@dataclass
class H4Persistence1D:
    """Persistent electrons, released during an exposure."""

    fiberId: np.ndarray[tuple[int], np.dtype[np.int32]]
    wavelength: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    flux: np.ndarray[tuple[int, int], np.dtype[np.float32]]
    identity: Identity

    @classmethod
    def fromFits(cls, fits: astropy.io.fits.HDUList) -> "H4Persistence1D":
        """Construct from a FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            The FITS file.

        Returns
        -------
        instance : `H4Persistence1D`
            Constructed instance.
        """
        fiberId = np.array(fits["FIBERID"].data, dtype=np.int32)
        wavelength = np.array(fits["WAVELENGTH"].data, dtype=np.float64)
        flux = np.array(fits["FLUX"].data, dtype=np.float32)
        identity = Identity.fromFits(fits)

        return cls(
            fiberId=fiberId,
            wavelength=wavelength,
            flux=flux,
            identity=identity,
        )

    def toFits(self) -> astropy.io.fits.HDUList:
        """Convert self to a FITS file

        Returns
        -------
        fits : `astropy.io.fits.HDUList`
            The FITS file.
        """
        hdulist = astropy.io.fits.HDUList(
            [
                astropy.io.fits.PrimaryHDU(),
                astropy.io.fits.ImageHDU(name="FIBERID", data=self.fiberId.astype(np.int32)),
                astropy.io.fits.ImageHDU(name="WAVELENGTH", data=self.wavelength.astype(np.float64)),
                astropy.io.fits.ImageHDU(name="FLUX", data=self.flux.astype(np.float32)),
            ],
        )
        self.identity.toFits(hdulist)
        return hdulist

    def writeFits(self, path: str) -> None:
        """Write self to a FITS file

        Parameters
        ----------
        path : `str`
            Path to the output FITS file.
        """
        temp = io.BytesIO()
        self.toFits().writeto(temp, checksum=True)
        with open(path, "wb") as f:
            f.write(temp.getvalue())

    @classmethod
    def readFits(cls, path: str) -> "H4Persistence1D":
        """Read from a FITS file

        Parameters
        ----------
        path : `str`
            Path to the FITS file.

        Returns
        -------
        instance : `H4Persistence1D`
            Constructed instance.
        """
        with astropy.io.fits.open(path) as fits:
            return cls.fromFits(fits)

    def select(self, fiberId: IntArray1D) -> "H4Persistence1D":
        """Get a subset of this object containing only ``fiberId``

        Parameters
        ----------
        fiberId : `IntArray1D`
            fiber ID.

        Returns
        -------
        profile : `H4Persistence1D`
            New `H4Persistence1D`.
        """
        index = {int(fid): i for i, fid in enumerate(self.fiberId)}

        try:
            indices = np.array([index[int(fid)] for fid in fiberId], dtype=int)
        except KeyError as exc:
            raise RuntimeError(f"No persistence for fiberId={exc.args[0]}") from exc

        return type(self)(
            fiberId=fiberId.astype(np.int32),
            wavelength=self.wavelength[indices, ...],
            flux=self.flux[indices, ...],
            identity=self.identity,
        )
