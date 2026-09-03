"""
h4Persistence.py
==============
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
    f_mid : `float`
        Fractional time into the exposure at which decay begins.
        0 = decay starts immediately (standard model);
        1 = decay starts at exposure end (delayed model);
        0.5 = decay starts halfway through.
        Must be in [0, 1].
    fraction_error : `float`
        1-sigma error of ``fraction``.
        This parameter is not used effectively for now.
    """

    fraction: float
    tau: float
    f_mid: float
    fraction_error: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.fraction <= 1.0):
            raise ValueError(f"fraction must be in [0, 1], got {self.fraction}")
        if self.tau <= 0:
            raise ValueError(f"tau must be > 0, got {self.tau}")
        if not (0.0 <= self.f_mid <= 1.0):
            raise ValueError(f"f_mid must be in [0, 1], got {self.f_mid}")

    @property
    def label(self) -> str:
        """Brief description of this component intended for a label in a plot."""
        return f"tau={self.tau:.1f}s, f_mid={self.f_mid:.1f}"

    def charge(
        self,
        t: FloatArray1D,
        t0: float,
        texp_quartz: float,
        flux: float,
    ) -> FloatArray1D:
        """Compute trapped charge at every time in *t*.

        Parameters
        ----------
        t : `np.ndarray`, shape (N,)
            Evaluation times [s].
        t0 : `float`
            Illumination start time [s].
        texp_quartz : `float`
            Illumination duration [s].
        flux : `float`
            Constant photon flux during illumination [electrons/s].

        Returns
        -------
        `np.ndarray`, shape (N,)
            Trapped charge [electrons].
        """
        t_delay = self.f_mid * texp_quartz  # duration with no decay
        t_mid = t0 + t_delay
        t_end = t0 + texp_quartz
        dt_2nd = texp_quartz - t_delay  # = (1 - f_mid) * texp_quartz

        result = np.zeros_like(t, dtype=float)

        # Phase 1: linear accumulation, no decay
        mask1 = (t >= t0) & (t <= t_mid)
        if np.any(mask1):
            result[mask1] = self.fraction * flux * (t[mask1] - t0)

        # Phase 2: accumulated charge decays + new charge decays immediately
        mask2 = (t > t_mid) & (t <= t_end)
        if np.any(mask2):
            dt2 = t[mask2] - t_mid
            exp2 = np.exp(-dt2 / self.tau)
            q0 = self.fraction * flux * t_delay  # charge at t_mid
            result[mask2] = q0 * exp2 + self.fraction * flux * self.tau * (1.0 - exp2)

        # After illumination: Q(t_end) * exp decay
        mask_after = t > t_end
        if np.any(mask_after):
            exp_2nd = np.exp(-dt_2nd / self.tau)
            q0 = self.fraction * flux * t_delay
            q_at_end = q0 * exp_2nd + self.fraction * flux * self.tau * (1.0 - exp_2nd)
            result[mask_after] = q_at_end * np.exp(-(t[mask_after] - t_end) / self.tau)

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


class H4PersistenceBasicModel:
    """H4RG detector persistence model --- middle-decay variant.

    Trapped charges accumulate linearly during the first ``f_mid`` fraction
    of the exposure.  After that switch time, existing trapped charges begin
    exponential decay, and newly captured charges also decay immediately.

    Parameters
    ----------
    components : `list` [`TrapComponent`]
        Trap components in any order.  Each component carries its own ``f_mid``.
    spatialProfile : `SpatialProfile`
        Spatial profile (as opposed to temporal profile determined by
        ``components``) of a persistence model.
    name : `str`, optional
        H4PersistenceBasicModel identifier used in plot titles and summary output.
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
    def fromFits(cls, fits: astropy.io.fits.HDUList) -> "H4PersistenceBasicModel":
        """Construct from a FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            The FITS file.

        Returns
        -------
        instance : `H4PersistenceBasicModel`
            Constructed instance.
        """
        name = fits[0].header.get("MODELNAM", "H4")

        components = [
            TrapComponent(
                fraction=float(row["fraction"]),
                tau=float(row["tau"]),
                f_mid=float(row["f_mid"]),
                fraction_error=float(row["fraction_error"]),
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

        fits_description = """
        This file holds parameters of an H4RG persistence curve.

            *TODO: Write detailed descriptions.*

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
            property_name : `str`
                Property name in the parent object.
            doc : `str`
                Documentation.
            """

            name: str
            type: tuple[type] | tuple[type, int]
            unit: str
            property_name: str
            doc: str

        columns: list[ColumnDef] = [
            {
                "name": "tau",
                "type": (float,),
                "unit": "s",
                "property_name": "taus",
                "doc": "Lifetime of exponential decay",
            },
            {
                "name": "f_mid",
                "type": (float,),
                "unit": "",
                "property_name": "f_mids",
                "doc": "Fractional time into exp. when decay begins.",
            },
            {
                "name": "fraction",
                "type": (float,),
                "unit": "",
                "property_name": "fractions",
                "doc": "Fract. of incoming flux trapped by this comp.",
            },
            {
                "name": "fraction_error",
                "type": (float,),
                "unit": "",
                "property_name": "fraction_errors",
                "doc": "Error of fraction.",
            },
        ]

        comments = textwrap.dedent(fits_description).strip("\n").split("\n")
        column_to_def = {columndef["name"]: columndef for columndef in columns}

        # Primary HDU (nothing)
        hdu = astropy.io.fits.PrimaryHDU()
        header = hdu.header
        for line in comments:
            header.add_comment(line)
        header["MODELNAM"] = self.name
        hdulist = astropy.io.fits.HDUList([hdu])

        # Temporal profile
        table = np.empty(
            shape=(self.n_components,),
            dtype=[(columndef["name"], *columndef["type"]) for columndef in columns],
        )
        for columndef in columns:
            table[columndef["name"]] = getattr(self, columndef["property_name"])

        hdu = astropy.io.fits.BinTableHDU(name="TEMPORALPROF", data=table)
        header = hdu.header
        for i in range(len(hdu.data.columns)):
            typekey = f"TTYPE{i + 1}"
            columndef = column_to_def[header[typekey]]
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
    def readFits(cls, path: str) -> "H4PersistenceBasicModel":
        """Read from a FITS file

        Parameters
        ----------
        path : `str`
            Path to the FITS file.

        Returns
        -------
        instance : `H4PersistenceBasicModel`
            Constructed instance.
        """
        with astropy.io.fits.open(path) as fits:
            return cls.fromFits(fits)

    @property
    def components(self) -> list[TrapComponent]:
        """Read-only list of trap components."""
        return list(self._components)

    @property
    def n_components(self) -> int:
        """Number of trap components."""
        return len(self._components)

    @property
    def total_fraction(self) -> float:
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
    def f_mids(self) -> FloatArray1D:
        """Array of delay fractions (one per component)."""
        return np.array([c.f_mid for c in self._components])

    @property
    def fraction_errors(self) -> FloatArray1D:
        """Array of 1-sigma errors of trapping fractions"""
        return np.array([c.fraction_error for c in self._components])

    def __repr__(self) -> str:
        comp_str = ", ".join(f"(f={c.fraction}, tau={c.tau}s, f_mid={c.f_mid})" for c in self._components)
        return f"H4PersistenceBasicModel(name={self.name!r}, components=[{comp_str}])"

    def __len__(self) -> int:
        return self.n_components


@dataclass
class H4Persistence:
    """Persistent electrons, released during an exposure."""

    fiberId: np.ndarray[tuple[int], np.dtype[np.int32]]
    wavelength: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    flux: np.ndarray[tuple[int, int], np.dtype[np.float32]]
    identity: Identity

    @classmethod
    def fromFits(cls, fits: astropy.io.fits.HDUList) -> "H4Persistence":
        """Construct from a FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            The FITS file.

        Returns
        -------
        instance : `H4Persistence`
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
    def readFits(cls, path: str) -> "H4Persistence":
        """Read from a FITS file

        Parameters
        ----------
        path : `str`
            Path to the FITS file.

        Returns
        -------
        instance : `H4Persistence`
            Constructed instance.
        """
        with astropy.io.fits.open(path) as fits:
            return cls.fromFits(fits)

    def select(self, fiberId: IntArray1D) -> "H4Persistence":
        """Get a subset of this object containing only ``fiberId``

        Parameters
        ----------
        fiberId : `IntArray1D`
            fiber ID.

        Returns
        -------
        profile : `H4Persistence`
            New `H4Persistence`.
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
