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

Class hierarchy
---------------
    TrapComponent   -- immutable value object: (fraction, tau, f_mid, label)
    H4PersistenceModel        -- owns a list of TrapComponent objects and provides
                       signal(), signal_by_component(), persistence(),
                       persistence_by_component(), fit_fractions(), step(),
                       plot(), summary()

Usage example
-------------
    import numpy as np
    from persistence import TrapComponent, H4PersistenceModel

    det = H4PersistenceModel.from_arrays(
        fractions=[0.020, 0.010, 0.005],
        taus=[10.0, 100.0, 1000.0],
        f_mid=0.5,
    )
    t = np.logspace(1, 5, 2000)
    Q = det.signal(t, t0=10.0, texp_quartz=100.0, flux=1e5)
    fig = det.plot(10.0, 100.0, 1e5, show=True)
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import io
import textwrap
import typing
from typing import TypedDict

import astropy.io.fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from .identity import Identity

type FloatArray0D = np.ndarray[tuple[()], np.dtype[np.floating]]
type FloatArray1D = np.ndarray[tuple[int], np.dtype[np.floating]]
type FloatArray2D = np.ndarray[tuple[int, int], np.dtype[np.floating]]
type FloatArray3D = np.ndarray[tuple[int, int, int], np.dtype[np.floating]]

type IntArray1D = np.ndarray[tuple[int], np.dtype[np.integer]]


def _bounded_lsq_pgd(
    A: np.ndarray,
    b: np.ndarray,
    lo: float = 0.0,
    hi: float = 1.0,
    max_iter: int = 20_000,
    tol: float = 1e-12,
) -> np.ndarray:
    """Solve  min ||Ax - b||^2  s.t.  lo <= x_i <= hi  via FISTA.

    Uses the Fast Iterative Shrinkage-Thresholding Algorithm (Beck & Teboulle
    2009) with box-projection, which converges at O(1/k^2) rate.
    """
    AtA = A.T @ A
    Atb = A.T @ b
    L = float(np.linalg.eigvalsh(AtA).max())
    if L == 0.0:
        return np.zeros(A.shape[1])
    alpha = 1.0 / L

    n = A.shape[1]
    x = np.clip(np.zeros(n), lo, hi)
    y = x.copy()
    t = 1.0

    for _ in range(max_iter):
        x_prev = x.copy()
        grad = AtA @ y - Atb
        x = np.clip(y - alpha * grad, lo, hi)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = x + ((t - 1.0) / t_new) * (x - x_prev)
        t = t_new
        if np.linalg.norm(x - x_prev) < tol:
            break

    return x


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
        If the last dimension does not exist, the array is broadcasted.
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


class H4PersistenceModel:
    """H4RG detector persistence model --- middle-decay variant.

    Trapped charges accumulate linearly during the first ``f_mid`` fraction
    of the exposure.  After that switch time, existing trapped charges begin
    exponential decay, and newly captured charges also decay immediately.

    Parameters
    ----------
    components : `list` [`TrapComponent`]
        Trap components in any order.  Each component carries its own ``f_mid``.
    spatialProfile : `SpatialProfile`
    name : `str`, optional
        H4PersistenceModel identifier used in plot titles and summary output.
    """

    def __init__(
        self,
        components: Iterable[TrapComponent],
        spatialProfile: np.ndarray,
        name: str = "H4RG",
    ) -> None:
        if not components:
            raise ValueError("components must not be empty")
        self._components: list[TrapComponent] = list(components)
        self.spatialProfile = spatialProfile
        self.name = name

    @classmethod
    def fromFits(cls, fits: astropy.io.fits.HDUList) -> "H4PersistenceModel":
        """Construct from a FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            The FITS file.

        Returns
        -------
        instance : `H4PersistenceModel`
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
    def readFits(cls, path: str) -> "H4PersistenceModel":
        """Read from a FITS file

        Parameters
        ----------
        path : `str`
            Path to the FITS file.

        Returns
        -------
        instance : `H4PersistenceModel`
            Constructed instance.
        """
        with astropy.io.fits.open(path) as fits:
            return cls.fromFits(fits)

    def update_fractions(
        self,
        fractions: Sequence[float],
        name: str | None = None,
    ) -> "H4PersistenceModel":
        """Return a new H4PersistenceModel with updated fractions; taus, f_mids, and labels unchanged.

        Parameters
        ----------
        fractions : sequence of float
            New fractions, one per component.
        name : str or None, optional
            Name for the new H4PersistenceModel.  Defaults to the current name.

        Returns
        -------
        H4PersistenceModel
        """
        if len(fractions) != self.n_components:
            raise ValueError(f"Expected {self.n_components} fractions, got {len(fractions)}")
        new_comps = [
            TrapComponent(fraction=float(f), tau=c.tau, f_mid=c.f_mid, label=c.label)
            for f, c in zip(fractions, self._components)
        ]
        return H4PersistenceModel(
            new_comps,
            name=name if name is not None else self.name,
            spatialProfile=self.spatialProfile,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Core calculations
    # ------------------------------------------------------------------

    def step(
        self,
        q_state: np.ndarray,
        flux_rate: np.ndarray,
        texp: float,
        dt_gap: float = 0.0,
        new_shape: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Propagate trap state through one exposure and an optional trailing gap.

        *q_state* holds charges from the previous exposure that begin decaying
        at the start of this exposure.  New charges from this exposure start
        decaying at ``t_mid = f_mid * texp`` into the exposure.

        Parameters
        ----------
        q_state : np.ndarray, shape (..., K)
            Trapped charge that has been decaying since the start of this
            exposure (from the previous exposure).
        flux_rate : np.ndarray, shape (...)
            Photon flux rate during the exposure [e-/s].
        texp : float
            Exposure duration [s].
        dt_gap : float
            Time between the end of this exposure and the next evaluation
            point [s].
        new_shape : np.ndarray, optional
            Multiplicative spatial profile applied **only** to the charge newly
            trapped during this exposure, before it is added to the carried-over
            (decayed) state.  Must broadcast against the newly-trapped charge of
            shape ``(..., K)`` (e.g. ``(..., 1)`` to apply the same profile to
            every component).  Because the shape is baked into each charge packet
            exactly once at trapping time and is *not* re-applied to carried-over
            charge, it does not compound across successive :meth:`step` calls.
            If None (default), no shaping is applied.

        Returns
        -------
        q_end : ndarray, shape (..., K)
            Total trapped charge immediately after the exposure ends.
        q_next : ndarray, shape (..., K)
            State to pass as *q_state* to the next :meth:`step` call.
        persistence_released : ndarray, shape (...)
            Total charge released from traps during this exposure
            (from q_state only; new charges start decaying from t_mid).
        """
        t_delays = self.f_mids * texp  # (K,) no-decay duration per component
        # (K,) decay-active duration per component
        dt_2nds = texp - t_delays

        decay_full = np.exp(-texp / self.taus)  # (K,)
        decay_2nd = np.exp(-dt_2nds / self.taus)  # (K,)

        # Per-component coefficient for the charge newly trapped this exposure:
        #   q_new = flux_rate * fraction * (t_delay*decay_2nd + tau*(1-decay_2nd))
        # combining the old Phase 1 + Phase 2 terms.  This is a tiny (K,) array,
        # so the expensive broadcast over (..., K) happens only once below.
        coeff_new = self.fractions * (t_delays * decay_2nd + self.taus * (1.0 - decay_2nd))

        # Match the input dtype (typically float32) so the large (..., K)
        # arrays below are not silently promoted to float64.
        dtype = q_state.dtype
        decay_full = decay_full.astype(dtype, copy=False)
        coeff_new = coeff_new.astype(dtype, copy=False)

        # Charge released from carried-over traps: a matrix-vector product over
        # the component axis avoids materialising a (..., K) temporary.
        persistence_released = q_state @ (1.0 - decay_full)

        # Total trapped charge after the exposure: new charge + decayed old charge.
        # The spatial profile is applied to the newly-trapped charge only, so it
        # is imprinted on each charge packet once and never re-applied to the
        # carried-over (already-shaped) state --- avoiding runaway compounding.
        q_end = flux_rate[..., np.newaxis] * coeff_new
        if new_shape is not None:
            q_end *= np.asarray(new_shape, dtype=dtype)
        q_end += q_state * decay_full

        if dt_gap > 0:
            decay_gap = np.exp(-dt_gap / self.taus).astype(dtype, copy=False)
            q_next = q_end * decay_gap
        else:
            q_next = q_end.copy()

        return q_end, q_next, persistence_released

    @typing.overload
    def signal(
        self,
        t: float | FloatArray1D,
        t0: float,
        texp_quartz: float,
        flux: float,
    ) -> FloatArray0D:
        ...

    @typing.overload
    def signal(
        self,
        t: float | FloatArray1D,
        t0: FloatArray1D,
        texp_quartz: FloatArray1D,
        flux: FloatArray1D,
    ) -> FloatArray1D:
        ...

    def signal(
        self,
        t,
        t0,
        texp_quartz,
        flux,
    ):
        """Total trapped charge at time(s) *t*.

        Parameters
        ----------
        t : float or array-like
            Evaluation time(s) [s].
        t0 : float or sequence of float
            Illumination start time(s) [s].
        texp_quartz : float or sequence of float
            Illumination duration(s) [s].
        flux : float or sequence of float
            Photon flux during each illumination [electrons/s].

        Returns
        -------
        np.ndarray
            Total trapped charge [electrons].  Scalar when *t* is scalar.
        """
        illums = self._to_illuminations(t0, texp_quartz, flux)
        self._validate_illuminations(illums)
        input_shape = np.broadcast(t).shape
        t_arr = self._to_array(t)
        total = np.zeros_like(t_arr)
        for comp in self._components:
            for t0_k, texp_k, flux_k in illums:
                total += comp.charge(t_arr, t0_k, texp_k, flux_k)
        return total.reshape(input_shape)

    @typing.overload
    def signal_by_component(
        self,
        t: float | FloatArray1D,
        t0: float,
        texp_quartz: float,
        flux: float,
    ) -> dict[str, np.ndarray]:
        ...

    @typing.overload
    def signal_by_component(
        self,
        t: float | FloatArray1D,
        t0: FloatArray1D,
        texp_quartz: FloatArray1D,
        flux: FloatArray1D,
    ) -> dict[str, np.ndarray]:
        ...

    def signal_by_component(
        self,
        t,
        t0,
        texp_quartz,
        flux,
    ) -> dict[str, np.ndarray]:
        """Trapped charge broken down by component.

        Parameters
        ----------
        t, t0, texp_quartz, flux : see :meth:`signal`.

        Returns
        -------
        dict[str, np.ndarray]
            Keys are component labels plus ``"total"``.
        """
        illums = self._to_illuminations(t0, texp_quartz, flux)
        self._validate_illuminations(illums)
        t_arr = self._to_array(t)
        result: dict[str, np.ndarray] = {}
        total = np.zeros_like(t_arr)
        for comp in self._components:
            q = np.zeros_like(t_arr)
            for t0_k, texp_k, flux_k in illums:
                q += comp.charge(t_arr, t0_k, texp_k, flux_k)
            result[comp.label] = q
            total += q
        result["total"] = total
        return result

    @typing.overload
    def persistence(
        self,
        t: float | FloatArray1D,
        t0: float,
        texp_quartz: float,
        flux: float,
        texp_dark: float = ...,
    ) -> np.ndarray:
        ...

    @typing.overload
    def persistence(
        self,
        t: float | FloatArray1D,
        t0: FloatArray1D,
        texp_quartz: FloatArray1D,
        flux: FloatArray1D,
        texp_dark: float = ...,
    ) -> np.ndarray:
        ...

    def persistence(self, t, t0, texp_quartz, flux, texp_dark: float = 300.0):
        """Charge released during a dark exposure starting at time *t*.

        Persistence is defined as the total trapped charge that decays
        during the dark exposure window [t, t + texp_dark]:

            P(t) = Q(t) - Q(t + texp_dark)

        Parameters
        ----------
        t : float or array-like
            Start time(s) of the dark exposure [s].
        t0 : float or sequence of float
            Illumination start time(s) [s].
        texp_quartz : float or sequence of float
            Illumination duration(s) [s].
        flux : float or sequence of float
            Photon flux during each illumination [electrons/s].
        texp_dark : float, optional
            Dark exposure duration [s].  Default: 300.

        Returns
        -------
        np.ndarray
            Persistence signal [electrons] at each time in *t*.
        """
        if texp_dark <= 0:
            raise ValueError(f"texp_dark must be > 0, got {texp_dark}")
        illums = self._to_illuminations(t0, texp_quartz, flux)
        self._validate_illuminations(illums)
        t_arr = self._to_array(t)
        result = np.zeros_like(t_arr)
        for comp in self._components:
            for t0_k, texp_k, flux_k in illums:
                mask = t_arr > t0_k + texp_k
                if not np.any(mask):
                    continue
                result[mask] += comp.charge(t_arr[mask], t0_k, texp_k, flux_k) - comp.charge(
                    t_arr[mask] + texp_dark, t0_k, texp_k, flux_k
                )
        np.maximum(result, 0.0, out=result)
        return result

    @typing.overload
    def persistence_by_component(
        self,
        t: float | FloatArray1D,
        t0: float,
        texp_quartz: float,
        flux: float,
        texp_dark: float = ...,
    ) -> dict[str, np.ndarray]:
        ...

    @typing.overload
    def persistence_by_component(
        self,
        t: float | FloatArray1D,
        t0: FloatArray1D,
        texp_quartz: FloatArray1D,
        flux: FloatArray1D,
        texp_dark: float = ...,
    ) -> dict[str, np.ndarray]:
        ...

    def persistence_by_component(self, t, t0, texp_quartz, flux, texp_dark: float = 300.0):
        """Persistence broken down by component.

        Parameters
        ----------
        t, t0, texp_quartz, flux, texp_dark : see :meth:`persistence`.

        Returns
        -------
        dict[str, np.ndarray]
            Keys are component labels plus ``"total"``.
        """
        if texp_dark <= 0:
            raise ValueError(f"texp_dark must be > 0, got {texp_dark}")
        illums = self._to_illuminations(t0, texp_quartz, flux)
        self._validate_illuminations(illums)
        t_arr = self._to_array(t)
        result: dict[str, np.ndarray] = {}
        total = np.zeros_like(t_arr)
        for comp in self._components:
            q = np.zeros_like(t_arr)
            for t0_k, texp_k, flux_k in illums:
                mask = t_arr > t0_k + texp_k
                if not np.any(mask):
                    continue
                q[mask] += comp.charge(t_arr[mask], t0_k, texp_k, flux_k) - comp.charge(
                    t_arr[mask] + texp_dark, t0_k, texp_k, flux_k
                )
            np.maximum(q, 0.0, out=q)
            result[comp.label] = q
            total += q
        result["total"] = total
        return result

    @typing.overload
    def fit_fractions(
        self,
        t_obs: np.ndarray,
        p_obs: np.ndarray,
        t0: float,
        texp_quartz: float,
        flux: float,
        texp_dark: float = ...,
        dp_obs: np.ndarray | None = ...,
        fixed_indices: Sequence[int] | None = ...,
    ) -> tuple["H4PersistenceModel", dict]:
        ...

    @typing.overload
    def fit_fractions(
        self,
        t_obs: np.ndarray,
        p_obs: np.ndarray,
        t0: FloatArray1D,
        texp_quartz: FloatArray1D,
        flux: FloatArray1D,
        texp_dark: float = ...,
        dp_obs: np.ndarray | None = ...,
        fixed_indices: Sequence[int] | None = ...,
    ) -> tuple["H4PersistenceModel", dict]:
        ...

    def fit_fractions(
        self,
        t_obs,
        p_obs,
        t0,
        texp_quartz,
        flux,
        texp_dark: float = 300.0,
        dp_obs: np.ndarray | None = None,
        fixed_indices: Sequence[int] | None = None,
    ):
        r"""Fit trap fractions to observed persistence data.

        The model persistence is linear in fractions, so the problem reduces
        to bounded linear least squares:

            min  ||A @ f - p_obs||^2    subject to   0 <= f_i <= 1

        Normalization convention
        ------------------------
        p_obs must be normalized by (\sum_k flux_k * texp_k) * texp_dark:

            p_obs = P_actual [e^-] / (\sum_k flux_k \cdot texp_k [e^-] * texp_dark [s])

        Parameters
        ----------
        t_obs : array-like
            Observation times [s].  Should be > t0 + texp_quartz.
        p_obs : array-like
            Observed persistence normalized by (\sum flux_k \cdot texp_k) * texp_dark.
        t0 : float or sequence of float
            Illumination start time(s) [s].
        texp_quartz : float or sequence of float
            Illumination duration(s) [s].
        flux : float or sequence of float
            Photon flux during each illumination [electrons/s].
        texp_dark : float, optional
            Dark exposure duration [s].  Default: 300.
        dp_obs : array-like, optional
            1-sigma errors on *p_obs*.  When given, the fit minimises
            ``((model - p_obs) / dp_obs)^2``.
        fixed_indices : sequence of int, optional
            Indices of components whose fractions are held fixed.

        Returns
        -------
        fitted_det : H4PersistenceModel
            New H4PersistenceModel with fitted fractions; taus and f_mids unchanged.
        info : dict
            Fitting diagnostics: fractions, df_fractions, residuals, rms,
            cost, success, solver, dataframe, metadata.
        """
        illums = self._to_illuminations(t0, texp_quartz, flux)
        self._validate_illuminations(illums)
        t_arr = self._to_array(t_obs)
        p_arr = self._to_array(p_obs)
        norm = sum(flux_k * texp_k for _, texp_k, flux_k in illums) * texp_dark
        mask_illuminated = self._illuminated_mask(t_arr, illums)

        A = np.zeros((len(t_arr), self.n_components))
        for j, comp in enumerate(self._components):
            unit = TrapComponent(fraction=1.0, tau=comp.tau, f_mid=comp.f_mid, label=comp.label)
            col = np.zeros(len(t_arr))
            for t0_k, texp_k, flux_k in illums:
                col += unit.charge(t_arr, t0_k, texp_k, flux_k) - unit.charge(
                    t_arr + texp_dark, t0_k, texp_k, flux_k
                )
            col[mask_illuminated] = 0.0
            A[:, j] = col / norm

        fixed_set = set(fixed_indices) if fixed_indices is not None else set()
        free_idx = [j for j in range(self.n_components) if j not in fixed_set]

        p_adj = p_arr.copy()
        for j in fixed_set:
            p_adj -= A[:, j] * self._components[j].fraction

        if dp_obs is not None:
            w = 1.0 / self._to_array(dp_obs)
            A_fit = A[:, free_idx] * w[:, None]
            p_fit = p_adj * w
        else:
            A_fit = A[:, free_idx]
            p_fit = p_adj

        finite_mask = np.isfinite(p_fit) & np.all(np.isfinite(A_fit), axis=1)
        if not np.any(finite_mask):
            raise ValueError("No finite data points after weighting. " "Check dp_obs for zero or NaN values.")
        A_solve = A_fit[finite_mask]
        p_solve = p_fit[finite_mask]

        fitted_f_free = None
        solver = ""
        try:
            try:
                result = lsq_linear(A_solve, p_solve, bounds=(0.0, 1.0), method="bvls")
                if np.all(np.isfinite(result.x)):
                    fitted_f_free = result.x
                    solver = "scipy lsq_linear (bvls)"
            except np.linalg.LinAlgError:
                pass
            if fitted_f_free is None:
                result = lsq_linear(
                    A_solve,
                    p_solve,
                    bounds=(0.0, 1.0),
                    method="trf",
                    lsq_solver="lsmr",
                )
                if np.all(np.isfinite(result.x)):
                    fitted_f_free = result.x
                    solver = "scipy lsq_linear (trf+lsmr)"
        except ImportError:
            pass
        if fitted_f_free is None:
            fitted_f_free = _bounded_lsq_pgd(A_solve, p_solve, lo=0.0, hi=1.0)
            solver = "numpy projected gradient descent"

        fitted_f = np.array([comp.fraction for comp in self._components])
        for k, j in enumerate(free_idx):
            fitted_f[j] = fitted_f_free[k]

        residuals = A @ fitted_f - p_arr
        rms = np.sqrt(np.mean(residuals**2))

        AtA_fit = A_solve.T @ A_solve
        n_free = len(free_idx)
        try:
            cov_free = np.linalg.lstsq(AtA_fit, np.eye(n_free), rcond=None)[0]
        except np.linalg.LinAlgError:
            cov_free = np.zeros((n_free, n_free))
        if dp_obs is None:
            dof = max(1, len(p_arr) - n_free)
            cov_free = cov_free * (np.dot(residuals, residuals) / dof)
        df_free = np.sqrt(np.maximum(0.0, np.diag(cov_free)))

        df = np.zeros(self.n_components)
        for k, j in enumerate(free_idx):
            df[j] = df_free[k]

        fitted_det = self.update_fractions(fitted_f.tolist(), name=self.name + " (fitted)")

        result_df = pd.DataFrame(
            {
                "tau": [c.tau for c in fitted_det.components],
                "f_mid": [c.f_mid for c in fitted_det.components],
                "fraction": fitted_f,
                "fraction_error": df,
            }
        )

        info = {
            "fractions": fitted_f,
            "df_fractions": df,
            "dataframe": result_df,
            "metadata": {"name": self.name},
            "residuals": residuals,
            "rms": rms,
            "cost": float(0.5 * np.dot(residuals, residuals)),
            "success": True,
            "solver": solver,
        }
        return fitted_det, info

    # ------------------------------------------------------------------
    # Validation / helpers
    # ------------------------------------------------------------------

    @typing.overload
    @staticmethod
    def _to_illuminations(
        t0: float,
        texp_quartz: float,
        flux: float,
    ) -> list[tuple[float, float, float]]:
        ...

    @typing.overload
    @staticmethod
    def _to_illuminations(
        t0: FloatArray1D,
        texp_quartz: FloatArray1D,
        flux: FloatArray1D,
    ) -> list[tuple[float, float, float]]:
        ...

    @staticmethod
    def _to_illuminations(
        t0,
        texp_quartz,
        flux,
    ) -> list[tuple[float, float, float]]:
        """Normalise illumination args to a list of (t0, texp, flux)."""
        if not (np.isscalar(t0) == np.isscalar(texp_quartz) == np.isscalar(flux)):
            raise ValueError("mixing scalars and arrays is not allowed.")
        if np.isscalar(t0):
            return [(float(typing.cast(float, t0)), float(texp_quartz), float(flux))]
        return [(float(a), float(b), float(c)) for a, b, c in zip(t0, texp_quartz, flux)]

    @staticmethod
    def _validate_illuminations(illums: list[tuple[float, float, float]]) -> None:
        for t0_k, texp_k, flux_k in illums:
            if texp_k <= 0:
                raise ValueError(f"texp_quartz must be > 0, got {texp_k}")
            if flux_k < 0:
                raise ValueError(f"flux must be >= 0, got {flux_k}")

    @staticmethod
    def _illuminated_mask(
        t_arr: np.ndarray,
        illums: list[tuple[float, float, float]],
    ) -> np.ndarray:
        """Boolean mask: True for times within any illumination window."""
        mask = np.zeros(len(t_arr), dtype=bool)
        for t0_k, texp_k, _ in illums:
            mask |= (t_arr >= t0_k) & (t_arr <= t0_k + texp_k)
        return mask

    @staticmethod
    def _to_array(t: float | Sequence[float] | np.ndarray) -> np.ndarray:
        return np.atleast_1d(np.asarray(t, dtype=float))

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    @typing.overload
    def plot(
        self,
        t0: float,
        texp_quartz: float,
        flux: float,
        *,
        texp_dark: float = ...,
        t_extra: float = ...,
        n_points: int = ...,
        log_scale: bool = ...,
        t_obs: np.ndarray | None = ...,
        p_obs: np.ndarray | None = ...,
        dp_obs: np.ndarray | None = ...,
        xlim: tuple[float, float] | None = ...,
        ylim1: tuple[float, float] | None = ...,
        ylim2: tuple[float, float] | None = ...,
        show: bool = ...,
        save_path: str | None = ...,
    ) -> plt.Figure:
        ...

    @typing.overload
    def plot(
        self,
        t0: FloatArray1D,
        texp_quartz: FloatArray1D,
        flux: FloatArray1D,
        *,
        texp_dark: float = ...,
        t_extra: float = ...,
        n_points: int = ...,
        log_scale: bool = ...,
        t_obs: np.ndarray | None = ...,
        p_obs: np.ndarray | None = ...,
        dp_obs: np.ndarray | None = ...,
        xlim: tuple[float, float] | None = ...,
        ylim1: tuple[float, float] | None = ...,
        ylim2: tuple[float, float] | None = ...,
        show: bool = ...,
        save_path: str | None = ...,
    ) -> plt.Figure:
        ...

    def plot(
        self,
        t0,
        texp_quartz,
        flux,
        *,
        texp_dark: float = 300.0,
        t_extra: float = 3600.0,
        n_points: int = 2000,
        log_scale: bool = False,
        t_obs: np.ndarray | None = None,
        p_obs: np.ndarray | None = None,
        dp_obs: np.ndarray | None = None,
        xlim: tuple[float, float] | None = None,
        ylim1: tuple[float, float] | None = None,
        ylim2: tuple[float, float] | None = None,
        show: bool = True,
        save_path: str | None = None,
    ):
        """Plot illumination timeline, trapped charge, and persistence.

        Parameters
        ----------
        t0 : float or sequence of float
            Illumination start time(s) [s].
        texp_quartz : float or sequence of float
            Illumination duration(s) [s].
        flux : float or sequence of float
            Photon flux during each illumination [electrons/s].
        texp_dark : float, optional
            Dark exposure duration [s].  Default: 300.
        t_extra : float, optional
            Time window after the end of the last illumination [s].  Default: 3600.
            Ignored when *xlim* is provided.
        n_points : int, optional
            Number of sample points on the time axis.  Default: 2000.
        log_scale : bool, optional
            If True, use log scale on both axes of the signal panels.
        t_obs, p_obs, dp_obs : array-like or None
            Observed data to overlay on the persistence panel.
        xlim, ylim1, ylim2 : tuple or None
            Axis limits.
        show : bool, optional
            Call ``plt.show()`` if True.
        save_path : str or None, optional
            Save figure to this path (dpi=150) if given.

        Returns
        -------
        matplotlib.figure.Figure
        """
        illums = self._to_illuminations(t0, texp_quartz, flux)
        self._validate_illuminations(illums)
        t_start = min(t0_k for t0_k, _, _ in illums)
        t_end_last = max(t0_k + texp_k for t0_k, texp_k, _ in illums)

        if xlim is not None:
            t_plot_start, t_stop = float(xlim[0]), float(xlim[1])
        else:
            t_plot_start, t_stop = t_start, t_end_last + t_extra

        if log_scale:
            t = np.logspace(np.log10(max(t_plot_start, 1e-3)), np.log10(t_stop), n_points)
        else:
            t = np.linspace(t_plot_start, t_stop, n_points)

        data_signal = self.signal_by_component(t, t0, texp_quartz, flux)
        data_pers = self.persistence_by_component(t, t0, texp_quartz, flux, texp_dark)
        colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

        fig, (ax0, ax1, ax2) = plt.subplots(
            3,
            1,
            figsize=(10, 10),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 3, 3]},
        )
        fig.suptitle(
            f"PFS H4RG Persistence Model --- {self.name}",
            fontsize=14,
        )

        # --- Panel 0: illumination timeline with t_mid markers ---
        for k, (t0_k, texp_k, flux_k) in enumerate(illums):
            t_end_k = t0_k + texp_k
            label = "Illumination" if k == 0 else None
            ax0.axvspan(t0_k, t_end_k, ymin=0.2, ymax=0.8, color="gold", alpha=0.8, label=label)
            ax0.axvline(t_end_k, color="steelblue", lw=1.5, ls="--")
            # Mark t_mid for each component (use mean f_mid if they differ)
            mean_f_mid = float(np.mean(self.f_mids))
            t_mid_k = t0_k + mean_f_mid * texp_k
            mid_label = "t_mid (mean)" if k == 0 else None
            ax0.axvline(t_mid_k, color="darkorange", lw=1.2, ls=":", label=mid_label)
            ax0.text(
                (t0_k + t_end_k) / 2,
                0.50,
                f"#{k+1}  $\\Phi$={flux_k:.2e}\ntexp={texp_k:.1f} s",
                ha="center",
                va="center",
                fontsize=7,
                color="saddlebrown",
                transform=ax0.get_xaxis_transform(),
            )
        ax0.set_yticks([])
        ax0.set_ylabel("Timeline", fontsize=9)
        ax0.set_title(f"Illumination timeline  ({len(illums)} event(s))", fontsize=10)
        ax0.legend(loc="upper right", fontsize=8)
        ax0.set_ylim(0, 1)

        # --- Panel 1: individual components + total (trapped charge) ---
        for i, comp in enumerate(self._components):
            ax1.plot(
                t,
                data_signal[comp.label],
                color=colors[i % 10],
                label=f"{comp.label}  (f={comp.fraction:.4f}, f_mid={comp.f_mid:.2f})",
            )
        ax1.plot(t, data_signal["total"], color="black", lw=2.5, label="Total")
        for _, (t0_k, texp_k, _) in enumerate(illums):
            ax1.axvline(t0_k + texp_k, color="steelblue", lw=1.5, ls="--")
            mean_f_mid = float(np.mean(self.f_mids))
            ax1.axvline(t0_k + mean_f_mid * texp_k, color="darkorange", lw=1.2, ls=":")
        ax1.set_ylabel("Trapped charge [e${}^-$]")
        ax1.set_title("Trapped charge --- components & total")
        ax1.legend(fontsize=8)
        ax1.grid(True, which="both", alpha=0.3)
        if log_scale:
            ax1.set_yscale("log")
            ax1.set_xscale("log")

        # --- Panel 2: persistence normalized by \sum(flux \cdot texp) * texp_dark ---
        norm = sum(flux_k * texp_k for _, texp_k, flux_k in illums) * texp_dark
        for i, comp in enumerate(self._components):
            ax2.plot(
                t,
                data_pers[comp.label] / norm,
                color=colors[i % 10],
                label=f"{comp.label}  (f={comp.fraction:.4f}, f_mid={comp.f_mid:.2f})",
            )
        ax2.plot(t, data_pers["total"] / norm, color="black", lw=2.5, label="Total")
        if t_obs is not None and p_obs is not None:
            if dp_obs is not None:
                ax2.errorbar(
                    np.asarray(t_obs),
                    np.asarray(p_obs),
                    yerr=np.asarray(dp_obs),
                    fmt="o",
                    ms=4,
                    color="red",
                    ecolor="red",
                    elinewidth=1,
                    capsize=3,
                    zorder=5,
                    label="Observed",
                    alpha=0.7,
                )
            else:
                ax2.scatter(
                    np.asarray(t_obs),
                    np.asarray(p_obs),
                    s=15,
                    color="red",
                    zorder=5,
                    label="Observed",
                    alpha=0.7,
                )
        for _, (t0_k, texp_k, _) in enumerate(illums):
            ax2.axvline(t0_k + texp_k, color="steelblue", lw=1.5, ls="--")
            mean_f_mid = float(np.mean(self.f_mids))
            ax2.axvline(t0_k + mean_f_mid * texp_k, color="darkorange", lw=1.2, ls=":")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel(r"Persistence / ($\sum$ flux $\cdot$ texp $\times$ t_exp,dark)")
        ax2.set_title(f"Persistence  (t_exp,dark = {texp_dark:.0f} s)")
        ax2.legend(fontsize=8)
        ax2.grid(True, which="both", alpha=0.3)
        if log_scale:
            ax2.set_yscale("log")
            ax2.set_xscale("log")

        if xlim is not None:
            ax0.set_xlim(xlim)
        if ylim1 is not None:
            ax1.set_ylim(ylim1)
        if ylim2 is not None:
            ax2.set_ylim(ylim2)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # Summary / display
    # ------------------------------------------------------------------

    def summary(
        self,
        t0: float = 0.0,
        texp_quartz: float | None = None,
        flux: float | None = None,
        df_fractions: np.ndarray | None = None,
    ) -> None:
        """Print a formatted summary of trap components.

        If *t0*, *texp_quartz*, and *flux* are provided, also prints the
        trapped charge at key times relative to the end of illumination.
        """
        print(f"H4PersistenceModel: {self.name}")
        print(f"  Components   : {self.n_components}")
        print(f"  Total fraction: {self.total_fraction:.4f}")
        print()
        if df_fractions is not None:
            df_arr = np.asarray(df_fractions)
            print(
                f"  {'#':>3}  {'label':<22}  {'fraction':>10}  {'+- error':>10}"
                f"  {'f_mid':>6}  {'tau [s]':>10}"
            )
            print("  " + "-" * 74)
            for i, comp in enumerate(self._components):
                print(
                    f"  {i:>3}  {comp.label:<22}  {comp.fraction:>10.5f}"
                    f"  {df_arr[i]:>10.5f}  {comp.f_mid:>6.3f}  {comp.tau:>10.1f}"
                )
        else:
            print(f"  {'#':>3}  {'label':<22}  {'fraction':>10}  {'f_mid':>6}  {'tau [s]':>10}")
            print("  " + "-" * 60)
            for i, comp in enumerate(self._components):
                print(
                    f"  {i:>3}  {comp.label:<22}  {comp.fraction:>10.5f}"
                    f"  {comp.f_mid:>6.3f}  {comp.tau:>10.1f}"
                )

        if texp_quartz is not None and flux is not None:
            t_end = t0 + texp_quartz
            total_in = flux * texp_quartz
            check_dt = [0.0, texp_quartz, 10 * texp_quartz, 100 * texp_quartz]
            labels = [
                "end of illumination (t_end)",
                f"+{texp_quartz:.0f} s after t_end",
                f"+{10*texp_quartz:.0f} s after t_end",
                f"+{100*texp_quartz:.0f} s after t_end",
            ]
            print()
            print(f"  t0={t0} s, texp_quartz={texp_quartz} s, flux={flux:.2e} e-/s")
            print(f"  Total incoming charge = {total_in:.3e} e-")
            print()
            print(f"  {'label':<30} {'time [s]':>10} {'trapped [e-]':>14} {'fraction':>10}")
            print("  " + "-" * 69)
            for dt, lbl in zip(check_dt, labels):
                tc = t_end + dt
                q = self.signal(tc, t0, texp_quartz, flux)
                print(f"  {lbl:<30} {tc:>10.1f} {float(q):>14.2f}" f" {float(q)/total_in:>10.2e}")

    def __repr__(self) -> str:
        comp_str = ", ".join(f"(f={c.fraction}, tau={c.tau}s, f_mid={c.f_mid})" for c in self._components)
        return f"H4PersistenceModel(name={self.name!r}, components=[{comp_str}])"

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

    def select(
        self,
        fiberId: IntArray1D
    ) -> "H4Persistence":
        """Get a subset of this object containing only ``fiberId``

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
            raise RuntimeError(f"No persistence for fiberId={exc.args[0]}") from exc

        return type(self)(
            fiberId=fiberId,
            wavelength=self.wavelength,
            flux=self.flux[indices, ...],
            identity=self.identity,
        )
