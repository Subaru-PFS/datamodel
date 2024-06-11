import enum
import types
from typing import Dict, Union

import numpy as np
import astropy.io.fits

from .utils import combineArms

__all__ = ("Identity", "CalibIdentity", "ObsTimeMergeStrategy", "ExpTimeMergeStrategy")


class ObsTimeMergeStrategy(enum.Enum):
    """Strategy enum for merging observation times

    * ``EARLIEST``: Use the earliest observation time.
    * ``LATEST``: Use the latest observation time.

    In the future, we may add an ``AVERAGE`` strategy, but this has not been
    implemented yet in order to avoid the need to parse the observation times.
    """
    EARLIEST = enum.auto()
    LATEST = enum.auto()


class ExpTimeMergeStrategy(enum.Enum):
    """Strategy enum for merging exposure times

    * ``SUM``: Sum the exposure times.
    * ``AVERAGE``: Average the exposure times.
    """
    SUM = enum.auto()
    AVERAGE = enum.auto()


class Identity(types.SimpleNamespace):
    """Identification of an exposure

    An exposure is identified by its ``visit``, ``arm`` and ``spectrograph``,
    for which a corresponding ``pfsDesignId`` was used. We also include
    ``obsTime`` (time of observation) and ``expTime`` (exposure time), for
    convenience.

    Sometimes, exposures are combined, in which case the ``arm`` and
    ``spectrograph`` will be set to default values (``defaultArm`` and
    ``defaultSpectrograph`` class attributes).

    Since the ``pfsDesignId`` is uniquely specified from a ``visit``, it
    may not be known at construction time, in which case it will be set to a
    default value ``defaultPfsDesignId`` class attribute).

    Parameters
    ----------
    visit : `int`
        Visit identifier.
    arm : `str`, optional
        Spectrograph arm identifier.
    spectrograph: `int`, optional
        Spectrograph module identifier.
    pfsDesignId : `int`, optional
        Top-end design identifier.
    obsTime : `str`, optional
        Time of observation.
    expTime : `float`, optional
        Exposure time (sec).
    """
    defaultArm = "x"  # Default value for 'arm'
    defaultSpectrograph = 0  # Default value for spectrograph
    defaultPfsDesignId = -1  # Default value for pfsDesignId
    defaultObsTime = "UNKNOWN"  # Default value for obsTime
    defaultExpTime = np.nan  # Default value for expTime
    fitsExtension = "CONFIG"  # Name for FITS extension; choice of "CONFIG" is historical

    def __init__(self, visit, arm=None, spectrograph=None, pfsDesignId=None, obsTime=None, expTime=None):
        super().__init__(
            visit=visit,
            _arm=arm,
            _spectrograph=spectrograph,
            _pfsDesignId=pfsDesignId,
            _obsTime=obsTime,
            _expTime=expTime,
        )

    @property
    def arm(self):
        return self._arm if self._arm is not None else self.defaultArm

    @property
    def spectrograph(self):
        return self._spectrograph if self._spectrograph is not None else self.defaultSpectrograph

    @property
    def pfsDesignId(self):
        return self._pfsDesignId if self._pfsDesignId is not None else self.defaultPfsDesignId

    @property
    def obsTime(self):
        return self._obsTime if self._obsTime is not None else self.defaultObsTime

    @property
    def expTime(self):
        return self._expTime if self._expTime is not None else self.defaultExpTime

    def getDict(self):
        """Generate a set of keyword-value pairs

        Returns
        -------
        identity : `dict` (`str`: POD)
            Keyword-value pairs for this identity.
        """
        identity = dict(visit=self.visit)
        if self._arm is not None:
            identity["arm"] = self._arm
        if self._spectrograph is not None:
            identity["spectrograph"] = self._spectrograph
        if self._pfsDesignId is not None:
            identity["pfsDesignId"] = self._pfsDesignId
        if self._obsTime is not None:
            identity["obsTime"] = self._obsTime
        if self._expTime is not None:
            identity["expTime"] = self._expTime
        return identity

    @classmethod
    def fromFits(cls, fits):
        """Construct from a FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            Opened FITS file.

        Returns
        -------
        self : `Identity`
            Constructed identity.
        """
        hdu = fits[cls.fitsExtension]
        kwargs = {col: hdu.data[col][0] for col in ("visit", "arm", "spectrograph", "pfsDesignId")}
        if kwargs["arm"] == cls.defaultArm:
            del kwargs["arm"]
        if kwargs["spectrograph"] == cls.defaultSpectrograph:
            del kwargs["spectrograph"]
        if kwargs["pfsDesignId"] == cls.defaultPfsDesignId:
            del kwargs["pfsDesignId"]
        # obsTime and expTime are optional, for backwards compatibility
        if "obsTime" in hdu.columns.names:
            obsTime = hdu.data["obsTime"][0].tobytes().decode("utf-32")
            if obsTime != cls.defaultObsTime:
                kwargs["obsTime"] = obsTime
        if "expTime" in hdu.columns.names:
            expTime = hdu.data["expTime"][0]
            if not np.isfinite(expTime):
                kwargs["expTime"] = expTime
        return cls(**kwargs)

    def toFits(self, fits):
        """Write to a FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            Opened FITS file.
        """
        # NOTE: When making any changes to this method that modify the output
        # format, increment the DAMD_VER header value and record the change in
        # the versions.txt file.
        header = astropy.io.fits.Header()
        header['DAMD_VER'] = (1, "Identity datamodel version")
        columns = [astropy.io.fits.Column(name="visit", format="J", array=[self.visit]),
                   astropy.io.fits.Column(name="arm", format=f"{len(self.arm)}A", array=[self.arm]),
                   astropy.io.fits.Column(name="spectrograph", format="J", array=[self.spectrograph]),
                   astropy.io.fits.Column(name="pfsDesignId", format="K", array=[self.pfsDesignId]),
                   astropy.io.fits.Column(name="obsTime", format="PA()", array=[self.obsTime]),
                   astropy.io.fits.Column(name="expTime", format="D", array=[self.expTime]),
                   ]
        hdu = astropy.io.fits.BinTableHDU.from_columns(columns, name=self.fitsExtension, header=header)
        fits.append(hdu)

    @classmethod
    def fromMerge(
        cls,
        identities,
        *,
        obsTimeStrategy=ObsTimeMergeStrategy.EARLIEST,
        expTimeStrategy=ExpTimeMergeStrategy.SUM,
    ):
        """Construct by merging multiple identities

        Parameters
        ----------
        identities : `list` of `Identity`
            Identities to merge.
        obsTimeStrategy : `ObsTimeMergeStrategy`, optional
            Strategy for merging observation times.
        expTime : `ExpTimeMergeStrategy`, optional
            Strategy for merging exposure times.

        Returns
        -------
        merged : `Identity`
            Merged identity.

        Raises
        ------
        RuntimeError
            If the identities cannot be merged (because they do not have a
            common ``visit`` and ``pfsDesignId``).
        """
        visit = set([ident.visit for ident in identities])
        if len(visit) != 1:
            raise RuntimeError(f"visit is non-unique: {visit}")
        visit = visit.pop()

        pfsDesignId = set([ident._pfsDesignId for ident in identities if ident._pfsDesignId is not None])
        if len(pfsDesignId) not in (0, 1):
            raise RuntimeError(f"pfsDesignId is non-unique: {pfsDesignId}")
        pfsDesignId = pfsDesignId.pop() if pfsDesignId else None

        armSet = set([ident._arm for ident in identities if ident._arm is not None])
        arm = combineArms(armSet)

        spectrograph = set([ident._spectrograph for ident in identities if ident._spectrograph is not None])
        spectrograph = spectrograph.pop() if len(spectrograph) == 1 else None

        obsTime = [ident._obsTime for ident in identities if ident._obsTime is not None]
        if not obsTime:
            obsTime = None
        elif obsTimeStrategy == ObsTimeMergeStrategy.EARLIEST:
            obsTime = min(obsTime)
        elif obsTimeStrategy == ObsTimeMergeStrategy.LATEST:
            obsTime = max(obsTime)

        expTime = np.array([ident._expTime for ident in identities if ident._expTime is not None])
        if expTime.size == 0:
            expTime = None
        elif expTimeStrategy == ExpTimeMergeStrategy.SUM:
            expTime = np.sum(expTime)
        elif expTimeStrategy == ExpTimeMergeStrategy.AVERAGE:
            expTime = np.mean(expTime)

        return cls(visit, arm, spectrograph, pfsDesignId, obsTime, expTime)

    @classmethod
    def fromDict(cls, identity):
        """Construct from a dict

        This is intended for use with a ``dataId`` such as used by the LSST
        butler.

        Parameters
        ----------
        identity : `dict` (`str`: POD)
            Keyword-value pairs identifying the data. Must have ``visit`` and
            ``pfsDesignId`` keywords, and may have ``arm`` and ``spectrograph``.

        Returns
        -------
        self : `Identity`
            Constructed identity.
        """
        return cls(
            identity["visit"],
            identity.get("arm", None),
            identity.get("spectrograph", None),
            identity.get("pfsDesignId", None),
            identity.get("obsTime", None),
            identity.get("expTime", None),
        )


class CalibIdentity(types.SimpleNamespace):
    """Keyword-value pairs describing a calibration

    Parameters
    ----------
    obsDate : `str`
        Observation date of calibration, in ISO-8601 format.
    spectrograph : `int`
        Spectrograph number.
    arm : `str`
        Arm letter: ``b``, ``r``, ``n``, ``m``.
    visit0 : `int`
        First visit number calibration was constructed from.
    """

    elements = ("obsDate", "spectrograph", "arm", "visit0")
    """Required keywords"""

    headerKeywords = ("DATEOBS", "W_SPMOD", "W_ARM", "W_VISIT")
    """Corresponding header keywords to use"""

    def __init__(self, obsDate, spectrograph, arm, visit0):
        super().__init__(obsDate=obsDate, spectrograph=int(spectrograph), arm=arm, visit0=int(visit0))

    def __reduce__(self):
        """Support pickling"""
        return (self.__class__, (self.obsDate, self.spectrograph, self.arm, self.visit0))

    @classmethod
    def fromDict(cls, identity):
        """Build from a `dict`

        Parameters
        ----------
        identity : `dict`
            The data identifier.

        Returns
        -------
        self : `CalibIdentity`
            Calibration identity.s
        """
        identity = identity.copy()
        if "obsDate" not in identity:
            identity["obsDate"] = identity["dateObs"]
        if "visit0" not in identity:
            identity["visit0"] = identity["visit"]
        kwargs = {elem: identity[elem] for elem in cls.elements}
        return cls(**kwargs)

    def toDict(self):
        """Convert to a `dict`

        Returns
        -------
        calibId : `dict`
            Data identity for calibration.
        """
        return {elem: getattr(self, elem) for elem in self.elements}

    def toHeader(self) -> Dict[str, Union[str, int]]:
        """Convert to FITS header keyword-value pairs

        Returns
        -------
        header : `dict`
            Header keyword-value pairs.
        """
        return {key: getattr(self, name) for name, key in zip(self.elements, self.headerKeywords)}

    @classmethod
    def fromHeader(cls, header: Dict[str, Union[str, int]]) -> "CalibIdentity":
        """Construct from FITS header

        Parameters
        ----------
        header : `dict`
            Header keyword-value pairs.

        Returns
        -------
        self : `CalibIdentity`
            Constructed `CalibIdentity`.
        """

        return cls(**{name: header[key] for name, key in zip(cls.elements, cls.headerKeywords)})
