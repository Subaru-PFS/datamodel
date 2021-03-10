import types

import astropy.io.fits

from .utils import combineArms

__all__ = ("Identity", "CalibIdentity")


class Identity(types.SimpleNamespace):
    """Identification of an exposure

    An exposure is identified by its ``visit``, ``arm`` and ``spectrograph``,
    for which a corresponding ``pfsDesignId`` was used.

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
    """
    defaultArm = "x"  # Default value for 'arm'
    defaultSpectrograph = 0  # Default value for spectrograph
    defaultPfsDesignId = -1  # Default value for pfsDesignId
    fitsExtension = "CONFIG"  # Name for FITS extension; choice of "CONFIG" is historical

    def __init__(self, visit, arm=None, spectrograph=None, pfsDesignId=None):
        super().__init__(visit=visit, _arm=arm, _spectrograph=spectrograph, _pfsDesignId=pfsDesignId)

    @property
    def arm(self):
        return self._arm if self._arm is not None else self.defaultArm

    @property
    def spectrograph(self):
        return self._spectrograph if self._spectrograph is not None else self.defaultSpectrograph

    @property
    def pfsDesignId(self):
        return self._pfsDesignId if self._pfsDesignId is not None else self.defaultPfsDesignId

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
                   ]
        hdu = astropy.io.fits.BinTableHDU.from_columns(columns, name=self.fitsExtension, header=header)
        fits.append(hdu)

    @classmethod
    def fromMerge(cls, identities):
        """Construct by merging multiple identities

        Parameters
        ----------
        identities : `list` of `Identity`
            Identities to merge.

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

        return cls(visit, arm, spectrograph, pfsDesignId)

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
        return cls(identity["visit"], identity.get("arm", None), identity.get("spectrograph", None),
                   identity.get("pfsDesignId", None))


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

    def __init__(self, obsDate, spectrograph, arm, visit0):
        super().__init__(obsDate=obsDate, spectrograph=int(spectrograph), arm=arm, visit0=int(visit0))

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
