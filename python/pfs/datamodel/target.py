import types
import numpy as np

from .utils import astropyHeaderFromDict, astropyHeaderToDict
from .pfsConfig import TargetType

__all__ = ("Target",)


class Target(types.SimpleNamespace):
    """A spectroscopic target

    Parameters
    ----------
    catId : `int`
        Catalog identifier of the object.
    tract : `int`
        Tract in which the object resides.
    patch : `str`
        Patch in which the object resides.
    objId : `objId`
        Object identifier of the object.
    ra : `float`, optional
        Right Ascension of the object.
    dec : `float`, optional
        Declination of the object.
    targetType : `TargetType`, optional
        Type of target (typically ``SCIENCE``).
    fiberFlux : `dict` mapping `str` to `float`, optional
        Filter names and corresponding fiber fluxes.
    """
    _attributes = ("catId", "tract", "patch", "objId", "ra", "dec", "targetType")  # Read from header
    """Attributes to read from FITS header (iterable of `str`)"""

    def __init__(self, catId, tract, patch, objId, ra=np.nan, dec=np.nan, targetType=-1, fiberFlux=None):
        self.catId = catId
        self.tract = tract
        self.patch = patch
        self.objId = objId
        self.ra = ra
        self.dec = dec
        self.targetType = targetType
        self.fiberFlux = fiberFlux if fiberFlux is not None else {}
        self.identity = dict(catId=catId, tract=tract, patch=patch, objId=objId)

    def __str__(self):
        """Stringify"""
        return "catId=%d tract=%d patch=%s objId=%d" % (self.catId, self.tract, self.patch, self.objId)

    @classmethod
    def fromFits(cls, fits):
        """Construct from a FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            Opened FITS file.

        Returns
        -------
        self : `Target`
            Constructed `Target`.
        """
        hdu = fits["TARGET"]
        header = astropyHeaderToDict(hdu.header)
        kwargs = {}
        for attr in cls._attributes:
            kwargs[attr] = header[attr.upper()]
        kwargs["fiberFlux"] = dict(zip(hdu.data["filterName"], hdu.data["fiberFlux"]))
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
        from astropy.io.fits import BinTableHDU, Column
        maxLength = max(len(ff) for ff in self.fiberFlux.keys()) if self.fiberFlux else 1
        header = astropyHeaderFromDict({attr.upper(): getattr(self, attr) for attr in self._attributes})
        header.update(TargetType.getFitsHeaders())
        header['DAMD_VER'] = (1, "Target datamodel version")
        hdu = BinTableHDU.from_columns([
            Column("filterName", "%dA" % maxLength, array=list(self.fiberFlux.keys())),
            Column("fiberFlux", "E", array=np.array(list(self.fiberFlux.values()))),
        ], header=header, name="TARGET")
        fits.append(hdu)

    def __eq__(self, other):
        """Comparison

        We do not compare the full set of contents. We care only about the
        parts that should uniquely identify the `Target`, viz., the ``catId``,
        ``tract``, ``patch``, and ``objId``. The other attributes are
        considered helpful additions, but are not used in determining whether
        one `Target` is "equal" to another `Target`.
        """
        if not isinstance(other, self.__class__):
            return False
        for attr in ("catId", "tract", "patch", "objId"):
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

    def __hash__(self):
        return hash((self.catId, self.tract, self.patch, self.objId))

    @classmethod
    def fromPfsConfig(cls, pfsConfig, index):
        """Construct from a PfsConfig

        Parameters
        ----------
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration.
        index : `int`
            Index into the ``pfsConfig`` arrays for the target of interest.

        Returns
        -------
        self : cls
            Constructed `Target`.
        """
        catId = pfsConfig.catId[index]
        tract = pfsConfig.tract[index]
        patch = pfsConfig.patch[index]
        objId = pfsConfig.objId[index]
        ra = pfsConfig.ra[index]
        dec = pfsConfig.dec[index]
        fiberFlux = dict(zip(pfsConfig.filterNames[index], pfsConfig.fiberFlux[index]))
        targetType = pfsConfig.targetType[index]
        return cls(catId, tract, patch, objId, ra, dec, targetType, fiberFlux)

    def __reduce__(self):
        """How to pickle"""
        return type(self), (self.catId, self.tract, self.patch, self.objId, self.ra, self.dec,
                            self.targetType, self.fiberFlux)
