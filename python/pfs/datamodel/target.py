import types
import hashlib
import numpy as np

from .utils import astropyHeaderFromDict, astropyHeaderToDict, createHash, wraparoundNVisit

__all__ = ["TargetData", "TargetObservations"]

class TargetData(types.SimpleNamespace):
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
    ra : `float`
        Right Ascension of the object.
    dec : `float`
        Declination of the object.
    targetType : `TargetType`
        Type of target (typically ``SCIENCE``).
    fiberMags : `dict` mapping `str` to `float`
        Filter names and corresponding fiber magnitudes.
    """
    _attributes = ("catId", "tract", "patch", "objId", "ra", "dec", "targetType")  # Read from header
    """Attributes to read from FITS header (iterable of `str`)"""

    def __init__(self, catId, tract, patch, objId, ra, dec, targetType, fiberMags):
        self.catId = catId
        self.tract = tract
        self.patch = patch
        self.objId = objId
        self.ra = ra
        self.dec = dec
        self.targetType = targetType
        self.fiberMags = fiberMags
        self.identity = dict(catId=catId, tract=tract, patch=patch, objId=objId, targetType=targetType)

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
        self : `TargetData`
            Constructed `TargetData`.
        """
        hdu = fits["TARGET"]
        header = astropyHeaderToDict(hdu.header)
        kwargs = {}
        for attr in cls._attributes:
            kwargs[attr] = header[attr.upper()]
        kwargs["fiberMags"] = dict(zip(hdu.data["filterName"], hdu.data["fiberMag"]))
        return cls(**kwargs)

    def toFits(self, fits):
        """Write to a FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            Opened FITS file.
        """
        from astropy.io.fits import BinTableHDU, Column
        maxLength = max(len(ff) for ff in self.fiberMags.keys()) if self.fiberMags else 1
        header = {attr.upper(): getattr(self, attr) for attr in self._attributes}
        hdu = BinTableHDU.from_columns([
            Column("filterName", "%dA" % maxLength, array=list(self.fiberMags.keys())),
            Column("fiberMag", "D", array=np.array(list(self.fiberMags.values()))),
        ], header=astropyHeaderFromDict(header), name="TARGET")
        fits.append(hdu)


class TargetObservations(types.SimpleNamespace):
    """A group of observations of a spectroscopic target

    Parameters
    ----------
    target : `TargetData`
        The spectroscopic target we observed.
    identity : `list` of `dict`
        A list of keyword-value pairs identifying each observation.
    fiberId : `numpy.ndarray` of `int`
        Array of fiber identifiers for this object in each observation.
    pfiNominal : `numpy.ndarray` of `float`
        Array of nominal fiber positions (x,y) for this object in each
        observation.
    pfiCenter : `numpy.ndarray` of `float`
        Array of actual fiber positions (x,y) for this object in each
        observation.
    """
    def __init__(self, identity, fiberId, pfiNominal, pfiCenter):
        self.identity = identity
        self.fiberId = fiberId
        self.pfiNominal = pfiNominal
        self.pfiCenter = pfiCenter

        self.num = len(self.fiberId)
        self.validate()

    def __len__(self):
        """Number of observations"""
        return self.num

    def validate(self):
        """Validate that all arrays are of the expected shape"""
        assert len(self.identity) == self.num
        assert self.fiberId.shape == (self.num,)
        assert self.pfiNominal.shape == (self.num, 2)
        assert self.pfiNominal.shape == (self.num, 2)

    def calculateExpHash(self, keys=("visit",)):
        """Calculate hash of the exposure inputs

        Parameters
        ----------
        keys : iterable
            Iterable of keys from the ``identity`` to use in constructing the
            hash.

        Returns
        -------
        hash : `int`
            Hash, truncated to 63 bits.
        """
        return createHash([str(indent[kk]).encode() for indent in self.identity for kk in sorted(keys)])

    def getIdentity(self, hashKeys=("visit",)):
        """Return the identity of these observations

        Parameters
        ----------
        hashKeys : iterable of `str`
            Iterable of keys from the ``identity`` to use in constructing the
            hash.

        Returns
        -------
        identity : `dict`
            Keyword-value pairs identifying these observations.
        """
        return dict(numExp=wraparoundNVisit(len(self)), expHash=self.calculateExpHash(hashKeys))

    @classmethod
    def fromFits(cls, fits):
        """Construct from a FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            Opened FITS file.

        Returns
        -------
        self : `TargetObservations`
            Constructed observations.
        """
        hdu = fits["OBSERVATIONS"]
        kwargs = {col: hdu.data[col] for col in ("fiberId", "pfiNominal", "pfiCenter")}
        kwargs["identity"] = [eval(ident) for ident in hdu.data["identity"]]
        return cls(**kwargs)

    def toFits(self, fits):
        """Write to a FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            Opened FITS file.
        """
        from astropy.io.fits import BinTableHDU, Column
        identityLength = max(len(str(ident)) for ident in self.identity)
        hdu = BinTableHDU.from_columns([
            Column("identity", "%dA" % identityLength, array=self.identity),
            Column("fiberId", "K", array=self.fiberId),
            Column("pfiNominal", "2D", array=self.pfiNominal),
            Column("pfiCenter", "2D", array=self.pfiCenter),
        ], name="OBSERVATIONS")
        fits.append(hdu)
