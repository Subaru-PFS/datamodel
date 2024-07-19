from typing import Any, Dict, Optional

import numpy as np
import astropy.io.fits

from .utils import astropyHeaderFromDict, astropyHeaderToDict, createHash
from .identity import CalibIdentity
from .pfsConfig import PfsConfig
from .pfsFiberArraySet import PfsFiberArraySet
from .masks import MaskHelper

__all__ = ("PfsFiberNorms",)


class PfsFiberNorms:
    """Fiber normalisations

    Coefficients of the flux correction for each fiber, used to normalise the
    science spectra in order to remove instrumental features.

    Parameters
    ----------
    identity : `pfs.datamodel.Identity`
        Identity of the data.
    fiberId : `numpy.ndarray` of `int`
        Fiber identifiers for each spectrum.
    wavelength : `numpy.ndarray` of `float`
        Wavelength for each pixel of each fiber.
    values : `numpy.ndarray` of `float`
        Norm value for each pixel of each fiber.
    fiberProfilesHash : `dict` mapping `int` to `int`
        Hash of the fiberProfiles used to generate the coefficients, indexed
        by spectrograph number.
    model : `astropy.io.fits.BinTableHDU`
        Table of model parameters. Since we already have the values, this is
        unused except for provenance and QA. The format is not specified here.
    metadata : `dict` (`str`: POD), optional
        Keyword-value pairs for the header.
    """

    def __init__(
        self,
        identity: CalibIdentity,
        fiberId: np.ndarray,
        wavelength: np.ndarray,
        values: np.ndarray,
        fiberProfilesHash: Dict[int, int],
        model: astropy.io.fits.BinTableHDU,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.identity = identity
        self.fiberId = fiberId
        self.wavelength = wavelength
        self.values = values
        self.fiberProfilesHash = fiberProfilesHash
        self.model = model
        self.metadata = metadata if metadata is not None else {}

        self.numFibers = len(fiberId)
        self.height = wavelength.shape[1]

        self.validate()

    def validate(self):
        """Validate that all the arrays are of the expected shape"""
        assert self.fiberId.shape == (self.numFibers,)
        assert self.wavelength.shape == (self.numFibers, self.height)
        assert self.values.shape == (self.numFibers, self.height)

    def __len__(self) -> int:
        """Return the number of fibers"""
        return self.numFibers

    def __getitem__(self, logical: np.ndarray) -> "PfsFiberNorms":
        """Sub-selection

        Parameters
        ----------
        logical : `numpy.ndarray` of `bool`
            Boolean array (of same length as ``self``) indicating which fibers
            to select.

        Returns
        -------
        new : ``type(self)``
            A new instance containing only the selected fibers.
        """
        kwargs = {
            name: getattr(self, name) for name in ("identity", "fiberProfilesHash", "model", "metadata")
        }
        kwargs.update(**{
            name: getattr(self, name)[logical] for name in ("fiberId", "wavelength", "values")
        })
        return type(self)(**kwargs)

    def select(self, pfsConfig: Optional[PfsConfig] = None, **kwargs) -> "PfsFiberNorms":
        """Return an instance containing only the selected attributes

        Multiple attributes will be combined with ``AND``.

        Parameters
        ----------
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration.  Optional if the only selection is on spectrograph or fiberId
        fiberId : `int` (scalar or array_like), optional
            Fiber identifier to select.  pfsConfig may be omitted
        targetType : `TargetType` (scalar or array_like), optional
            Target type to select.
        fiberStatus : `FiberStatus` (scalar or array_like), optional
            Fiber status to select.
        catId : `int` (scalar or array_like), optional
            Catalog identifier to select.
        tract : `int` (scalar or array_like), optional
            Tract number to select.
        patch : `str` (scalar or array_like), optional
            Patch name to select.
        objId : `int` (scalar or array_like), optional
            Object identifier to select.
        spectrograph : `int` (scalar or array_like), optional
            Spectrograph number to select.  pfsConfig may be omitted

        Returns
        -------
        selected : ``type(self)``
            An instance containing only the selected attributes.
        """
        keys = set(kwargs)
        ll = np.ones(len(self), dtype=bool)
        for kw in ["fiberId", "spectrograph"]:    # no need for a pfsConfig
            if kw in kwargs:
                ll &= np.isin(getattr(self, kw), kwargs[kw])
                keys.discard(kw)

        if len(keys) == 0:
            return self[ll]

        if pfsConfig is None:
            raise RuntimeError(
                "You must provide a pfsConfig file for all selections except"
                " spectrograph=[...], fiberId=[...]" + ("; saw \"%s\"" % '", "'.join(keys))
            )

        selection = pfsConfig.getSelection(**kwargs)
        return self[np.isin(self.fiberId, pfsConfig.fiberId[selection])]

    def __eq__(self, other):
        """Compare for equality"""
        for attr in ("identity",):
            if getattr(self, attr) != getattr(other, attr):
                return False
        for attr in ("fiberId", "wavelength", "values"):
            if not np.array_equal(getattr(self, attr), getattr(other, attr)):
                return False
        # Not comparing metadata
        return True

    @property
    def hash(self):
        """Provide hash of this object

        Note: not using ``__hash__`` because the seed for that varies between
        runs, and we want this to be constant for the same data.
        """
        return createHash((
            self.identity,
            self.fiberId.tobytes(),
            self.wavelength.tobytes(),
            self.values.tobytes(),
        ))

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
            Keyword arguments for constructing PfsFiberNorms.
        """
        if fits[0].header["DAMD_VER"] == 1:
            raise RuntimeError("Cannot read obsolete PfsFiberNorms with DAMD_VER=1")

        data: Dict[str, Any] = {}
        data["identity"] = CalibIdentity.fromHeader(fits[0].header)
        data["fiberId"] = fits["FIBERID"].data.astype(int)
        data["wavelength"] = fits["WAVELENGTH"].data.astype(float)
        data["values"] = fits["VALUES"].data.astype(float)
        data["fiberProfilesHash"] = dict(zip(
            fits["FIBERPROFILESHASH"].data["SPECTROGRAPH"],
            fits["FIBERPROFILESHASH"].data["HASH"],
        ))
        data["model"] = fits["MODEL"].copy()
        data["metadata"] = astropyHeaderToDict(fits[0].header)
        return data

    @classmethod
    def readFits(cls, filename: str) -> "PfsFiberNorms":
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
        import astropy.io.fits

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
        from astropy.io.fits import BinTableHDU, Column, ImageHDU, PrimaryHDU, Header

        metadata = self.identity.toHeader()
        if self.metadata:
            metadata.update(self.metadata)
        header = astropyHeaderFromDict(metadata)
        header["DAMD_VER"] = (2, "PfsFiberNorms datamodel version")
        header["HIERARCH PFS.HASH.FIBERNORMS"] = (self.hash, "Hash of this fiberNorms")
        fits.append(PrimaryHDU(header=header))

        header = Header()
        header["INHERIT"] = True
        fits.append(ImageHDU(self.fiberId.astype(int), name="FIBERID", header=header))
        fits.append(ImageHDU(self.wavelength.astype(float), name="WAVELENGTH", header=header))
        fits.append(ImageHDU(self.values.astype(float), name="VALUES", header=header))
        fits.append(
            BinTableHDU.from_columns(
                [
                    Column(name="SPECTROGRAPH", format="I", array=list(self.fiberProfilesHash.keys())),
                    Column(name="HASH", format="K", array=list(self.fiberProfilesHash.values())),
                ],
                name="FIBERPROFILESHASH",
                header=header,
            )
        )
        model = self.model.copy()
        model.header["EXTNAME"] = "MODEL"
        fits.append(model)

    def writeFits(self, filename: str):
        """Write to FITS file

        This API is intended for use by the LSST data butler, which handles
        translating the desired identity into a filename.

        Parameters
        ----------
        filename : `str`
            Filename of FITS file.
        """
        from astropy.io.fits import HDUList

        fits = HDUList()
        self._writeImpl(fits)
        with open(filename, "wb") as fd:
            fits.writeto(fd)

    def toPfsFiberArraySet(self) -> PfsFiberArraySet:
        """Convert to PfsFiberArraySet

        The converted object won't be ideal, but it could be useful for some
        purposes (e.g., plotting).

        Returns
        -------
        fiberArraySet : `PfsFiberArraySet`
            Equivalent PfsFiberArraySet.
        """
        flags = MaskHelper(NO_DATA=0)
        return PfsFiberArraySet(
            self.identity,
            self.fiberId,
            self.wavelength,
            self.values,
            np.where(np.isfinite(self.values), 0, flags.get("NO_DATA")),
            np.zeros_like(self.values, dtype=np.float32),
            np.ones_like(self.values, dtype=np.float32),
            np.zeros((self.numFibers, 3, self.height), dtype=np.float32),
            flags,
            self.metadata,
        )
