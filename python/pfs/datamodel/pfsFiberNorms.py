from typing import Any, Dict, Optional

import numpy as np
import astropy.io.fits

from .utils import astropyHeaderFromDict, astropyHeaderToDict
from .identity import CalibIdentity

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
    height : `int`
        Height of the detector; used for normalizing the polynomial inputs.
    coeff : `numpy.ndarray` of `float`
        Array of coefficients for each fiber.
    metadata : `dict` (`str`: POD), optional
        Keyword-value pairs for the header.
    """

    def __init__(
        self,
        identity: CalibIdentity,
        fiberId: np.ndarray,
        height: int,
        coeff: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.identity = identity
        self.fiberId = fiberId
        self.height = height
        self.coeff = coeff
        self.metadata = metadata if metadata is not None else {}

        self.numFibers = len(fiberId)
        self.numCoeff = coeff.shape[1]
        self._lookup = {fiberId[ii]: ii for ii in range(self.numFibers)}

        self.validate()

    def validate(self):
        """Validate that all the arrays are of the expected shape"""
        assert self.fiberId.shape == (self.numFibers,)
        assert self.coeff.shape == (self.numFibers, self.numCoeff)

    def __len__(self) -> int:
        """Return the number of fibers"""
        return self.numFibers

    def __getitem__(self, fiberId: int) -> np.ndarray:
        """Return the coefficients for a given fiberId"""
        index = self._lookup[fiberId]
        return self.coeff[index]

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
        data: Dict[str, Any] = {}
        data["identity"] = CalibIdentity.fromHeader(fits[0].header)
        data["fiberId"] = fits["FIBERID"].data.astype(int)
        data["height"] = fits[0].header["HEIGHT"]
        data["coeff"] = fits["COEFF"].data.astype(float)

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
        from astropy.io.fits import ImageHDU, PrimaryHDU, Header

        metadata = self.identity.toHeader()
        if self.metadata:
            metadata.update(self.metadata)
        header = astropyHeaderFromDict(metadata)
        header["DAMD_VER"] = (1, "PfsFiberNorms datamodel version")
        header["HEIGHT"] = (self.height, "Height of detector")
        fits.append(PrimaryHDU(header=header))

        header = Header()
        header["INHERIT"] = True
        fits.append(ImageHDU(self.fiberId.astype(int), name="FIBERID", header=header))
        fits.append(ImageHDU(self.coeff.astype(float), name="COEFF", header=header))

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
