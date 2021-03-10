import os

import numpy as np

from .masks import MaskHelper
from .target import Target
from .utils import astropyHeaderFromDict, inheritDocstrings
from .wavelengthArray import WavelengthArray

__all__ = ["PfsSimpleSpectrum"]


@inheritDocstrings
class PfsSimpleSpectrum:
    """Spectrum for a single object

    This base class is suitable for model spectra which have not been extracted
    from observations.

    Parameters
    ----------
    target : `pfs.datamodel.Target`
        Target information.
    wavelength : `numpy.ndarray` of `float`
        Array of wavelengths.
    flux : `numpy.ndarray` of `float`
        Array of fluxes.
    mask : `numpy.ndarray` of `int`
        Array of mask pixels.
    flags : `pfs.datamodel.MaskHelper`
        Helper for dealing with symbolic names for mask values.
    metadata : `dict` (`str`: POD), optional
        Keyword-value pairs for the header.
    """
    filenameFormat = None  # Subclasses should override

    def __init__(self, target, wavelength, flux, mask, flags, metadata=None):
        self.target = target
        self.wavelength = wavelength
        self.flux = flux
        self.mask = mask
        self.flags = flags
        self.metadata = metadata if metadata is not None else {}

        self.length = len(wavelength)
        self.validate()

    def validate(self):
        """Validate that all the arrays are of the expected shape"""
        assert self.wavelength.shape == (self.length,)
        assert self.flux.shape == (self.length,)
        assert self.mask.shape == (self.length,)

    def __len__(self):
        """Return the length of the arrays"""
        return self.length

    def getIdentity(self):
        """Return the identity of the spectrum

        Returns
        -------
        identity : `dict`
            Key-value pairs that identify this spectrum.
        """
        return self.target.identity

    @classmethod
    def _readImpl(cls, fits):
        """Implementation for reading from FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            Opened FITS file.

        Returns
        -------
        kwargs : ``dict``
            Keyword arguments for constructing spectrum.
        """
        data = {}
        data["flux"] = fits["FLUX"].data.astype(float)
        data["mask"] = fits["MASK"].data.astype(np.int32)

        # Wavelength can be specified in an explicit extension, or as a WCS in the header
        if "WAVELENGTH" in fits:
            wavelength = fits["WAVELENGTH"].data.astype(float)
        else:
            wavelength = WavelengthArray.fromFitsHeader(fits["FLUX"].header, len(fits["FLUX"].data))
        data["wavelength"] = wavelength

        data["flags"] = MaskHelper.fromFitsHeader(fits["MASK"].header)
        data["target"] = Target.fromFits(fits)
        return data

    @classmethod
    def readFits(cls, filename):
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

    @classmethod
    def read(cls, identity, dirName="."):
        """Read from file

        This API is intended for use by science users, as it allows selection
        of the correct file from parameters that make sense, such as which
        catId, objId, etc.

        Parameters
        ----------
        identity : `dict`
            Keyword-value pairs identifying the data of interest. Common keywords
            include ``catId``, ``tract``, ``patch``, ``objId``.
        dirName : `str`, optional
            Directory from which to read.

        Returns
        -------
        self : ``cls``
            Spectrum read from file.
        """
        filename = os.path.join(dirName, cls.filenameFormat % identity)
        return cls.readFits(filename)

    def _writeImpl(self, fits):
        """Implementation for writing to FITS file

        We attempt to write the wavelength to the header (as a WCS; this results
        in a modest size savings), which works if the wavelength is a specified
        as a `WavelengthArray`; otherwise we write it as an explicit extension.

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            List of FITS HDUs. This has a Primary HDU already, the header of
            which may be supplemented with additional keywords.

        Returns
        -------
        header : `astropy.io.fits.Header`
            FITS headers which may contain the wavelength WCS.
        """
        # NOTE: When making any changes to this method that modify the output
        # format, increment the DAMD_VER header value and record the change in
        # the versions.txt file.
        from astropy.io.fits import ImageHDU, Header
        haveWavelengthHeader = False
        try:
            header = self.wavelength.toFitsHeader()  # For WavelengthArray
            haveWavelengthHeader = True
        except AttributeError:
            header = Header()

        # NOTE: The datamodel version also gets incremented here for the PfsFiberArray
        header['DAMD_VER'] = (1, "PfsSimpleSpectrum datamodel version")

        if self.metadata:
            header.extend(astropyHeaderFromDict(self.metadata))
        fits.append(ImageHDU(self.flux, header=header, name="FLUX"))
        maskHeader = astropyHeaderFromDict(self.flags.toFitsHeader())
        maskHeader.extend(header)
        fits.append(ImageHDU(self.mask, header=maskHeader, name="MASK"))
        if not haveWavelengthHeader:
            fits.append(ImageHDU(self.wavelength, header=header, name="WAVELENGTH"))
        self.target.toFits(fits)
        return header

    def writeFits(self, filename):
        """Write to FITS file

        This API is intended for use by the LSST data butler, which handles
        translating the desired identity into a filename.

        Parameters
        ----------
        filename : `str`
            Filename of FITS file.
        """
        from astropy.io.fits import HDUList, PrimaryHDU
        fits = HDUList()
        fits.append(PrimaryHDU())
        self._writeImpl(fits)
        with open(filename, "wb") as fd:
            fits.writeto(fd)

    def write(self, dirName="."):
        """Write to file

        This API is intended for use by science users, as it allows setting the
        correct filename from parameters that make sense, such as which
        catId, objId, etc.

        Parameters
        ----------
        dirName : `str`, optional
            Directory to which to write.
        """
        identity = self.getIdentity()
        filename = os.path.join(dirName, self.filenameFormat % identity)
        return self.writeFits(filename)
