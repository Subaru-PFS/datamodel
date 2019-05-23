import os

import numpy as np

from .masks import MaskHelper
from .target import TargetData, TargetObservations
from .utils import astropyHeaderFromDict, wraparoundNVisit

__all__ = ["PfsSimpleSpectrum", "PfsSpectrum"]


class PfsSimpleSpectrum:
    """Spectrum for a single object

    This base class is suitable for model spectra which have not been extracted
    from observations.

    Parameters
    ----------
    target : `pfs.datamodel.TargetData`
        Target information.
    wavelength : `numpy.ndarray` of `float`
        Array of wavelengths.
    flux : `numpy.ndarray` of `float`
        Array of fluxes.
    mask : `numpy.ndarray` of `int`
        Array of mask pixels.
    flags : `pfs.datamodel.MaskHelper`
        Helper for dealing with symbolic names for mask values.
    """
    filenameFormat = None  # Subclasses should override

    def __init__(self, target, wavelength, flux, mask, flags):
        self.target = target
        self.wavelength = wavelength
        self.flux = flux
        self.mask = mask
        self.flags = flags

        self.length = len(wavelength)
        self.validate()

    def validate(self):
        """Validate that all the arrays are of the expected shape"""
        assert self.wavelength.shape == (self.length,)
        assert self.flux.shape == (self.length,)
        assert self.mask.shape == (self.length,)

    def __imul__(self, rhs):
        """Flux multiplication, in-place"""
        self.flux *= rhs
        return self

    def __itruediv__(self, rhs):
        """Flux division, in-place"""
        self.flux /= rhs
        return self

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
        for col in ("wavelength", "flux", "mask"):
            data[col] = fits["FLUXTBL"].data[col]
        data["flags"] = MaskHelper.fromFitsHeader(fits["FLUXTBL"].header)
        data["target"] = TargetData.fromFits(fits)
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

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            List of FITS HDUs. This has a Primary HDU already, the header of
            which may be supplemented with additional keywords.
        """
        from astropy.io.fits import BinTableHDU, Column
        fits.append(BinTableHDU.from_columns([
            Column("wavelength", "D", array=self.wavelength),
            Column("flux", "D", array=self.flux),
            Column("mask", "K", array=self.mask),
        ], header=astropyHeaderFromDict(self.flags.toFitsHeader()), name="FLUXTBL"))
        self.target.toFits(fits)

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
        with open(filename, "w") as fd:
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

    def plot(self, ignorePixelMask=0x0, show=True):
        """Plot the object spectrum

        Parameters
        ----------
        ignorePixelMask : `int`
            Mask to apply to flux pixels.
        show : `bool`, optional
            Show the plot?

        Returns
        -------
        figure : `matplotlib.Figure`
            Figure containing the plot.
        axes : `matplotlib.Axes`
            Axes containing the plot.
        """
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots()
        good = (self.mask & ignorePixelMask) == 0
        axes.plot(self.wavelength[good], self.flux[good], 'k-', label="Flux")
        axes.set_xlabel("Wavelength (nm)")
        axes.set_ylabel("Flux (nJy)")
        axes.set_title(str(self.getIdentity()))
        if show:
            figure.show()
        return figure, axes


class PfsSpectrum(PfsSimpleSpectrum):
    """Spectrum for a single object

    This base class is suitable for spectra which have been extracted from
    observations.

    Parameters
    ----------
    target : `pfs.datamodel.target.TargetData`
        Target information.
    observations : `pfs.datamodel.target.TargetObservations`
        Observations of the target.
    wavelength : `numpy.ndarray` of `float`
        Array of wavelengths.
    flux : `numpy.ndarray` of `float`
        Array of fluxes.
    mask : `numpy.ndarray` of `int`
        Array of mask pixels.
    sky : `numpy.ndarray` of `float`
        Array of sky values.
    covar : `numpy.ndarray` of `float`
        Near-diagonal (diagonal and either side) part of the covariance matrix.
    covar2 : `numpy.ndarray` of `float`
        Low-resolution non-sparse covariance estimate.
    flags : `MaskHelper`
        Helper for dealing with symbolic names for mask values.
    """
    filenameFormat = None  # Subclasses should override

    def __init__(self, target, observations, wavelength, flux, mask, sky, covar, covar2, flags):
        self.observations = observations
        self.sky = sky
        self.covar = covar
        self.covar2 = covar2
        self.nVisit = wraparoundNVisit(len(self.observations))
        super().__init__(target, wavelength, flux, mask, flags)

    @property
    def variance(self):
        """Variance in the flux"""
        return self.covar[0]

    def getIdentity(self):
        """Return the identity of the spectrum

        Returns
        -------
        identity : `dict`
            Key-value pairs that identify this spectrum.
        """
        identity = super().getIdentity().copy()
        identity.update(self.observations.getIdentity())
        return identity

    def validate(self):
        """Validate that all the arrays are of the expected shape"""
        self.observations.validate()
        assert wraparoundNVisit(len(self.observations))== self.nVisit
        assert self.sky.shape == (self.length,)
        assert self.covar.shape == (3, self.length)
        assert self.covar2.ndim == 2

    def __imul__(self, rhs):
        """Flux multiplication, in-place"""
        super().__imul__(rhs)
        self.covar *= rhs
        return self

    def __itruediv__(self, rhs):
        """Flux division, in-place"""
        super().__itruediv__(rhs)
        for ii in range(3):
            self.covar[ii] /= rhs
        return self

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
        data = super()._readImpl(fits)
        data["sky"] = fits["SKY"].data
        data["observations"] = TargetObservations.fromFits(fits)
        data["covar"] = fits["COVAR"].data
        data["covar2"] = fits["COVAR2"].data
        return data

    def _writeImpl(self, fits):
        """Implementation for writing to FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            List of FITS HDUs. This has a Primary HDU already, the header of
            which may be supplemented with additional keywords.
        """
        from astropy.io.fits import ImageHDU
        super()._writeImpl(fits)
        fits.append(ImageHDU(self.sky, name="SKY"))
        self.observations.toFits(fits)
        fits.append(ImageHDU(self.covar, name="COVAR"))
        fits.append(ImageHDU(self.covar2, name="COVAR2"))

    def plot(self, plotSky=True, plotErrors=True, ignorePixelMask=0x0, show=True):
        """Plot the object spectrum

        Parameters
        ----------
        plotSky : `bool`
            Plot sky measurements?
        plotErrors : `bool`
            Plot flux errors?
        ignorePixelMask : `int`
            Mask to apply to flux pixels.
        show : `bool`, optional
            Show the plot?

        Returns
        -------
        figure : `matplotlib.Figure`
            Figure containing the plot.
        axes : `matplotlib.Axes`
            Axes containing the plot.
        """
        figure, axes = super().plot(ignorePixelMask=ignorePixelMask, show=False)
        good = (self.mask & ignorePixelMask) == 0
        if plotSky:
            axes.plot(self.wavelength[good], self.sky[good], 'r-', label="Sky")
        if plotErrors:
            axes.plot(self.wavelength[good], np.sqrt(self.variance[good]), 'b-', label="Flux errors")
        if show:
            figure.show()
        return figure, axes
