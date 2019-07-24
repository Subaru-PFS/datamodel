import numpy as np

from .utils import astropyHeaderToDict, astropyHeaderFromDict
from .masks import MaskHelper
from .interpolate import interpolateFlux, interpolateMask

__all__ = ["FluxTable"]


class FluxTable:
    """Table of coadded fluxes at near-original sampling

    Merged and coadded spectra have been resampled to a standard wavelength
    sampling. This representation provides coadded fluxes at approximately the
    native wavelength sampling, for those that want the data with a minimum of
    resampling. This is mostly of use for single exposures and coadds made from
    back-to-back exposures with the same top-end configuration. For coadds made
    from exposures with different top-end configurations, the different
    wavelength samplings obtained from the different fibers means there's no
    single native wavelength sampling, and so this is less useful.

    This is like a `pfs.datamodel.PfsSimpleSpectrum`, except that it includes a
    variance array, and is written to a FITS HDU rather than a file (so it can
    be incorporated within a `pfs.datamodel.PfsSpectrum`).

    Parameters
    ----------
    wavelength : `numpy.ndarray` of `float`
        Array of wavelengths.
    flux : `numpy.ndarray` of `float`
        Array of fluxes.
    error : `numpy.ndarray` of `float`
        Array of flux errors.
    mask : `numpy.ndarray` of `int`
        Array of mask pixels.
    flags : `pfs.datamodel.MaskHelper`
        Helper for dealing with symbolic names for mask values.
    """
    _hduName = "FLUX_TABLE"  # HDU name to use

    def __init__(self, wavelength, flux, error, mask, flags):
        dims = np.array([len(wavelength.shape), len(flux.shape), len(error.shape), len(mask.shape)])
        lengths = set([wavelength.shape, flux.shape, error.shape, mask.shape])
        if np.any(dims != 1) or len(lengths) > 1:
            raise RuntimeError("Bad shapes for wavelength,flux,error,mask: %s,%s,%s,%s" %
                               (wavelength.shape, flux.shape, error.shape, mask.shape))
        self.wavelength = wavelength
        self.flux = flux
        self.error = error
        self.mask = mask
        self.flags = flags

    def __len__(self):
        """Return number of elements"""
        return len(self.wavelength)

    def plot(self, ignoreFlags=None, show=True):
        """Plot the object spectrum

        Parameters
        ----------
        ignorePixelMask : `int`
            Mask to apply to flux pixels.
        show : `bool`, optional
            Show the plot and block on the window?

        Returns
        -------
        figure : `matplotlib.Figure`
            Figure containing the plot.
        axes : `matplotlib.Axes`
            Axes containing the plot.
        """
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots()
        good = (((self.mask & self.flags.get(*ignoreFlags)) == 0) if ignoreFlags is not None else
                np.ones_like(self.mask, dtype=bool))
        axes.plot(self.wavelength[good], self.flux[good], 'k-', label="Flux")
        axes.set_xlabel("Wavelength (nm)")
        axes.set_ylabel("Flux (nJy)")
        if show:
            plt.show()
        return figure, axes

    def toFits(self, fits):
        """Write to a FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            Opened FITS file.
        """
        from astropy.io.fits import BinTableHDU, Column
        header = self.flags.toFitsHeader()
        hdu = BinTableHDU.from_columns([
            Column("wavelength", "E", array=self.wavelength),
            Column("flux", "E", array=self.flux),
            Column("error", "E", array=self.error),
            Column("mask", "K", array=self.mask),
        ], header=astropyHeaderFromDict(header), name=self._hduName)
        fits.append(hdu)

    @classmethod
    def fromFits(cls, fits):
        """Construct from a FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            Opened FITS file.

        Returns
        -------
        self : `FluxTable`
            Constructed `FluxTable`.
        """
        hdu = fits[cls._hduName]
        header = astropyHeaderToDict(hdu.header)
        flags = MaskHelper.fromFitsHeader(header)
        return cls(hdu.data["wavelength"], hdu.data["flux"], hdu.data["error"], hdu.data["mask"], flags)

    def resample(self, wavelength):
        """Resample to a common wavelength vector

        This is provided as a possible convenience to the user and a means to
        facilitate testing.

        Parameters
        ----------
        wavelength : `numpy.ndarray` of `float`
            New wavelength values (nm).

        Returns
        -------
        resampled : `FluxTable`
            Resampled flux table.
        """
        flags = self.flags.copy()
        flags.add("NO_DATA")

        flux = interpolateFlux(self.wavelength, self.flux, wavelength)
        error = interpolateFlux(self.wavelength, self.error, wavelength)
        mask = interpolateMask(self.wavelength, self.mask, wavelength,
                               fill=flags["NO_DATA"]).astype(self.mask.dtype)
        return type(self)(wavelength, flux, error, mask, flags)
