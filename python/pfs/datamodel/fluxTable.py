import numpy as np

from .utils import astropyHeaderToDict, astropyHeaderFromDict
from .masks import MaskHelper

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
        self.checkShapes(wavelength=wavelength,
                         flux=flux,
                         error=error,
                         mask=mask)

        self.wavelength = wavelength
        self.flux = flux
        self.error = error
        self.mask = mask
        self.flags = flags

    def __len__(self):
        """Return number of elements"""
        return len(self.wavelength)

    def checkShapes(self, **kwargs):
        keys = list(sorted(kwargs.keys()))
        dims = np.array([len(kwargs[k].shape) for k in keys])
        lengths = set([kwargs[k].shape for k in keys])

        if np.any(dims != 1) or len(lengths) > 1:
            names = ','.join(keys)
            shapes = ','.join([str(kwargs[k].shape) for k in keys])
            raise RuntimeError("Bad shapes for %s: %s" %
                               (names, shapes))

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
        header = astropyHeaderFromDict(self.flags.toFitsHeader())
        header['DAMD_VER'] = (1, "FluxTable datamodel version")
        hdu = BinTableHDU.from_columns([
            Column("wavelength", "D", array=self.wavelength),
            Column("flux", "E", array=self.flux),
            Column("error", "E", array=self.error),
            Column("mask", "K", array=self.mask),
        ], header=header, name=self._hduName)
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
        return cls(hdu.data["wavelength"].astype(float), hdu.data["flux"].astype(np.float32),
                   hdu.data["error"].astype(np.float32), hdu.data["mask"].astype(np.int32),
                   flags)
