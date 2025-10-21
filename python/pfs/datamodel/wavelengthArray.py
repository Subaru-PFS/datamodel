import numpy as np
import astropy.wcs
import astropy.units

__all__ = ["WavelengthArray"]


class WavelengthArray(np.ndarray):
    """An array of wavelengths

    This subclass of `numpy.ndarray` keeps track of the construction parameters
    and uses them to create a FITS header with WCS. Specifying the wavelength
    in this way allows persisting the wavelength as a small handful of header
    keywords instead of thousands of values in a predictable series.

    This is a functional numpy array, although it is read-only to prevent the
    construction parameters from getting out of sync with the array values. If
    you find the need to change the values, ``copy()`` this array first.

    Parameters
    ----------
    minWavelength : `float`
        Minimum wavelength (nm).
    maxWavelength : `float`
        Maximum wavelength (nm).
    length : `int`
        Number of values.
    dtype : `numpy.dtype`, optional
        Data type.
    """
    def __new__(cls, minWavelength, maxWavelength, length, dtype=np.float64):
        obj = np.linspace(minWavelength, maxWavelength, length, dtype=dtype).view(cls)
        obj.minWavelength = minWavelength
        obj.maxWavelength = maxWavelength
        obj.flags.writeable = False
        return obj

    def __array_finalize__(self, obj):
        """Numpy mechanics to create array"""
        if obj is None:
            return
        self.minWavelength = getattr(obj, "minWavelength", None)
        self.maxWavelength = getattr(obj, "maxWavelength", None)

    def __repr__(self):
        """String representation"""
        return (f"{type(self).__name__}({self.minWavelength}, {self.maxWavelength}, "
                f"{len(self)}, {self.dtype})")

    def toFitsHeader(self):
        """Convert to a FITS header

        Returns
        -------
        header : `astropy.io.fits.Header`
            FITS header with WCS specifying wavelength array.
        """
        wcs = astropy.wcs.WCS(naxis=1)
        wcs.wcs.ctype = ["WAVE"]
        wcs.wcs.cname = ["Wavelength"]
        wcs.wcs.crval = [0.5*(self.minWavelength + self.maxWavelength)]
        wcs.wcs.crpix = [0.5*len(self) + 1.5]  # FITS is unit-indexed
        wcs.wcs.cunit = [astropy.units.nm]
        dWavelength = (self.maxWavelength - self.minWavelength)/(len(self) - 1)
        wcs.wcs.cdelt = [dWavelength]
        return wcs.to_header()

    @classmethod
    def fromFitsHeader(cls, header, length, dtype=np.float64):
        """Construct from a FITS header

        Parameters
        ----------
        header : `astropy.io.fits.Header`
            FITS header with WCS specifying wavelength array.
        length : `int`
            Length of the array.
        dtype : `numpy.dtype`
            Array data type.

        Returns
        -------
        self : cls
            Constructed wavelength array.
        """
        wcs = astropy.wcs.WCS(header=header)
        wcs = wcs.sub([astropy.wcs.WCSSUB_SPECTRAL])
        if wcs.wcs.ctype[0] != "WAVE":
            raise RuntimeError(f"Unexpected CTYPE in header: {wcs.wcs.ctype}")
        minWavelength = wcs.pixel_to_world(1.0).to(astropy.units.nm).value
        maxWavelength = wcs.pixel_to_world(length).to(astropy.units.nm).value
        return cls(minWavelength, maxWavelength, length, dtype=dtype)
