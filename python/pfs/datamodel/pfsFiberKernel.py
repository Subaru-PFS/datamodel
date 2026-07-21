from __future__ import annotations

from importlib import metadata
from typing import Any, Dict, Optional

import numpy as np
import astropy.io.fits

from .utils import astropyHeaderFromDict, astropyHeaderToDict

__all__ = ("PfsFiberKernel",)


class PfsFiberKernel:
    """Fiber kernel

    Applicable to a single spectrograph arm, used to convolve the fiber traces.

    Parameters
    ----------
    imageWidth, imageHeight : `int`
        Dimensions of the image for which the kernel is defined.
    halfWidth : `int`
        Half-width of the kernel.
    xNumBlocks, yNumBlocks : `int`
        Number of blocks in the x and y directions.
    values : `numpy.ndarray`
        Values of the kernel.
    metadata : `dict` (`str`: POD), optional
        Keyword-value pairs for the header.
    """

    hduName = "FIBERKERNEL"  # name of the HDU in the FITS file where the data is stored

    def __init__(
        self,
        imageWidth: int,
        imageHeight: int,
        halfWidth: int,
        xNumBlocks: int,
        yNumBlocks: int,
        values: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.halfWidth = halfWidth
        self.xNumBlocks = xNumBlocks
        self.yNumBlocks = yNumBlocks
        self.values = values
        self.metadata = metadata if metadata is not None else {}

        self.numParams = 2*halfWidth*xNumBlocks*yNumBlocks

        self.validate()

    def validate(self):
        """Validate that all the arrays are of the expected shape"""
        assert self.values.shape == (self.numParams,)

    def __eq__(self, other):
        """Compare for equality"""
        for attr in ("imageWidth", "imageHeight", "halfWidth", "xNumBlocks", "yNumBlocks"):
            if getattr(self, attr) != getattr(other, attr):
                return False
        for attr in ("values",):
            if not np.array_equal(getattr(self, attr), getattr(other, attr)):
                return False
        # Not comparing metadata
        return True

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
            Keyword arguments for constructing PfsFiberKernel.
        """
        data: Dict[str, Any] = {}
        data["imageWidth"] = fits[cls.hduName].data["imageWidth"][0]
        data["imageHeight"] = fits[cls.hduName].data["imageHeight"][0]
        data["halfWidth"] = fits[cls.hduName].data["halfWidth"][0]
        data["xNumBlocks"] = fits[cls.hduName].data["xNumBlocks"][0]
        data["yNumBlocks"] = fits[cls.hduName].data["yNumBlocks"][0]
        data["values"] = fits[cls.hduName].data["values"][0]
        data["metadata"] = astropyHeaderToDict(fits[0].header)
        return data

    @classmethod
    def readFits(cls, filename: str) -> "PfsFiberKernel":
        """Read from FITS file

        This API is intended for use by the LSST data butler, which knows the
        filename.

        Parameters
        ----------
        filename : `str`
            Filename of FITS file.

        Returns
        -------
        self : ``cls``
            Constructed instance, from FITS file.
        """
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
        from astropy.io.fits import BinTableHDU, Column, PrimaryHDU

        if self.metadata:
            header = astropyHeaderFromDict(self.metadata)
        else:
            header = astropy.io.fits.Header()
        header["DAMD_VER"] = (1, "PfsFiberKernel datamodel version")
        fits.append(PrimaryHDU(header=header))

        table = BinTableHDU.from_columns([
            Column(name="imageWidth", format="I", array=np.array([self.imageWidth], dtype=int)),
            Column(name="imageHeight", format="I", array=np.array([self.imageHeight], dtype=int)),
            Column(name="halfWidth", format="I", array=np.array([self.halfWidth], dtype=int)),
            Column(name="xNumBlocks", format="I", array=np.array([self.xNumBlocks], dtype=int)),
            Column(name="yNumBlocks", format="I", array=np.array([self.yNumBlocks], dtype=int)),
            Column(name="values", format="PD()", array=[self.values])
        ], name=self.hduName)
        fits.append(table)

    def writeFits(self, filename: str):
        """Write to FITS file

        This API is intended for use by the LSST data butler, which chooses the
        filename.

        Parameters
        ----------
        filename : `str`
            Filename of FITS file.
        """
        fits = astropy.io.fits.HDUList()
        self._writeImpl(fits)
        with open(filename, "wb") as fd:
            fits.writeto(fd)
