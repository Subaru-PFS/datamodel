from typing import Type
import numpy as np

from .pfsSimpleSpectrum import PfsSimpleSpectrum
from .utils import wraparoundNVisit, inheritDocstrings
from .fluxTable import FluxTable
from .notes import Notes
from .observations import Observations


@inheritDocstrings
class PfsFiberArray(PfsSimpleSpectrum):
    """Spectrum arrays for a single object

    This base class is suitable for spectra which have been extracted from
    observations.

    Parameters
    ----------
    target : `pfs.datamodel.Target`
        Target information.
    observations : `pfs.datamodel.Observations`
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
    metadata : `dict` (`str`: POD), optional
        Keyword-value pairs for the header.
    fluxTable : `pfs.datamodel.FluxTable`, optional
        Table of fluxes from contributing observations.
    notes : `Notes`, optional
        Reduction notes.
    """
    filenameFormat = None  # Subclasses should override
    NotesClass: Type[Notes]  # Subclasses should override
    FluxTableClass: Type[FluxTable] = FluxTable  # Subclasses may override

    def __init__(
        self,
        target,
        observations,
        wavelength,
        flux,
        mask,
        sky,
        covar,
        covar2,
        flags,
        metadata=None,
        fluxTable=None,
        notes: Notes = None,
    ):
        self.observations = observations
        self.sky = sky
        self.covar = covar
        self.covar2 = covar2
        self.nVisit = wraparoundNVisit(len(self.observations))
        self.fluxTable = fluxTable
        self.notes = notes if notes is not None else self.NotesClass()
        super().__init__(target, wavelength, flux, mask, flags, metadata=metadata)

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
        assert wraparoundNVisit(len(self.observations)) == self.nVisit
        assert self.sky.shape == (self.length,)
        assert self.covar.shape == (3, self.length)
        assert self.covar2.ndim == 2

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
        data["sky"] = fits["SKY"].data.astype(np.float32)
        data["observations"] = Observations.fromFits(fits)
        data["covar"] = fits["COVAR"].data.astype(np.float32)
        data["covar2"] = fits["COVAR2"].data.astype(np.float32)
        try:
            fluxTable = cls.FluxTableClass.fromFits(fits)
        except KeyError as exc:
            # Only want to catch "Extension XXX not found."
            if not exc.args[0].startswith("Extension"):
                raise
            fluxTable = None
        data["fluxTable"] = fluxTable

        version = fits[1].header["DAMD_VER"]
        if version >= 2:
            data["notes"] = cls.NotesClass.readFits(fits)

        return data

    def _writeImpl(self, fits):
        """Implementation for writing to FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            List of FITS HDUs. This has a Primary HDU already, the header of
            which may be supplemented with additional keywords.
        """
        # NOTE: When making any changes to this method that modify the output
        # format, increment the DAMD_VER header value in the
        # PfsSimpleSpectrum._writeImpl method, and record the change in
        # the versions.txt file.
        from astropy.io.fits import ImageHDU
        header = super()._writeImpl(fits)
        header["DAMD_VER"] = (2, "PfsFiberArray datamodel version")
        fits.append(ImageHDU(self.sky.astype(np.float32), header=header, name="SKY"))
        fits.append(ImageHDU(self.covar.astype(np.float32), header=header, name="COVAR"))
        fits.append(ImageHDU(self.covar2.astype(np.float32), name="COVAR2"))
        self.observations.toFits(fits)
        self.notes.writeFits(fits)
        if self.fluxTable is not None:
            self.fluxTable.toFits(fits)
        return header