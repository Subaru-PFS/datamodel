from typing import Type
import numpy as np

from .notes import makeNotesClass, Notes
from .pfsFiberArray import PfsFiberArray
from .fluxTable import FluxTable
from .pfsTable import PfsTable, Column
from .utils import inheritDocstrings
from .utils import astropyHeaderToDict, astropyHeaderFromDict
from .masks import MaskHelper

__all__ = [
    "VelocityCorrections",
    "StellarParams",
    "Abundances",
    "GAFluxTable",
    "PfsGAObjectNotes",
    "PfsGAObject",
]

class VelocityCorrections(PfsTable):
    """A table of velocity corrections applied to the individual visits."""

    damdVer = 2
    schema = [
        Column("visit", np.int32, "ID of the visit these corrections apply for", -1),
        Column("JD", np.float32, "Julian date of the visit", -1),
        Column("helio", np.float32, "Heliocentric correction", np.nan),
        Column("bary", np.float32, "Barycentric correction", np.nan),
    ]
    fitsExtName = 'VELCORR'

class StellarParams(PfsTable):
    """List of measured stellar parameters for a target."""

    damdVer = 2
    schema = [
        Column("method", str, "Line-of-sight velocity measurement method", ""),
        Column("frame", str, "Reference frame of velocity: helio, bary", ""),
        Column("param", str, "Stellar parameter: v_los, M_H, T_eff, log_g, a_M", ""),
        Column("covarId", np.uint8, "Param position within covariance matrix", -1),
        Column("unit", str, "Physical unit of parameter", ""),
        Column("value", np.float32, "Stellar parameter value", np.nan),
        Column("valueErr", np.float32, "Stellar parameter error", np.nan),
        # TODO: add quantiles or similar for MCMC results
        Column("flag", bool, "Measurement flag (true means bad)", False),
        Column("status", str, "Measurement flags", ""),
    ]
    fitsExtName = 'STELLARPARAM'

class Abundances(PfsTable):
    """List of measured abundance parameters for stellar targets."""

    damdVer = 2
    schema = [
        Column("method", str, "Abundance measurement method", ""),
        Column("element", str, "Chemical element the abundance is measured for", ""),
        Column("covarId", np.uint8, "Param position within covariance matrix", -1),
        Column("value", np.float32, "Abundance value", np.nan),
        Column("valueErr", np.float32, "Abundance error", np.nan),
    ]
    fitsExtName = 'ABUND'

class GAFluxTable(FluxTable):
    """Table of coadded fluxes at near-original sampling and model fits

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
    model : `numpy.ndarray` of `float`
        Array of best-fit model flux.
    cont : `numpy.ndarray` of `float`
        Array of continuum model.
    norm_flux : `numpy.ndarray` of `float`
        Array of continuum-normalized flux.
    norm_error : `numpy.ndarray` of `float`
        Array of continuum-normalized flux error.
    norm_model : `numpy.ndarray` of `float`
        Array of continuum-normalized model.
    mask : `numpy.ndarray` of `int`
        Array of mask pixels.
    flags : `pfs.datamodel.MaskHelper`
        Helper for dealing with symbolic names for mask values.
    """
    _hduName = "FLUX_TABLE"  # HDU name to use

    def __init__(self, wavelength, flux, error, model, cont, norm_flux, norm_error, norm_model, mask, flags):
        self.checkShapes(wavelength=wavelength,
                         flux=flux,
                         error=error,
                         model=model,
                         cont=cont,
                         norm_flux=norm_flux,
                         norm_error=norm_error,
                         norm_model=norm_model,
                         mask=mask)
        
        self.wavelength = wavelength
        self.flux = flux
        self.error = error
        self.model = model
        self.cont = cont
        self.norm_flux = norm_flux
        self.norm_error = norm_error
        self.norm_model = norm_model
        self.mask = mask
        self.flags = flags

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
        header['DAMD_VER'] = (1, "GAFluxTable datamodel version")
        hdu = BinTableHDU.from_columns([
            Column("wavelength", "D", array=self.wavelength),
            Column("flux", "E", array=self.flux),
            Column("error", "E", array=self.error),
            Column("model", "E", array=self.model),
            Column("cont", "E", array=self.cont),
            Column("norm_flux", "E", array=self.norm_flux),
            Column("norm_error", "E", array=self.norm_error),
            Column("norm_model", "E", array=self.norm_model),
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
        return cls(hdu.data["wavelength"].astype(float),
                   hdu.data["flux"].astype(np.float32),
                   hdu.data["error"].astype(np.float32),
                   hdu.data["model"].astype(np.float32),
                   hdu.data["cont"].astype(np.float32),
                   hdu.data["norm_flux"].astype(np.float32),
                   hdu.data["norm_error"].astype(np.float32),
                   hdu.data["norm_model"].astype(np.float32),
                   hdu.data["mask"].astype(np.int32),
                   flags)

PfsGAObjectNotes = makeNotesClass(
    "PfsGAObjectNotes",
    []
)

@inheritDocstrings
class PfsGAObject(PfsFiberArray):
    """Coadded spectrum of a GA target with derived quantities.
    
    Produced by ˙˙gapipe``
    
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
    fluxTable : `pfs.datamodel.GAFluxTable`, optional
        Table of coadded fluxes and continuum-normalized flux from contributing observations.
    stellarParams: `pfs.datamodel.StellarParams`, optional
        Table of measured stellar parameters.
    velocityCorrections: `pfs.datamodel.VelocityCorrections`, optional
        Table of velocity corrections applied to the individual visits.
    abundances: `pfs.datamodel.Abundances`, optional
        Table of measured abundance parameters.
    paramsCovar: `numpy.ndarray` of `float`, optional
        Covariance matrix for stellar parameters.
    abundCovar: `numpy.ndarray` of `float`, optional
        Covariance matrix for abundance parameters.
    notes : `Notes`, optional
        Reduction notes.
    """

    filenameFormat = ("pfsGAObject-%(catId)05d-%(tract)05d-%(patch)s-%(objId)016x"
                      "-%(nVisit)03d-0x%(pfsVisitHash)016x.fits")
    filenameRegex = r"^pfsGAObject-(\d{5})-(\d{5})-(.*)-([0-9a-f]{16})-(\d{3})-0x([0-9a-f]{16})\.fits.*$"
    filenameKeys = [("catId", int), ("tract", int), ("patch", str), ("objId", int),
                    ("nVisit", int), ("pfsVisitHash", int)]
    NotesClass = PfsGAObjectNotes
    FluxTableClass = GAFluxTable

    StellarParamsFitsExtName = "STELLARCOVAR"
    AbundancesFitsExtName = "ABUNDCOVAR"

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
        stellarParams=None,
        velocityCorrections=None,
        abundances=None,
        paramsCovar=None,
        abundCovar=None,
        notes: Notes = None,
    ):
        super().__init__(target, observations, wavelength, flux, mask, sky, covar, covar2, flags, metadata=metadata, fluxTable=fluxTable, notes=notes)

        self.stellarParams = stellarParams
        self.velocityCorrections = velocityCorrections
        self.abundances = abundances
        self.paramsCovar = paramsCovar
        self.abundCovar = abundCovar

    def validate(self):
        """Validate that all the arrays are of the expected shape"""
        super().validate()

        # TODO: write any validation code

    @classmethod
    def _readImpl(cls, fits):
        data = super()._readImpl(fits)

        # TODO: handle missing extensions

        data["stellarParams"] = StellarParams.readHdu(fits)
        data["velocityCorrections"] = VelocityCorrections.readHdu(fits)
        data["abundances"] = Abundances.readHdu(fits)
        if cls.StellarParamsFitsExtName in fits:
            data["paramsCovar"] = fits[cls.StellarParamsFitsExtName].data.astype(np.float32)
        if cls.AbundancesFitsExtName in fits:
            data["abundCovar"] = fits[cls.AbundancesFitsExtName].data.astype(np.float32)

        return data

    def _writeImpl(self, fits):
        from astropy.io.fits import ImageHDU

        header = super()._writeImpl(fits)

        if self.stellarParams is not None:
            self.stellarParams.writeHdu(fits)
        if self.velocityCorrections is not None:
            self.velocityCorrections.writeHdu(fits)
        if self.abundances is not None:
            self.abundances.writeHdu(fits)
        if self.paramsCovar is not None:
            fits.append(ImageHDU(self.paramsCovar.astype(np.float32), header=header, name=self.StellarParamsFitsExtName))
        if self.abundCovar is not None:
            fits.append(ImageHDU(self.abundCovar.astype(np.float32), header=header, name=self.AbundancesFitsExtName))

        return header