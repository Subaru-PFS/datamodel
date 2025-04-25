import os
import numpy as np

from .notes import makeNotesClass, Notes
from .pfsFiberArray import PfsFiberArray
from .fluxTable import FluxTable
from .pfsTable import PfsTable, Column
from .utils import inheritDocstrings
from .utils import astropyHeaderToDict, astropyHeaderFromDict
from .utils import calculatePfsVisitHash, wraparoundNVisit
from .masks import MaskHelper
from .observations import Observations

GA_DAMD_VER = 2

__all__ = [
    "VelocityCorrections",
    "StellarParams",
    "Abundances",
    "GAFluxTable",
    "PfsGAObjectNotes",
    "PfsGAObject",
    "GACatalogTable",
    "PfsGACatalogNotes",
    "PfsGACatalog",
]


class VelocityCorrections(PfsTable):
    """A table of velocity corrections applied to the individual visits."""

    damdVer = GA_DAMD_VER
    schema = [
        Column("visit", np.int32, "ID of the visit these corrections apply for", -1),
        Column("JD", np.float32, "Julian date of the visit", -1),
        Column("helio", np.float32, "Heliocentric correction", np.nan),
        Column("bary", np.float32, "Barycentric correction", np.nan),
    ]
    fitsExtName = 'VELCORR'


class StellarParams(PfsTable):
    """List of measured stellar parameters for a target."""

    damdVer = GA_DAMD_VER
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

    damdVer = GA_DAMD_VER
    schema = [
        Column("method", str, "Abundance measurement method", ""),
        Column("element", str, "Chemical element the abundance is measured for", ""),
        Column("covarId", np.uint8, "Param position within covariance matrix", -1),
        Column("value", np.float32, "Abundance value", np.nan),
        Column("valueErr", np.float32, "Abundance error", np.nan),
        # TODO: will we have systematic errors?
        Column("flag", bool, "Measurement flag (true means bad)", False),
        Column("status", str, "Measurement flags", ""),
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
        header['DAMD_VER'] = (GA_DAMD_VER, "GAFluxTable datamodel version")
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
        super().__init__(target, observations, wavelength, flux, mask, sky,
                         covar, covar2, flags, metadata=metadata, fluxTable=fluxTable, notes=notes)

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

        data["velocityCorrections"] = VelocityCorrections.readHdu(fits)
        data["stellarParams"] = StellarParams.readHdu(fits)
        if cls.StellarParamsFitsExtName in fits:
            data["paramsCovar"] = fits[cls.StellarParamsFitsExtName].data.astype(np.float32)
        data["abundances"] = Abundances.readHdu(fits)
        if cls.AbundancesFitsExtName in fits:
            data["abundCovar"] = fits[cls.AbundancesFitsExtName].data.astype(np.float32)

        return data

    def _writeImpl(self, fits):
        from astropy.io.fits import ImageHDU

        header = super()._writeImpl(fits)

        if self.velocityCorrections is not None:
            self.velocityCorrections.writeHdu(fits)
        if self.stellarParams is not None:
            self.stellarParams.writeHdu(fits)
        if self.paramsCovar is not None:
            fits.append(ImageHDU(self.paramsCovar.astype(np.float32),
                        header=header,
                        name=self.StellarParamsFitsExtName))
        if self.abundances is not None:
            self.abundances.writeHdu(fits)
        if self.abundCovar is not None:
            fits.append(ImageHDU(self.abundCovar.astype(np.float32),
                        header=header,
                        name=self.AbundancesFitsExtName))

        return header


class GACatalogTable(PfsTable):
    """Catalog of GA objects with associated parameters."""

    damdVer = GA_DAMD_VER
    schema = [
        Column("catId", np.int32, "PFS catalog identifier", -1),
        Column("objId", np.int32, "Object identifier.", -1),
        Column("gaiaId", np.int64, "GAIA identifier", -1),
        Column("ps1Id", np.int64, "PS1 identifier", -1),
        Column("hscId", np.int64, "HSC identifier", -1),
        Column("miscId", np.int64, "Miscellaneous identifier", -1),
        Column("ra", np.float32, "Right ascension ICRS [deg]", np.nan),
        Column("dec", np.float32, "Declination ICRS [deg]", np.nan),
        Column("epoch", str, "Coordinate epoch [Jyr]", ''),
        Column("pmRa", np.float32, "Proper motion pmracosdec [mas/yr]", np.nan),
        Column("pmDec", np.float32, "Proper motion dec [mas/yr]", np.nan),
        Column("parallax", np.float32, "Parallax [mas]", np.nan),
        Column("targetType", np.int16, "Target type.", -1),
        Column("proposalId", str, "Proposal ID", ""),
        Column("obCode", str, "Observing Block", ""),

        Column("nVisit_b", np.int16, "Number of visits in B", -1),
        Column("nVisit_m", np.int16, "Number of visits in M", -1),
        Column("nVisit_r", np.int16, "Number of visits in R", -1),
        Column("nVisit_n", np.int16, "Number of visits in N", -1),

        Column("expTimeEff_b", np.float32, "Effective exposure time in B [s]", np.nan),
        Column("expTimeEff_m", np.float32, "Effective exposure time in M [s]", np.nan),
        Column("expTimeEff_r", np.float32, "Effective exposure time in R [s]", np.nan),
        Column("expTimeEff_n", np.float32, "Effective exposure time in N [s]", np.nan),

        # TODO: add SNR

        Column("v_los", np.float32, "Radial velocity [km/s]", np.nan),
        Column("v_losErr", np.float32, "Radial velocity error [km/s]", np.nan),
        Column("T_eff", np.float32, "Effective temperature [K]", np.nan),
        Column("T_effErr", np.float32, "Effective temperature error [K]", np.nan),
        Column("M_H", np.float32, "Metallicity [dex]", np.nan),
        Column("M_HErr", np.float32, "Metallicity error [dex]", np.nan),
        Column("log_g", np.float32, "log g", np.nan),
        Column("log_gErr", np.float32, "log g error", np.nan),

        Column("flag", bool, "Measurement flag (true means bad)", False),
        Column("status", str, "Measurement flags", ""),
    ]
    fitsExtName = 'GACATALOG'


PfsGACatalogNotes = makeNotesClass(
    "PfsGACatalogNotes",
    []
)


class PfsGACatalog():
    filenameFormat = ("pfsGACatalog-%(catId)05d-%(nVisit)03d-0x%(pfsVisitHash)016x.fits")
    filenameRegex = r"^pfsGACatalog-(\d{5})-([0-9]{3})-0x([0-9a-f]{16})\.fits.*$"
    filenameKeys = [("catId", int), ("nVisit", int), ("pfsVisitHash", int)]

    NotesClass = PfsGACatalogNotes

    def __init__(
            self,
            catId,
            observations: Observations,
            catalog: GACatalogTable,
            metadata=None,
            notes: Notes = None):

        self.catId = catId
        self.observations = observations
        self.nVisit = wraparoundNVisit(len(observations.visit))
        self.pfsVisitHash = calculatePfsVisitHash(observations.visit)
        self.catalog = catalog
        self.metadata = metadata if metadata is not None else {}
        self.notes = notes if notes is not None else self.NotesClass()
        self.length = len(catalog)

        self.validate()

    def getIdentity(self):
        """Return the identity of the catalog

        Returns
        -------
        identity : `dict`
            Key-value pairs that identify this catalog
        """
        identity = dict(
            catId=self.catId,
            visit=self.observations.visit,
            nVisit=self.nVisit,
            pfsVisitHash=self.pfsVisitHash
        )
        return identity

    def validate(self):

        # TODO: Make sure catId is the same for each object

        pass

    def __len__(self):
        """Return the length of the arrays"""
        return self.length

    @classmethod
    def _readImpl(cls, fits):
        data = {}

        version = fits[0].header["DAMD_VER"]

        if version >= GA_DAMD_VER:
            data["notes"] = cls.NotesClass.readFits(fits)

        try:
            observations = Observations.fromFits(fits)
            data['observations'] = observations
        except KeyError as exc:
            # Only want to catch "Extension XXX not found."
            if not exc.args[0].startswith("Extension"):
                raise
            data['observations'] = None

        try:
            catalog = GACatalogTable.readHdu(fits)
            data['catalog'] = catalog
            data['catId'] = catalog.catId[0]
        except KeyError as exc:
            # Only want to catch "Extension XXX not found."
            if not exc.args[0].startswith("Extension"):
                raise
            data['catalog'] = None

        return data

    @classmethod
    def readFits(cls, filename):
        """
        Read from FITS file

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
    def read(cls, catId, pfsVisitHash, dirName="."):
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
        filename = os.path.join(dirName, cls.filenameFormat % (catId, pfsVisitHash))
        return cls.readFits(filename)

    def _writeImpl(self, fits):
        """
        Implementation for writing to FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            List of FITS HDUs. This has a Primary HDU already, the header of
            which may be supplemented with additional keywords.
        """

        from astropy.io.fits import Header

        header = Header()

        header['DAMD_VER'] = GA_DAMD_VER

        if self.observations is not None:
            # Override catId from class
            self.observations.catId = np.array(self.observations.num * [self.catId])
            self.observations.toFits(fits)
        if self.catalog is not None:
            self.catalog.writeHdu(fits)
        if self.metadata is not None:
            header.extend(astropyHeaderFromDict(self.metadata))
        if self.notes is not None:
            self.notes.writeFits(fits)

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
        from astropy.io.fits import HDUList
        fits = HDUList()
        header = self._writeImpl(fits)
        fits[0].header.update(header)

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
