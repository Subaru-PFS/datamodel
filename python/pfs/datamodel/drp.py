import numpy as np

from .notes import makeNotesClass
from .pfsSimpleSpectrum import PfsSimpleSpectrum
from .pfsFiberArray import PfsFiberArray
from .pfsFiberArraySet import PfsFiberArraySet
from .pfsTable import PfsTable, PfsTableWithSeparateStrings, Column
from .pfsTargetSpectra import PfsTargetSpectra
from .utils import inheritDocstrings

__all__ = [
    "PfsArmNotes",
    "PfsArm",
    "PfsMergedNotes",
    "PfsMerged",
    "PfsReference",
    "PfsSingleNotes",
    "PfsSingle",
    "PfsObjectNotes",
    "PfsObject",
    "LineMeasurements",
    "LsfLineMeasurements",
    "PfsCalibrated",
    "PfsCoadd",
]


class PfsArmNotes(PfsTable):
    """Notes about spectral reduction for PfsArm"""

    fitsExtName = "NOTES"
    schema = [
        Column("blackSpotId", np.int32, "ID of black spot", -1),
        Column("blackSpotDistance", np.float32, "Distance to nearest black spot center (mm)", np.nan),
        Column("blackSpotCorrection", np.float32, "Black spot flux correction", np.nan),
        Column("barycentricCorrection", np.float32, "Barycentric correction to be applied (km/s)", np.nan),
        Column("pfiCorrection", np.float32, "PFI correction applied", np.nan),
    ]


@inheritDocstrings
class PfsArm(PfsFiberArraySet):
    """Spectra from reducing a single arm

    Produced by ``reduceExposure``.
    """
    filenameFormat = "pfsArm-%(visit)06d-%(arm)1s%(spectrograph)1d.fits"
    filenameRegex = r"^pfsArm-(\d{6})-([brnm])(\d)\.fits.*$"
    filenameKeys = [("visit", int), ("arm", str), ("spectrograph", int)]
    NotesClass = PfsArmNotes


class PfsMergedNotes(PfsTable):
    """Notes about spectral reduction for PfsMerged"""

    fitsExtName = "NOTES"
    schema = [
        Column("blackSpotId", np.int32, "ID of black spot", -1),
        Column("blackSpotDistance", np.float32, "Distance to nearest black spot center (mm)", np.nan),
        Column("blackSpotCorrection", np.float32, "Black spot flux correction", np.nan),
        Column("barycentricCorrection", np.float32, "Barycentric correction applied (km/s)", np.nan),
        Column("pfiCorrection", np.float32, "PFI correction applied", np.nan),
    ]


@inheritDocstrings
class PfsMerged(PfsFiberArraySet):
    """Spectra from merging all arms within an exposure

    Produced by ``mergeArms``.
    """
    filenameFormat = "pfsMerged-%(visit)06d.fits"
    filenameRegex = r"^pfsMerged-(\d{6})\.fits.*$"
    filenameKeys = [("visit", int)]
    NotesClass = PfsMergedNotes


@inheritDocstrings
class PfsReference(PfsSimpleSpectrum):
    """Reference spectrum for flux calibration

    Produced by ``calculateReferenceFlux``.
    """
    filenameFormat = "pfsReference-%(catId)05d-%(tract)05d-%(patch)s-%(objId)016x.fits"
    filenameRegex = r"^pfsReference-(\d{5})-(\d{5})-(.*)-([0-9a-f]{16})\.fits.*$"
    filenameKeys = [("catId", int), ("tract", int), ("patch", str), ("objId", int)]


PfsSingleNotes = makeNotesClass(
    "PfsSingleNotes",
    [],
)


@inheritDocstrings
class PfsSingle(PfsFiberArray):
    """Flux-calibrated, single epoch spectrum

    Produced by ``fluxCalibrate``.
    """
    filenameFormat = "pfsSingle-%(catId)05d-%(tract)05d-%(patch)s-%(objId)016x-%(visit)06d.fits"
    filenameRegex = r"^pfsSingle-(\d{5})-(\d{5})-(.*)-([0-9a-f]{16})-(\d{6})\.fits.*$"
    filenameKeys = [("catId", int), ("tract", int), ("patch", str), ("objId", int), ("visit", int)]
    NotesClass = PfsSingleNotes


PfsObjectNotes = makeNotesClass(
    "PfsObjectNotes",
    []
)


@inheritDocstrings
class PfsObject(PfsFiberArray):
    """Coadded spectrum

    Produced by ``coaddSpectra``.
    """
    filenameFormat = ("pfsObject-%(catId)05d-%(tract)05d-%(patch)s-%(objId)016x"
                      "-%(nVisit)03d-0x%(pfsVisitHash)016x.fits")
    filenameRegex = r"^pfsObject-(\d{5})-(\d{5})-(.*)-([0-9a-f]{16})-(\d{3})-0x([0-9a-f]{16})\.fits.*$"
    filenameKeys = [("catId", int), ("tract", int), ("patch", str), ("objId", int),
                    ("nVisit", int), ("pfsVisitHash", int)]
    NotesClass = PfsObjectNotes


@inheritDocstrings
class LineMeasurements(PfsTableWithSeparateStrings):
    schema = [
        Column("fiberId", np.int32, "Fiber identifier", -1),
        Column("wavelength", np.float64, "Wavelength (nm)", np.nan),
        Column("x", np.float32, "x position (pixels)", np.nan),
        Column("y", np.float32, "y position (pixels)", np.nan),
        Column("xErr", np.float32, "Error in x position (pixels)", np.nan),
        Column("yErr", np.float32, "Error in y position (pixels)", np.nan),
        Column("xx", np.float32, "xx second moment (pixels^2)", np.nan),
        Column("yy", np.float32, "yy second moment (pixels^2)", np.nan),
        Column("xy", np.float32, "xy second moment (pixels^2)", np.nan),
        Column("flux", np.float32, "Measured flux", np.nan),
        Column("fluxErr", np.float32, "Error in measured flux", np.nan),
        Column("fluxNorm", np.float32, "Normalisation flux", np.nan),
        Column("flag", bool, "Measurement flag (true means bad)", True),
        Column("status", np.int32, "Reference line status bitmask", -1),
        Column("description", str, "Reference line description", "UNNOWN"),
        Column("transition", str, "Reference line transition", "UNNOWN"),
        Column("source", np.int32, "Reference line source code", -1),
    ]
    fitsExtName = "ARCLINES"
    aliases = dict(flux=("intensity",), fluxErr=("intensityErr"),)


class LsfLineMeasurements(PfsTableWithSeparateStrings):
    """Per-line fits of the line-spread function to arc lines

    Each row is one arc line in one fibre of one exposure, fit with the
    disk (x) Edgeworth LSF model (the fibre face, a disk of radius ``radius``,
    blurred by a kernel of rms ``rms`` with standardised cumulants ``S``,
    ``K``, ``lam5``, ``lam6``, optionally convolved with a Lorentzian of FWHM
    ``lorentz``; NaN if not fit).
    """
    schema = [
        Column("visit", np.int32, "Exposure identifier", -1),
        Column("fiberId", np.int32, "Fiber identifier", -1),
        Column("wavelength", np.float64, "Reference (catalogue) wavelength (nm)", np.nan),
        Column("status", np.int32, "Reference line status bitmask", -1),
        Column("description", str, "Reference line description", "UNKNOWN"),
        Column("transition", str, "Reference line transition", "UNKNOWN"),
        Column("x", np.float32, "x position on the detector (pixels)", np.nan),
        Column("y", np.float32, "y position on the detector (pixels)", np.nan),
        Column("pixel", np.float64, "Fitted line centre (first moment) in the spectrum (pixels)", np.nan),
        Column("pixelErr", np.float32, "Error in pixel (pixels)", np.nan),
        Column("refPixel", np.float64, "Position of the reference wavelength in the spectrum (pixels)", np.nan),
        Column("dispersion", np.float32, "Dispersion at the line (nm/pixel)", np.nan),
        Column("dataCentroid", np.float32, "First moment of the data about pixel, in a window (pixels)", np.nan),
        Column("modelCentroid", np.float32, "First moment of the model about pixel, same window (pixels)",
               np.nan),
        Column("radius", np.float32, "Fibre disk radius (pixels)", np.nan),
        Column("radiusErr", np.float32, "Error in radius (pixels)", np.nan),
        Column("rms", np.float32, "RMS of the blur kernel (pixels)", np.nan),
        Column("rmsErr", np.float32, "Error in rms (pixels)", np.nan),
        Column("S", np.float32, "Skewness of the blur kernel", np.nan),
        Column("SErr", np.float32, "Error in S", np.nan),
        Column("K", np.float32, "Excess kurtosis of the blur kernel", np.nan),
        Column("KErr", np.float32, "Error in K", np.nan),
        Column("lam5", np.float32, "Fifth standardised cumulant of the blur kernel", np.nan),
        Column("lam5Err", np.float32, "Error in lam5", np.nan),
        Column("lam6", np.float32, "Sixth standardised cumulant of the blur kernel", np.nan),
        Column("lam6Err", np.float32, "Error in lam6", np.nan),
        Column("lorentz", np.float32, "FWHM of the Lorentzian convolved with the profile (pixels)", np.nan),
        Column("lorentzErr", np.float32, "Error in lorentz (pixels)", np.nan),
        Column("haloFrac", np.float32, "Fraction of the light in the halo", np.nan),
        Column("haloFracErr", np.float32, "Error in haloFrac", np.nan),
        Column("haloWidth", np.float32, "RMS of the halo's Gaussian blur (pixels)", np.nan),
        Column("haloWidthErr", np.float32, "Error in haloWidth (pixels)", np.nan),
        Column("flux", np.float32, "Line flux", np.nan),
        Column("fluxErr", np.float32, "Error in flux", np.nan),
        Column("peak", np.float32, "Peak flux of the line's model", np.nan),
        Column("chi2", np.float32, "chi^2 of the fit", np.nan),
        Column("ndof", np.int32, "Degrees of freedom of the fit", 0),
        Column("nBadCore", np.int32, "Number of masked pixels in the line's core", 0),
        Column("flag", np.int32, "Bitmask: how the line was used (see datamodel.txt)", 0),
    ]
    fitsExtName = "LSFLINES"


class PfsCalibratedNotesTable(PfsTable):
    """Table of notes for PfsCalibratedSpectra"""
    schema = PfsSingleNotes.schema
    fitsExtName = "NOTES"


class PfsCalibrated(PfsTargetSpectra):
    """A collection of PfsSingle (calibrated spectra) indexed by target"""
    PfsFiberArrayClass = PfsSingle
    NotesClass = PfsCalibratedNotesTable


class PfsObjectNotesTable(PfsTable):
    """Table of notes for PfsObjectSpectra"""
    schema = PfsObjectNotes.schema
    fitsExtName = "NOTES"


class PfsCoadd(PfsTargetSpectra):
    """A collection of PfsObject (coadded spectra) indexed by target"""
    PfsFiberArrayClass = PfsObject
    NotesClass = PfsObjectNotesTable
