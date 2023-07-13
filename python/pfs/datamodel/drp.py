import numpy as np

from .notes import makeNotesClass
from .pfsSimpleSpectrum import PfsSimpleSpectrum
from .pfsFiberArray import PfsFiberArray
from .pfsFiberArraySet import PfsFiberArraySet
from .pfsTable import PfsTable, Column
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
]


class PfsArmNotes(PfsTable):
    """Notes about spectral reduction for PfsArm"""

    fitsExtName = "NOTES"
    schema = [
        Column("blackSpotId", np.int32, "ID of black spot", -1),
        Column("blackSpotDistance", np.float32, "Distance to nearest black spot (mm)", np.nan),
        Column("blackSpotCorrection", np.float32, "Black spot flux correction", np.nan),
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
        Column("blackSpotDistance", np.float32, "Distance to nearest black spot (mm)", np.nan),
        Column("blackSpotCorrection", np.float32, "Black spot flux correction", np.nan),
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
class LineMeasurements(PfsTable):
    damdVer = 2
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
        Column("flux", np.float32, "Relative flux", np.nan),
        Column("fluxErr", np.float32, "Error in relative flux", np.nan),
        Column("flag", bool, "Measurement flag (true means bad)", True),
        Column("status", np.int32, "Reference line status bitmask", -1),
        Column("description", str, "Reference line description", "UNNOWN"),
        Column("transition", str, "Reference line transition", "UNNOWN"),
        Column("source", np.int32, "Reference line source code", -1),
    ]
    fitsExtName = "ARCLINES"
    aliases = dict(flux=("intensity",), fluxErr=("intensityErr"),)
