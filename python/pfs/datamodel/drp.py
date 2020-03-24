from .pfsSimpleSpectrum import PfsSimpleSpectrum
from .pfsFiberArray import PfsFiberArray
from .pfsFiberArraySet import PfsFiberArraySet
from .utils import inheritDocstrings

__all__ = ["PfsArm", "PfsMerged", "PfsReference", "PfsSingle", "PfsObject"]


@inheritDocstrings
class PfsArm(PfsFiberArraySet):
    """Spectra from reducing a single arm

    Produced by ``reduceExposure``.
    """
    filenameFormat = "pfsArm-%(visit)06d-%(arm)1s%(spectrograph)1d.fits"
    filenameRegex = r"^pfsArm-(\d{6})-([brnm])(\d)\.fits.*$"
    filenameKeys = [("visit", int), ("arm", str), ("spectrograph", int)]


@inheritDocstrings
class PfsMerged(PfsFiberArraySet):
    """Spectra from merging all arms within an exposure

    Produced by ``mergeArms``.
    """
    filenameFormat = "pfsMerged-%(visit)06d.fits"
    filenameRegex = r"^pfsMerged-(\d{6})\.fits.*$"
    filenameKeys = [("visit", int)]


@inheritDocstrings
class PfsReference(PfsSimpleSpectrum):
    """Reference spectrum for flux calibration

    Produced by ``calculateReferenceFlux``.
    """
    filenameFormat = "pfsReference-%(catId)03d-%(tract)05d-%(patch)s-%(objId)016x.fits"
    filenameRegex = r"^pfsReference-(\d{3})-(\d{5})-(.*)-([0-9a-f]{16})\.fits.*$"
    filenameKeys = [("catId", int), ("tract", int), ("patch", str), ("objId", int)]


@inheritDocstrings
class PfsSingle(PfsFiberArray):
    """Flux-calibrated, single epoch spectrum

    Produced by ``fluxCalibrate``.
    """
    filenameFormat = "pfsSingle-%(catId)03d-%(tract)05d-%(patch)s-%(objId)016x-%(visit)06d.fits"
    filenameRegex = r"^pfsSingle-(\d{3})-(\d{5})-(.*)-([0-9a-f]{16})-(\d{6})\.fits.*$"
    filenameKeys = [("catId", int), ("tract", int), ("patch", str), ("objId", int), ("visit", int)]


@inheritDocstrings
class PfsObject(PfsFiberArray):
    """Coadded spectrum

    Produced by ``coaddSpectra``.
    """
    filenameFormat = ("pfsObject-%(catId)03d-%(tract)05d-%(patch)s-%(objId)016x"
                      "-%(nVisit)03d-0x%(pfsVisitHash)016x.fits")
    filenameRegex = r"^pfsObject-(\d{3})-(\d{5})-(.*)-([0-9a-f]{16})-(\d{3})-0x([0-9a-f]{16})\.fits.*$"
    filenameKeys = [("catId", int), ("tract", int), ("patch", str), ("objId", int),
                    ("nVisit", int), ("pfsVisitHash", int)]
