from .pfsSpectra import PfsSpectra
from .pfsSpectrum import PfsSimpleSpectrum, PfsSpectrum

__all__ = ["PfsArm", "PfsMerged", "PfsReference", "PfsSingle", "PfsObject"]


class PfsArm(PfsSpectra):
    """Spectra from reducing a single arm

    Produced by ``reduceExposure``.
    """
    filenameFormat = "pfsArm-%(visit)06d-%(arm)1s%(spectrograph)1d.fits"
    filenameRegex = r"^pfsArm-(\d{6})-([brnm])(\d)\.fits.*$"
    filenameKeys = [("visit", int), ("arm", str), ("spectrograph", int)]


class PfsMerged(PfsSpectra):
    """Spectra from merging all arms within an exposure

    Produced by ``mergeArms``.
    """
    filenameFormat = "pfsMerged-%(visit)06d.fits"
    filenameRegex = r"^pfsMerged-(\d{6})\.fits.*$"
    filenameKeys = [("visit", int)]


class PfsReference(PfsSimpleSpectrum):
    """Reference spectrum for flux calibration

    Produced by ``calculateReferenceFlux``.
    """
    filenameFormat = "pfsReference-%(catId)03d-%(tract)05d-%(patch)s-%(objId)016x.fits"
    filenameRegex = r"^pfsReference-(\d{3})-(\d{5})-(.*)-(0x.{8})\.fits.*$"
    filenameKeys = [("catId", int), ("tract", int), ("patch", str), ("objId", int)]


class PfsSingle(PfsSpectrum):
    """Flux-calibrated, single epoch spectrum

    Produced by ``fluxCalibrate``.
    """
    filenameFormat = "pfsSingle-%(catId)03d-%(tract)05d-%(patch)s-%(objId)016x-%(visit)06d.fits"
    filenameRegex = r"^pfsSingle-(\d{3})-(\d{5})-(.*)-(0x.{16})-(\d{6})\.fits.*$"
    filenameKeys = [("catId", int), ("tract", int), ("patch", str), ("objId", int), ("visit", int)]


class PfsObject(PfsSpectrum):
    """Coadded spectrum

    Produced by ``coaddSpectra``.
    """
    filenameFormat = "pfsObject-%(catId)03d-%(tract)05d-%(patch)s-%(objId)016x-%(numExp)03d-%(pfsVisitHash)016x.fits"
    filenameRegex = r"^pfsObject-(\d{3})-(\d{5})-(.*)-(0x.{16})-(\d{3})-(0x.{16})\.fits.*$"
    filenameKeys = [("catId", int), ("tract", int), ("patch", str), ("objId", int),
                    ("numExp", int), ("pfsVisitHash", int)]
