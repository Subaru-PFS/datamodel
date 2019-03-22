from .pfsSpectra import PfsSpectra
from .pfsSpectrum import PfsSimpleSpectrum, PfsSpectrum

__all__ = ["PfsArm", "PfsMerged", "PfsReference", "PfsObject", "PfsCoadd"]


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
    filenameFormat = "pfsReference-%(catId)03d-%(tract)05d-%(patch)s-%(objId)08x.fits"
    filenameRegex = r"^pfsReference-(\d{3})-(\d{5})-(.*)-(0x.{8})\.fits.*$"
    filenameKeys = [("catId", int), ("tract", int), ("patch", str), ("objId", int)]


class PfsObject(PfsSpectrum):
    """Flux-calibrated, single epoch spectrum

    Produced by ``fluxCalibrate``.
    """
    filenameFormat = "pfsObject-%(catId)03d-%(tract)05d-%(patch)s-%(objId)08x-%(visit)06d.fits"
    filenameRegex = r"^pfsObject-(\d{3})-(\d{5})-(.*)-(0x.{8})-(\d{6})\.fits.*$"
    filenameKeys = [("catId", int), ("tract", int), ("patch", str), ("objId", int), ("visit", int)]


class PfsCoadd(PfsSpectrum):
    """Coadded spectrum

    Produced by ``coaddSpectra``.
    """
    filenameFormat = "pfsCoadd-%(catId)03d-%(tract)05d-%(patch)s-%(objId)08x-%(numExp)03d-%(expHash)08x.fits"
    filenameRegex = r"^pfsCoadd-(\d{3})-(\d{5})-(.*)-(0x.{8})-(\d{3})-(0x.{8})\.fits.*$"
    filenameKeys = [("catId", int), ("tract", int), ("patch", str), ("objId", int),
                    ("numExp", int), ("expHash", int)]
