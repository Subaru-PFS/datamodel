from types import SimpleNamespace
import warnings
import numpy as np
import os
import re
import enum
from collections import Counter
from logging import Logger
from typing import Optional

try:
    import astropy.io.fits as pyfits
except ImportError:
    pyfits = None

from .guideStars import GuideStars
from .utils import checkHeaderKeyword


__all__ = (
    "DocEnum",
    "TargetType",
    "FiberStatus",
    "PfsDesign",
    "PfsConfig",
    "PFSCONFIG_FILENAME_REGEX",
    "parsePfsConfigFilename",
    "checkPfsConfigHeader",
)


class DocEnum(enum.IntEnum):
    """An integer enumerated type with documented members

    From https://stackoverflow.com/a/50473952/834250
    """
    def __new__(cls, value, doc):
        self = int.__new__(cls, value)
        self._value_ = value
        self.__doc__ = doc
        return self

    @classmethod
    def getFitsHeaders(cls):
        """Return FITS headers documenting the options

        Returns
        -------
        header : `dict` (`str`: `str`)
            Keyword-value pairs to include in a FITS header.
        """
        keyBase = "HIERARCH " + cls.__name__ + "."
        return {keyBase + member.name: (member.value, member.__doc__) for member in cls}

    def __str__(self):
        """Return the enum's name"""
        return self.name

    def __invert__(self):
        """Return all the elements of the enum except self

        Returns
        -------
        elements : list of all members of enumeration except self
        """
        return [member for member in type(self) if member != self]

    @classmethod
    def fromString(cls, name):
        """Construct from the string name

        Parameters
        ----------
        name : `str`
            Name of the enum.

        Returns
        -------
        self : cls
            Enum with the supplied name.
        """
        return getattr(cls, name)


class TargetType(DocEnum):
    """Enumerated options for what a fiber is targeting"""
    SCIENCE = 1, "science target"
    SKY = 2, "blank sky; used for sky subtraction"
    FLUXSTD = 3, "flux standard; used for fluxcal"
    UNASSIGNED = 4, "no particular target"
    ENGINEERING = 5, "engineering fiber"
    SUNSS_IMAGING = 6, "fiber goes to the SuNSS imaging leg"
    SUNSS_DIFFUSE = 7, "fiber goes to the SuNSS diffuse leg"
    DCB = 8, "fiber goes to DCB/DCB2"
    HOME = 9, "cobra is going to home position"
    BLACKSPOT = 10, "cobra is going to black spot position"
    AFL = 11, "fiber goes to all fiber lamp"


class FiberStatus(DocEnum):
    """Enumerated options for the status of a fiber"""
    GOOD = 1, "working normally"
    BROKENFIBER = 2, "broken; ignore any flux"
    BLOCKED = 3, "temporarily blocked; ignore any flux"
    BLACKSPOT = 4, "hidden behind spot; ignore any flux"
    UNILLUMINATED = 5, "not illuminated; ignore any flux"
    BROKENCOBRA = 6, "Cobra does not move, but the fiber still carries flux."


class PfsDesign:
    """The design of the PFS top-end configuration for one or more observations

    Parameters
    ----------
    pfsDesignId : `int`
        PFI design identifier, specifies the intended top-end configuration.
    raBoresight : `float`, degrees
        Right Ascension of telescope boresight.
    decBoresight : `float`, degrees
        Declination of telescope boresight.
    posAng : `float`, degrees
        The position angle from the
        North Celestial Pole to the PFI_Y axis,
        measured clockwise with respect to the
        positive PFI_Z axis
    arms : `str`
        arms to expose. Eg 'brn', 'bmn'.
    fiberId : `numpy.ndarary` of `int32`
        Fiber identifier for each fiber.
    tract : `numpy.ndarray` of `int32`
        Tract index for each fiber.
    patch : `numpy.ndarray` of `str`
        Patch indices for each fiber, typically two integers separated by a
        comma, e.g,. "5,6".
    ra : `numpy.ndarray` of `float64`
        Right Ascension for each fiber, degrees.
    dec : `numpy.ndarray` of `float64`
        Declination for each fiber, degrees.
    catId : `numpy.ndarray` of `int32`
        Catalog identifier for each fiber.
    objId : `numpy.ndarray` of `int64`
        Object identifier for each fiber. Specifies the object within the
        catalog.
    targetType : `numpy.ndarray` of `int`
        Type of target for each fiber. Values must be convertible to
        `TargetType` (which limits the range of values).
    fiberStatus : `numpy.ndarray` of `int`
        Status of each fiber. Values must be convertible to `FiberStatus`
        (which limits the range of values).
    epoch : `numpy.chararray`
        reference epoch for each fiber.
    pmRa : `numpy.ndarray` of `float32`
        Proper motion in direction of Right Ascension
        for each fiber, mas/year.
    pmDec : `numpy.ndarray` of `float32`
        Proper motion in direction of Declination
        for each fiber, mas/year.
    parallax : `numpy.ndarray` of `float32`
        parallax for each fiber, mas.
    proposalId : `numpy.chararray`
        Proposal ID of each fiber (e.g, S23A-001QN).
    obCode : `numpy.chararray`
        Code for an Observing Block (OB) of each fiber.
    fiberFlux : `list` of `numpy.ndarray` of `float`
        Array of fiber fluxes for each fiber, in [nJy].
    psfFlux : `list` of `numpy.ndarray` of `float`
        Array of PSF fluxes for each target/fiber in [nJy]
    totalFlux : `list` of `numpy.ndarray` of `float`
        Array of total fluxes for each target/fiber in [nJy].
    fiberFluxErr : `list` of `numpy.ndarray` of `float`
        Array of fiber flux errors for each fiber in [nJy].
    psfFluxErr : `list` of `numpy.ndarray` of `float`
        Array of PSF flux errors for each target/fiber in [nJy].
    totalFluxErr : `list` of `numpy.ndarray` of `float`
        Array of total flux errors for each target/fiber in [nJy].
    filterNames : `list` of `list` of `str`
        List of filters used to measure the fiber fluxes for each filter.
    pfiNominal : `numpy.ndarray` of `float`
        Intended target position (2-vector) of each fiber on the PFI, millimeters.
    guideStars : `GuideStars`
        Guide star data. If `None`, an empty GuideStars instance will be created.
    designName : `str`, optional
        Human-readable name for the design.
    variant : `int`, optional
        Counter of which variant of `designId0` we are. Requires `designId0`.
    designId0 : `int`, optional
        pfsDesignId of the pfsDesign we are a variant of. Requires `variant`.
    """
    # Scalar values
    _scalars = ["pfsDesignId", "designName",
                "raBoresight", "decBoresight", "posAng", "arms", "guideStars",
                "variant", "designId0"]
    # List of fields required, and their FITS type
    # Some elements of the code expect the following to be present:
    #     fiberId, targetType
    # fiberStatus is handled separately, for backwards-compatibility
    _fields = {"fiberId": "J",
               "tract": "J",
               "patch": "A",
               "ra": "D",
               "dec": "D",
               "catId": "J",
               "objId": "K",
               "targetType": "J",
               "epoch": "A",
               "pmRa": "E",
               "pmDec": "E",
               "parallax": "E",
               "proposalId": "A",
               "obCode": "A",
               "pfiNominal": "2E",
               }
    # astrometry keywords; should be in _fields too
    _astrometry = ["epoch",
                   "pmRa",
                   "pmDec",
                   "parallax"]
    # keys for operation; should be present in _fields too
    _operation = ["proposalId",
                  "obCode"]
    _pointFields = ["pfiNominal"]  # List of point fields; should be in _fields too
    _photometry = ["fiberFlux",
                   "psfFlux",
                   "totalFlux",
                   "fiberFluxErr",
                   "psfFluxErr",
                   "totalFluxErr",
                   "filterNames"]  # List of photometry fields
    _keywords = list(_fields) + _photometry
    _hduName = "DESIGN"
    _POSANG_DEFAULT = 0.0

    fileNameFormat = "pfsDesign-0x%016x.fits"

    def validate(self):
        """Validate contents

        Ensures the lengths are what is expected.

        Raises
        ------
        RuntimeError
            If there are inconsistent lengths or the GuideStars instance passed is None.
        ValueError:
            If the ``targetType`` or ``fiberStatus`` is not recognised, or
            if the combination of ``catId`` and ``objId`` are not unique.
        """
        if self.guideStars is None:
            raise RuntimeError('The GuideStars instance cannot be None. '
                               'If no valid instance can be provided, '
                               'pass and empty instance, eg., GuideStars.empty().')

        if (self.variant != 0) ^ (self.designId0 != 0):
            raise RuntimeError("if either variant or designId0 is set both must be set.")

        if len(set([len(getattr(self, nn)) for nn in self._keywords])) != 1:
            raise RuntimeError("Inconsistent lengths: %s" % ({nn: len(getattr(self, nn)) for
                                                              nn in self._keywords}))
        for ii, tt in enumerate(self.targetType):
            try:
                TargetType(tt)
            except ValueError as exc:
                raise ValueError("targetType[%d] = %d is not a recognized TargetType" % (ii, tt)) from exc
        for ii, tt in enumerate(self.fiberStatus):
            try:
                FiberStatus(tt)
            except ValueError as exc:
                raise ValueError("fiberStatus[%d] = %d is not a recognised FiberStatus" % (ii, tt)) from exc

        for ii, (mag, names) in enumerate(zip(self.fiberFlux, self.filterNames)):
            if len(mag) != len(names):
                raise RuntimeError("Inconsistent lengths between fiberFlux (%d) and filterNames (%d) "
                                   "for fiberId=%d" % (len(mag), len(names), self.fiberId[ii]))
        for ii, (pFlux, names) in enumerate(zip(self.psfFlux, self.filterNames)):
            if len(pFlux) != len(names):
                raise RuntimeError("Inconsistent lengths between psfFlux (%d) and filterNames (%d) "
                                   "for fiberId=%d" % (len(pFlux), len(names), self.fiberId[ii]))
        for ii, (tFlux, names) in enumerate(zip(self.totalFlux, self.filterNames)):
            if len(tFlux) != len(names):
                raise RuntimeError("Inconsistent lengths between totalFlux (%d) and filterNames (%d) "
                                   "for fiberId=%d" % (len(tFlux), len(names), self.fiberId[ii]))
        for ii, (ffErr, names) in enumerate(zip(self.fiberFluxErr, self.filterNames)):
            if len(ffErr) != len(names):
                raise RuntimeError("Inconsistent lengths between fiberFluxErr (%d) and filterNames (%d) "
                                   "for fiberId=%d" % (len(ffErr), len(names), self.fiberId[ii]))
        for ii, (pfErr, names) in enumerate(zip(self.psfFluxErr, self.filterNames)):
            if len(pfErr) != len(names):
                raise RuntimeError("Inconsistent lengths between psfFluxErr (%d) and filterNames (%d) "
                                   "for fiberId=%d" % (len(pfErr), len(names), self.fiberId[ii]))
        for ii, (tfErr, names) in enumerate(zip(self.totalFluxErr, self.filterNames)):
            if len(tfErr) != len(names):
                raise RuntimeError("Inconsistent lengths between totalFluxErr (%d) and filterNames (%d) "
                                   "for fiberId=%d" % (len(tfErr), len(names), self.fiberId[ii]))
        for nn in self._pointFields:
            matrix = getattr(self, nn)
            if matrix.shape != (len(self.fiberId), 2):
                raise RuntimeError("Wrong shape for %s: %s vs (%d,2)" % (nn, matrix.shape, len(self.fiberId)))

        # Check for duplicates of catId, objId combinations.
        counts = Counter(zip(self.catId, self.objId))
        counts.pop((-1, -1), None)  # ignore untargetted fibers
        if counts and counts.most_common(1)[0][1] > 1:
            duplicates = {tup: count for tup, count in counts.items() if count > 1}
            raise ValueError(f'design {self.pfsDesignId:#016x} contains duplicate occurrences of'
                             ' the same (catId, objId) combination. Details below:\n'
                             f'{{(catId, objId): number of occurrences}}:\n\t {duplicates}')

    def __init__(self, pfsDesignId, raBoresight, decBoresight,
                 posAng,
                 arms,
                 fiberId, tract, patch, ra, dec, catId, objId,
                 targetType, fiberStatus,
                 epoch, pmRa, pmDec, parallax,
                 proposalId, obCode,
                 fiberFlux,
                 psfFlux,
                 totalFlux,
                 fiberFluxErr,
                 psfFluxErr,
                 totalFluxErr,
                 filterNames, pfiNominal,
                 guideStars,
                 designName="",
                 variant=0,
                 designId0=0):
        self.pfsDesignId = pfsDesignId
        self.raBoresight = raBoresight
        self.decBoresight = decBoresight
        self.posAng = posAng
        self.arms = arms
        self.fiberId = np.array(fiberId).astype(np.int32)
        self.tract = np.array(tract)
        self.patch = np.array(patch)
        self.ra = np.array(ra).astype(float)
        self.dec = np.array(dec).astype(float)
        self.catId = np.array(catId)
        self.objId = np.array(objId)
        self.targetType = np.array(targetType)
        self.fiberStatus = np.array(fiberStatus)
        self.epoch = np.array(epoch, dtype=str)
        self.pmRa = np.array(pmRa).astype(np.float32)
        self.pmDec = np.array(pmDec).astype(np.float32)
        self.parallax = np.array(parallax).astype(np.float32)
        self.proposalId = np.array(proposalId, dtype=str)
        self.obCode = np.array(obCode, dtype=str)
        self.fiberFlux = [np.array(flux).astype(float) for flux in fiberFlux]
        self.psfFlux = [np.array(pflux).astype(float) for pflux in psfFlux]
        self.totalFlux = [np.array(tflux).astype(float) for tflux in totalFlux]
        self.fiberFluxErr = [np.array(ffErr).astype(float) for ffErr in fiberFluxErr]
        self.psfFluxErr = [np.array(pfErr).astype(float) for pfErr in psfFluxErr]
        self.totalFluxErr = [np.array(tfErr).astype(float) for tfErr in totalFluxErr]
        self.filterNames = filterNames
        self.pfiNominal = np.array(pfiNominal)
        self.guideStars = guideStars if guideStars is not None else GuideStars.empty()
        self.designName = designName
        self.variant = variant
        self.designId0 = designId0
        self.isSubset = False
        self.validate()

    def __len__(self):
        """Number of fibers"""
        return len(self.fiberId)

    def __str__(self):
        """String representation"""
        return "PfsDesign(%d, ...)" % (self.pfsDesignId)

    def __iter__(self):
        """Iteration returns the target for each fiber"""
        for ii in range(len(self)):
            yield self.getTarget(ii)

    def __getitem__(self, logical):
        """Sub-selection

        Parameters
        ----------
        logical : `numpy.ndarray` of `bool`
            Boolean array (of same length as ``self``) indicating which fibers
            to select.

        Returns
        -------
        new : ``type(self)``
            A new ``PfsDesign`` or ``PfsConfig`` containing only the selected
            fibers.
        """
        numOriginal = len(self)
        kwargs = {name: getattr(self, name) for name in self._scalars}
        for name in self._keywords + ["fiberStatus"]:
            array = getattr(self, name)
            if isinstance(array, np.ndarray):
                subArray = array[logical]
            else:
                subArray = [array[ii] for ii in range(numOriginal) if logical[ii]]
            kwargs[name] = subArray
        new = type(self)(**kwargs)
        new.isSubset = True
        return new

    @property
    def fiberHole(self):
        """Fiber hole number, based on the fiberId"""
        from pfs.utils.fibers import fiberHoleFromFiberId
        return fiberHoleFromFiberId(self.fiberId)

    @property
    def spectrograph(self):
        """Spectrograph number, based on the fiberId"""
        from pfs.utils.fibers import spectrographFromFiberId
        return spectrographFromFiberId(self.fiberId)

    def getTarget(self, index):
        """Return target by index

        Parameters
        ----------
        index : `int`
            Index of fiber of interest.

        Returns
        -------
        target : `pfs.datamodel.Target`
            Target for fiber.
        """
        from pfs.datamodel.target import Target  # noqa: prevent circular import dependency
        return Target.fromPfsConfig(self, index)

    @property
    def filename(self):
        """Usual filename"""
        return self.fileNameFormat % (self.pfsDesignId)

    def getVariant(self):
        """Return the (variantNum, basePfsDesign) pair, or (0, 0) if we are not a variant """
        return self.variant, self.designId0

    def getPhotometry(self, filterName, psfFlux=False, fiberFlux=False, totalFlux=False, getError=False,
                      asAB=False):
        """Return the flux, and optionally errors, for a requested filter.

        If the filtername is invalid, the valid names printer and an exception raised

        Parameters
        ----------
            filterName : `str`
                Name of desired filter (e.g. g_hsc)
            psfFlux: `bool`
                Return the PSF flux (default)
            fiberFlux: `bool`
                Return the fiber flux
            totalFlux: `bool`
                Return the totalflux
            getError: `bool`
                Return the flux error in addition to the flux
            asAB: `bool`
                Return the flux/fluxError as AB magnitudes

        Returns
        -------
        flux: `np.ndarray`  if getError is False
        flux, fluxErr: (`np.ndarray`, `np.ndarray`)   if getError is False
        """

        if not (psfFlux or fiberFlux or totalFlux):
            psfFlux = True

        fluxRequested = []
        if psfFlux:
            fluxRequested.append("psf")
            flux, fluxErr = self.psfFlux, self.psfFluxErr
        elif fiberFlux:
            fluxRequested.append("fiber")
            flux, fluxErr = self.fiberFlux, self.fiberFluxErr
        else:
            fluxRequested.append("total")
            flux, fluxErr = self.totalFlux, self.totalFluxErr

        if len(fluxRequested) > 1:
            raise RuntimeError(f"Please only specify one type of flux at a time: saw "
                               f"{', '.join(fluxRequested)}")

        myFilterData = np.array(self.filterNames) == filterName
        if np.sum(myFilterData) == 0:
            filterNames = sorted(set([fn for fn in sum(self.filterNames, []) if fn != "none"]))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)  # All-NaN slice encountered

                goodFilterNames = []
                for possibleFilterName in filterNames:
                    myFilterData = np.array(self.filterNames) == possibleFilterName

                    if np.isfinite(np.nanmean(np.where(myFilterData, flux, np.NaN), axis=1)):
                        goodFilterNames.append(possibleFilterName)

            raise RuntimeError(f"No flux data for filter \"{filterName}\" are available. " +
                               (f"Options are: {', '.join(goodFilterNames)}" if len(goodFilterNames) > 0
                                else ""))

        def nJyToAB(nJy):
            return 8.9 - 2.5*np.log10(1e-9*nJy)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # All-NaN slice encountered

            flux = np.nanmean(np.where(myFilterData, flux, np.NaN), axis=1)

            if asAB:
                AB = nJyToAB(flux)

            if getError:
                fluxErr = np.nanmean(np.where(myFilterData, fluxErr, np.NaN), axis=1)

                if asAB:
                    if fluxErr < flux:
                        ABErr = np.mean([nJyToAB(flux + fluxErr) - AB, AB - nJyToAB(flux - fluxErr)])
                    else:
                        ABErr = nJyToAB(flux + fluxErr) - AB

                    return AB, ABErr
                else:
                    return flux, fluxErr
            else:
                return AB if asAB else flux

    @classmethod
    def _readHeader(cls, header, kwargs=None):
        """Read construction variables from the header.

        Parameters
        ----------
        header : `astropy.io.fits.Header`
            FITS header.
        kwargs : `dict`, optional
            Keyword arguments to be passed to the constructor; modified.

        Returns
        -------
        kwargs : `dict`
            Keyword arguments to be passed to the constructor.
        """
        if kwargs is None:
            kwargs = {}

        # If POSANG does not exist, use default.
        # This default should be removed when the
        # relevant test datasets have this keyword
        # populated.
        kwargs["posAng"] = header.get("POSANG", cls._POSANG_DEFAULT)

        # If ARM does not exist, use default.
        # This action should be removed once the
        # relevant test datasets have this keyword
        # populated.
        kwargs["arms"] = header.get("ARMS", "brn")

        # If DSGN_NAM does not exist, use default.
        # This action should be removed once the relevant test datasets have this keyword populated.
        kwargs["designName"] = header.get("DSGN_NAM", "")

        # We formerly got the pfsDesignId from the filename, which is fragile.
        # Now we look for it in the W_PFDSGN header, but need to allow for the possibility that it's
        # not present.
        pfsDesignId = header.get("W_PFDSGN", None)
        if pfsDesignId is not None:
            if "pfsDesignId" in kwargs and kwargs["pfsDesignId"] != pfsDesignId:
                raise RuntimeError(f"pfsDesignId mismatch: {kwargs['pfsDesignId']} vs pfsDesignId")
            kwargs["pfsDesignId"] = pfsDesignId
        elif "pfsDesignId" not in kwargs:
            raise RuntimeError("Unable to determine pfsDesignId")

        # Load design variant cards, added for DAMD-140. If none are set, declare the design
        # as having none.
        kwargs["variant"] = header.get("VARIANT", 0)
        kwargs["designId0"] = header.get("PFDSGN0", 0)

        return kwargs

    def _writeHeader(self, header):
        """Write to the header.

        Parameters
        ----------
        header : `astropy.io.fits.Header`
            FITS header; modified.
        """
        header["RA"] = (self.raBoresight, "[degree] pfsDesign field center RA")
        header["DEC"] = (self.decBoresight, "[degree] pfsDesign field center DEC")
        header["POSANG"] = (self.posAng, "[degree] PFI position angle")
        header["ARMS"] = (self.arms, "Exposed arms")
        header["DSGN_NAM"] = (self.designName, "Name of design")
        header["DAMD_VER"] = (3, "PfsDesign/PfsConfig datamodel version")
        header["W_PFDSGN"] = (self.pfsDesignId, "Identifier for fiber configuration")
        header["VARIANT"] = (self.variant, "Which variant of PFDSGN0 we are.")
        header["PFDSGN0"] = (self.designId0, "The base design of which we are a variant")

    @classmethod
    def _readImpl(cls, filename, **kwargs):
        """Implementation for reading from file

        Parameters
        ----------
        filename : `str`
            Full path for file to read.
        **kwargs : `dict`
            Additional arguments for Ctor (not read from FITS).

        Returns
        -------
        self : cls
            Constructed instance.
        """
        if not pyfits:
            raise RuntimeError("I failed to import astropy.io.fits, so cannot read from disk")

        with pyfits.open(filename) as fd:
            phu = fd[0].header
            raBoresight = phu['RA']
            decBoresight = phu['DEC']

            # If DAMD_VER does not exist, set to None.
            # This default should be removed when the
            # relevant test datasets have this keyword
            # populated.
            damdVer = phu.get('DAMD_VER', None)

            headerKwargs = cls._readHeader(phu, kwargs)
            for name in headerKwargs:
                if name in kwargs and headerKwargs[name] != kwargs[name]:
                    raise RuntimeError(f"{name} mismatch: {kwargs[name]} vs {headerKwargs[name]}")
            kwargs.update(headerKwargs)
            if "pfsDesignId" not in kwargs:
                raise RuntimeError("Unable to determine pfsDesignId")

            data = fd[cls._hduName].data

            # fill astrometry columns if not exist, for backwards compatibility
            kwargs["epoch"] = data["epoch"] if "epoch" in data.columns.names else np.full(
                len(data), "J2000.0")
            kwargs["pmRa"] = data["pmRa"] if "pmRa" in data.columns.names else np.full(
                len(data), 0.0, dtype=np.float32)
            kwargs["pmDec"] = data["pmDec"] if "pmDec" in data.columns.names else np.full(
                len(data), 0.0, dtype=np.float32)
            kwargs["parallax"] = data["parallax"] if "parallax" in data.columns.names else np.full(
                len(data), 1.0e-8, dtype=np.float32)

            # fill operation-related columns if not exist, for backwards compatibility
            kwargs["proposalId"] = (data["proposalId"]
                                    if "proposalId" in data.columns.names else np.full(len(data), "N/A"))
            kwargs["obCode"] = data["obCode"] if "obCode" in data.columns.names else np.full(len(data), "N/A")

            for nn in cls._fields:
                # skip astrometry and operation keywords as they have already been set
                if (nn not in cls._astrometry) and (nn not in cls._operation):
                    assert nn not in kwargs
                    kwargs[nn] = data[nn]

            # Handle fiberStatus explicitly, for backwards compatibility
            kwargs["fiberStatus"] = (data["fiberStatus"] if "fiberStatus" in (col.name for col in
                                                                              data.columns) else
                                     np.full(len(data), FiberStatus.GOOD))

            photometry = fd["PHOTOMETRY"].data

            fiberId = kwargs["fiberId"]
            fiberFlux = {ii: [] for ii in fiberId}
            psfFlux = {ii: [] for ii in fiberId}
            totalFlux = {ii: [] for ii in fiberId}
            fiberFluxErr = {ii: [] for ii in fiberId}
            psfFluxErr = {ii: [] for ii in fiberId}
            totalFluxErr = {ii: [] for ii in fiberId}
            filterNames = {ii: [] for ii in fiberId}
            for row in photometry:
                fiberFlux[row['fiberId']].append(row['fiberFlux'])
                psfFlux[row['fiberId']].append(row['psfFlux'])
                totalFlux[row['fiberId']].append(row['totalFlux'])
                fiberFluxErr[row['fiberId']].append(row['fiberFluxErr'])
                psfFluxErr[row['fiberId']].append(row['psfFluxErr'])
                totalFluxErr[row['fiberId']].append(row['totalFluxErr'])
                filterNames[row['fiberId']].append(row['filterName'])

            if damdVer is not None and damdVer >= 2:
                guideStars = GuideStars.fromFits(fd)
            else:
                # Create an empty HDU.
                # This action should be removed once the
                # relevant test datasets have this keyword
                # populated.
                guideStars = GuideStars.empty()

        return cls(**kwargs, raBoresight=raBoresight, decBoresight=decBoresight,
                   fiberFlux=[np.array(fiberFlux[ii]) for ii in fiberId],
                   psfFlux=[np.array(psfFlux[ii]) for ii in fiberId],
                   totalFlux=[np.array(totalFlux[ii]) for ii in fiberId],
                   fiberFluxErr=[np.array(fiberFluxErr[ii]) for ii in fiberId],
                   psfFluxErr=[np.array(psfFluxErr[ii]) for ii in fiberId],
                   totalFluxErr=[np.array(totalFluxErr[ii]) for ii in fiberId],
                   filterNames=[filterNames[ii] for ii in fiberId],
                   guideStars=guideStars)

    @classmethod
    def read(cls, pfsDesignId, dirName="."):
        """Construct from file

        Requires pyfits.

        Parameters
        ----------
        pfsDesignId : `int`
            PFI design identifier, specifies the intended top-end configuration.
        dirName : `str`, optional
            Directory from which to read the file. Defaults to the current
            directory.

        Returns
        -------
        self : `PfsDesign`
            Constructed `PfsDesign`.
        """
        filename = os.path.join(dirName, cls.fileNameFormat % (pfsDesignId))
        return cls._readImpl(filename, pfsDesignId=pfsDesignId)

    def _writeImpl(self, filename, *, allowSubset=False):
        """Implementation for writing to FITS file

        Parameters
        ----------
        filename : `str`
            Filename to which to write.
        allowSubset : `bool`, optional
            Allow writing a subset? Because of the danger in overwriting the
            original (both original and subset have the same pfsDesignId and
            therefore the same filename), this has to be specified explicitly.
        """
        # NOTE: When making any changes to this method that modify the output
        # format, increment the DAMD_VER header value and record the change in
        # the versions.txt file.
        if self.isSubset and not allowSubset:
            raise RuntimeError("Writing of subsets is disallowed in order to prevent clobbering the original;"
                               " use allowSubset=True if you really know what you're doing.")
        if not pyfits:
            raise RuntimeError("I failed to import astropy.io.fits, so cannot write to disk")

        fits = pyfits.HDUList()

        hdr = pyfits.Header()
        self._writeHeader(hdr)
        hdu = pyfits.PrimaryHDU(header=hdr)
        fits.append(hdu)

        # Add in the enumerations for the DESIGN fields
        hdr.update(TargetType.getFitsHeaders())
        hdr.update(FiberStatus.getFitsHeaders())

        lengths = [len(pp) for pp in self.patch]
        maxLength = 1 if len(lengths) == 0 else max(lengths)
        columns = []
        for name in self._fields:
            format = self._fields[name]
            if format == "A":
                lengths = [len(ss) for ss in getattr(self, name)]
                maxLength = 1 if len(lengths) == 0 else max(lengths)
                format = "A%d" % maxLength
            columns.append(pyfits.Column(name=name, format=format, array=getattr(self, name)))
        columns.append(pyfits.Column(name="fiberStatus", format="J", array=self.fiberStatus))
        fits.append(pyfits.BinTableHDU.from_columns(columns, hdr, name=self._hduName))

        numRows = sum(len(fFlux) for fFlux in self.fiberFlux)
        fiberId = np.array(sum(([ii]*len(mags) for ii, mags in zip(self.fiberId, self.fiberFlux)), []))
        fiberFlux = np.array(sum((fFlux.tolist() for fFlux in self.fiberFlux), []))
        psfFlux = np.array(sum((pflux.tolist() for pflux in self.psfFlux), []))
        totalFlux = np.array(sum((tflux.tolist() for tflux in self.totalFlux), []))
        fiberFluxErr = np.array(sum((ffErr.tolist() for ffErr in self.fiberFluxErr), []))
        psfFluxErr = np.array(sum((pfErr.tolist() for pfErr in self.psfFluxErr), []))
        totalFluxErr = np.array(sum((tfErr.tolist() for tfErr in self.totalFluxErr), []))
        filterNames = sum(self.filterNames, [])
        assert(len(fiberId) == numRows)
        assert(len(fiberFlux) == numRows)
        assert(len(psfFlux) == numRows)
        assert(len(totalFlux) == numRows)
        assert(len(fiberFluxErr) == numRows)
        assert(len(psfFluxErr) == numRows)
        assert(len(totalFluxErr) == numRows)
        assert(len(filterNames) == numRows)
        maxLength = max(len(ff) for ff in filterNames) if filterNames else 1

        fits.append(pyfits.BinTableHDU.from_columns([
            pyfits.Column(name='fiberId', format='J', array=fiberId),
            pyfits.Column(name='fiberFlux', format='E', array=fiberFlux, unit='nJy'),
            pyfits.Column(name='psfFlux', format='E', array=psfFlux, unit='nJy'),
            pyfits.Column(name='totalFlux', format='E', array=totalFlux, unit='nJy'),
            pyfits.Column(name='fiberFluxErr', format='E', array=fiberFluxErr, unit='nJy'),
            pyfits.Column(name='psfFluxErr', format='E', array=psfFluxErr, unit='nJy'),
            pyfits.Column(name='totalFluxErr', format='E', array=totalFluxErr, unit='nJy'),
            pyfits.Column(name='filterName', format='A%d' % maxLength, array=filterNames),
        ], hdr, name='PHOTOMETRY'))

        assert(self.guideStars is not None)
        self.guideStars.toFits(fits)

        # clobber=True in writeto prints a message, so use open instead
        with open(filename, "wb") as fd:
            fits.writeto(fd, checksum=True)

    def write(self, dirName=".", fileName=None, *, allowSubset=False):
        """Write to file

        Requires pyfits.

        Parameters
        ----------
        dirName : `str`, optional
            Directory to which to write the file. Defaults to the current
            directory.
        fileName : `str`, optional
            Filename to which to write. Defaults to using the filename template.
        allowSubset : `bool`, optional
            Allow writing a subset? Because of the danger in overwriting the
            original (both original and subset have the same pfsDesignId and
            therefore the same filename), this has to be specified explicitly.
        """
        if fileName is None:
            fileName = self.filename
        self._writeImpl(os.path.join(dirName, fileName), allowSubset=allowSubset)

    def getSelection(self, fiberId=None, targetType=None, fiberStatus=None,
                     catId=None, tract=None, patch=None, objId=None,
                     spectrograph=None):
        """Return a boolean array indicating which fibers are selected

        The values may be scalars or lists/tuples (sets are not supported)

        Multiple attributes will be combined with ``AND``, so
           select(fiberId=myFiberIds, fiberStatus=FiberStatus.GOOD, targetType=~TargetType.ENGINEERING)
        means
           fiberId is in myFiberIds and
           fiberStatus equals FiberStatus.GOOD and
           targetType is not in TargetType.ENGINEERING

        Parameters
        ----------
        fiberId : `int` (scalar or array_like), optional
            Fiber identifier to select.
        targetType : `TargetType` (scalar or array_like), optional
            Target type to select.
        fiberStatus : `FiberStatus` (scalar or array_like), optional
            Fiber status to select.
        catId : `int` (scalar or array_like), optional
            Catalog identifier to select.
        tract : `int` (scalar or array_like), optional
            Tract number to select.
        patch : `str` (scalar or array_like), optional
            Patch name to select.
        objId : `int` (scalar or array_like), optional
            Object identifier to select.
        spectrograph : `int` (scalar or array_like), optional
            Spectrograph to select.

        Returns
        -------
        logical : `numpy.ndarray` of `bool`
            A boolean array indicating which fibers are selected.
        """
        selection = np.ones(len(self), dtype=bool)
        if fiberId is not None:
            selection &= np.isin(self.fiberId, fiberId)
        if targetType is not None:
            selection &= np.isin(self.targetType, targetType)
        if fiberStatus is not None:
            selection &= np.isin(self.fiberStatus, fiberStatus)
        if catId is not None:
            selection &= np.isin(self.catId, catId)
        if tract is not None:
            selection &= np.isin(self.tract, tract)
        if patch is not None:
            selection &= np.isin(self.patch, patch)
        if objId is not None:
            selection &= np.isin(self.objId, objId)
        if spectrograph is not None:
            selection &= np.isin(self.spectrograph, spectrograph)
        return selection

    def select(self, **kwargs):
        """Return an instance containing only the selected attributes

        The values may be scalars or lists/tuples (sets are not supported)

        Multiple attributes will be combined with ``AND``, so
           select(fiberId=myFiberIds, fiberStatus=FiberStatus.GOOD, targetType=~TargetType.ENGINEERING)
        means
           fiberId is in myFiberIds and
           fiberStatus equals FiberStatus.GOOD and
           targetType is not in TargetType.ENGINEERING

        Parameters
        ----------
        fiberId : `int` (scalar or array_like), optional
            Fiber identifier to select.
        targetType : `TargetType` (scalar or array_like), optional
            Target type to select.
        fiberStatus : `FiberStatus` (scalar or array_like), optional
            Fiber status to select.
        catId : `int` (scalar or array_like), optional
            Catalog identifier to select.
        tract : `int` (scalar or array_like), optional
            Tract number to select.
        patch : `str` (scalar or array_like), optional
            Patch name to select.
        objId : `int` (scalar or array_like), optional
            Object identifier to select.
        spectrograph : `int` (scalar or array_like), optional
            Spectrograph to select.

        Returns
        -------
        selected : ``type(self)``
            An instance containing only the selected attributes.
        """
        selection = self.getSelection(**kwargs)
        return self[selection]

    def selectByTargetType(self, targetType, fiberId=None):
        """Select fibers by ``targetType``

        If a `fiberId` array is provided, returns indices for array;
        otherwise, returns indices for ``self``.

        Parameters
        ----------
        targetType : `TargetType`
            Target type to select.
        fiberId : `numpy.ndarray` of `int`, optional
            Array of fiber identifiers to select.

        Returns
        -------
        indices : `numpy.ndarray` of `int`
            Indices of selected elements.
        """
        targetType = int(targetType)
        select = self.targetType == targetType
        if fiberId is None:
            return np.nonzero(select)[0]
        return np.nonzero(np.isin(fiberId, self.fiberId[select]))[0]

    def selectByFiberStatus(self, fiberStatus, fiberId=None):
        """Select fibers by ``fiberStatus``

        If a `fiberId` array is provided, returns indices for array;
        otherwise, returns indices for ``self``.

        Parameters
        ----------
        fiberStatus : `FiberStatus`
            Fiber status to select.
        fiberId : `numpy.ndarray` of `int`, optional
            Array of fiber identifiers to select.

        Returns
        -------
        indices : `numpy.ndarray` of `int`
            Indices of selected elements.
        """
        fiberStatus = int(fiberStatus)
        select = self.fiberStatus == fiberStatus
        if fiberId is None:
            return np.nonzero(select)[0]
        return np.nonzero(np.isin(fiberId, self.fiberId[select]))[0]

    def selectTarget(self, catId, tract, patch, objId):
        """Select fiber by target

        Returns index for the fiber that matches the target identity.

        Parameters
        ----------
        catId : `int`
            Catalog identifier.
        tract : `int`
            Trace identifier.
        patch : `str`
            Patch name.
        objId : `int`
            Object identifier.

        Returns
        -------
        index : `int`
            Index of selected target.
        """
        index = np.argwhere((self.catId == catId) & (self.tract == tract) &
                            (self.patch == patch) & (self.objId == objId))
        if len(index) != 1:
            raise RuntimeError("Non-unique selection of target: %s" % (index,))
        return index[0][0]

    def selectFiber(self, fiberId):
        """Select fiber(s) by fiber identifier

        Returns the indices for the provided fiber identifiers.

        Note that the order will not be consistent.

        Parameters
        ----------
        fiberId : `int` or iterable of `int`
            Fiber identifiers to select.

        Returns
        -------
        index : array-like of `int`
            Indices for fibers.

        Raises
        ------
        RuntimeError
            If a scalar ``fiberId`` is requested but not present.
        """
        result = np.nonzero(np.isin(self.fiberId, fiberId))[0]
        if np.isscalar(fiberId):
            if result.size == 0:
                raise RuntimeError(f"No fiber with fiberId={fiberId}")
            return result.item()
        return result

    def getIdentityFromIndex(self, index):
        """Return the identity of the target indicated by the index

        Parameters
        ----------
        index : scalar or iterable of `int`
            Index for ``self``.

        Returns
        -------
        identity : single or `list` of `dict`
            Keword-value pairs identifying the target(s).
        """
        def impl(index):
            """Implementation: get identity given index"""
            return dict(catId=self.catId[index], tract=self.tract[index], patch=self.patch[index],
                        objId=self.objId[index])
        try:
            return [impl(ii) for ii in index]
        except TypeError:  # index is not iterable
            return impl(index)

    def getIdentity(self, fiberId):
        """Return the identity of the target indicated by the fiber(s)

        Parameters
        ----------
        fiberId : scalar or iterable of `int`
            Fiber identifier.

        Returns
        -------
        identity : single or `list` of `dict`
            Keyword-value pairs identifying the target.
        """
        index = self.selectFiber(fiberId)
        return self.getIdentityFromIndex(index)

    def extractNominal(self, fiberId):
        """Extract nominal positions for fibers

        Parameters
        ----------
        fiberId : iterable of `int`
            Fiber identifiers.

        Returns
        -------
        nominal : `numpy.ndarray` of shape ``(N, 2)``
            Nominal position for each fiber.
        """
        index = np.nonzero(np.isin(self.fiberId, fiberId))[0]
        return self.pfiNominal[index]


class PfsConfig(PfsDesign):
    """The configuration of the PFS top-end for one or more observations

    The realised version of a `PfsDesign`.

    Parameters
    ----------
    pfsDesignId : `int`
        PFI design identifier, specifies the intended top-end configuration.
    visit : `int`
        Exposure identifier.
    raBoresight : `float`, degrees
        Right Ascension of telescope boresight.
    decBoresight : `float`, degrees
        Declination of telescope boresight.
    posAng : `float`, degrees
        The position angle from the
        North Celestial Pole to the PFI_Y axis,
        measured clockwise with respect to the
        positive PFI_Z axis
    arms : `str`
        arms that are exposed. Eg 'brn', 'bmn'.
    fiberId : `numpy.ndarary` of `int32`
        Fiber identifier for each fiber.
    tract : `numpy.ndarray` of `int32`
        Tract index for each fiber.
    patch : `numpy.ndarray` of `str`
        Patch indices for each fiber, typically two integers separated by a
        comma, e.g,. "5,6".
    ra : `numpy.ndarray` of `float64`
        Right Ascension for each fiber, degrees.
    dec : `numpy.ndarray` of `float64`
        Declination for each fiber, degrees.
    catId : `numpy.ndarray` of `int32`
        Catalog identifier for each fiber.
    objId : `numpy.ndarray` of `int64`
        Object identifier for each fiber. Specifies the object within the
        catalog.
    targetType : `numpy.ndarray` of `int`
        Type of target for each fiber. Values must be convertible to
        `TargetType` (which limits the range of values).
    fiberStatus : `numpy.ndarray` of `int`
        Status of each fiber. Values must be convertible to `FiberStatus`
        (which limits the range of values).
    epoch : `numpy.chararray`
        reference epoch for each fiber.
    pmRa : `numpy.ndarray` of `float32`
        Proper motion in direction of Right Ascension
        for each fiber, mas/year.
    pmDec : `numpy.ndarray` of `float32`
        Proper motion in direction of Declination
        for each fiber, mas/year.
    parallax : `numpy.ndarray` of `float32`
        parallax for each fiber, mas.
    proposalId : `numpy.chararray`
        Proposal ID of each fiber (e.g, S23A-001QN).
    obCode : `numpy.chararray`
        Code for an Observing Block (OB) of each fiber.
    fiberFlux : `list` of `numpy.ndarray` of `float`
        Array of fiber fluxes for each fiber, in [nJy].
    psfFlux : `list` of `numpy.ndarray` of `float`
        Array of PSF fluxes for each target/fiber in [nJy].
    totalFlux : `list` of `numpy.ndarray` of `float`
        Array of total fluxes for each target/fiber in [nJy].
    fiberFluxErr : `list` of `numpy.ndarray` of `float`
        Array of fiber flux errors for each fiber in [nJy].
    psfFluxErr : `list` of `numpy.ndarray` of `float`
        Array of PSF flux errors for each target/fiber in [nJy].
    totalFluxErr : `list` of `numpy.ndarray` of `float`
        Array of total flux errors for each target/fiber in [nJy].
    filterNames : `list` of `list` of `str`
        List of filters used to measure the fiber fluxes for each filter.
    pfiCenter : `numpy.ndarray` of `float`
        Actual position (2-vector) of each fiber on the PFI, millimeters.
    pfiNominal : `numpy.ndarray` of `float`
        Intended target position (2-vector) of each fiber on the PFI, millimeters.
    guideStars : `GuideStars`
        Guide star data. If `None`, an empty GuideStars instance will be created.
    designName : `str`, optional
        Human-readable name for the design.
    variant : `int`, optional
        Counter of which variant of `designId0` we are. Requires `designId0`.
    designId0 : `int`, optional
        pfsDesignId of the pfsDesign we are a variant of. Requires `variant`.
    """
    # Scalar values
    _scalars = ["pfsDesignId", "designName",
                "visit", "raBoresight", "decBoresight", "posAng", "arms", "guideStars",
                "variant", "designId0", "header"]

    # List of fields required, and their FITS type
    # Some elements of the code expect the following to be present:
    #     fiberId, targetType
    # fiberStatus is handled separately, for backwards-compatibility
    _fields = {"fiberId": "J",
               "tract": "J",
               "patch": "A",
               "ra": "D",
               "dec": "D",
               "catId": "J",
               "objId": "K",
               "targetType": "J",
               "epoch": "A",
               "pmRa": "E",
               "pmDec": "E",
               "parallax": "E",
               "proposalId": "A",
               "obCode": "A",
               "pfiNominal": "2E",
               "pfiCenter": "2E",
               }
    _pointFields = ["pfiNominal", "pfiCenter"]  # List of point fields; should be in _fields too
    _photometry = ["fiberFlux",
                   "psfFlux",
                   "totalFlux",
                   "fiberFluxErr",
                   "psfFluxErr",
                   "totalFluxErr",
                   "filterNames"]  # List of photometry fields
    _keywords = list(_fields) + _photometry
    _hduName = "CONFIG"

    fileNameFormat = "pfsConfig-0x%016x-%06d.fits"

    def __init__(self, pfsDesignId, visit, raBoresight, decBoresight,
                 posAng,
                 arms,
                 fiberId, tract, patch, ra, dec, catId, objId,
                 targetType, fiberStatus,
                 epoch, pmRa, pmDec, parallax,
                 proposalId, obCode,
                 fiberFlux,
                 psfFlux,
                 totalFlux,
                 fiberFluxErr,
                 psfFluxErr,
                 totalFluxErr,
                 filterNames, pfiCenter, pfiNominal,
                 guideStars,
                 designName="",
                 variant=0,
                 designId0=0,
                 header=None):
        self.visit = visit
        self.pfiCenter = np.array(pfiCenter)
        self.header = dict() if header is None else header
        super().__init__(pfsDesignId, raBoresight, decBoresight,
                         posAng,
                         arms,
                         fiberId, tract, patch, ra, dec,
                         catId, objId, targetType, fiberStatus,
                         epoch, pmRa, pmDec, parallax,
                         proposalId, obCode,
                         fiberFlux,
                         psfFlux,
                         totalFlux,
                         fiberFluxErr,
                         psfFluxErr,
                         totalFluxErr,
                         filterNames, pfiNominal,
                         guideStars,
                         designName,
                         variant=variant,
                         designId0=designId0)

    def __str__(self):
        """String representation"""
        return "PfsConfig(%d, %d, ...)" % (self.pfsDesignId, self.visit)

    @property
    def filename(self):
        """Usual filename"""
        return self.fileNameFormat % (self.pfsDesignId, self.visit)

    @classmethod
    def fromPfsDesign(cls, pfsDesign, visit, pfiCenter, header=None):
        """Construct from a ``PfsDesign``

        Parameters
        ----------
        pfsDesign : `pfs.datamodel.PfsDesign`
            ``PfsDesign`` to use as the base for this ``PfsConfig``.
        visit : `int`
            Exposure identifier.
        pfiCenter : `numpy.ndarray` of `float`
            Actual position (2-vector) of each fiber on the PFI, microns.

        Returns
        -------
        self : `PfsConfig`
            Constructed ``PfsConfig`.
        """
        keywords = pfsDesign._keywords + PfsDesign._scalars + ['fiberStatus']

        kwargs = {kk: getattr(pfsDesign, kk) for kk in keywords}
        kwargs["visit"] = visit
        kwargs["pfiCenter"] = pfiCenter
        kwargs["header"] = header

        return PfsConfig(**kwargs)

    @classmethod
    def _readHeader(cls, header, kwargs=None):
        """Read construction variables from the header.

        Parameters
        ----------
        header : `astropy.io.fits.Header`
            FITS header.
        kwargs : `dict`, optional
            Keyword arguments to be passed to the constructor; modified.

        Returns
        -------
        kwargs : `dict`
            Keyword arguments to be passed to the constructor.
        """
        kwargs = super()._readHeader(header, kwargs)

        # We formerly got the visit from the filename, which is fragile.
        # Now we look for it in the W_VISIT header, but need to allow for the possibility that it's
        # not present.
        visit = header.get("W_VISIT", None)
        if visit is not None:
            if "visit" in kwargs and kwargs["visit"] != visit:
                raise RuntimeError(f"visit mismatch: {kwargs['visit']} vs visit")
            kwargs["visit"] = visit
        elif "visit" not in kwargs:
            raise RuntimeError("Unable to determine visit")

        return kwargs

    def _writeHeader(self, header):
        """Write to the header.

        Parameters
        ----------
        header : `astropy.io.fits.Header`
            FITS header; modified.
        """
        super()._writeHeader(header)
        header["W_VISIT"] = (self.visit, "Visit number")
        header.update(self.header)

    @classmethod
    def read(cls, pfsDesignId, visit, dirName="."):
        """Construct from file

        Requires pyfits.

        Parameters
        ----------
        pfsDesignId : `int`
            PFI design identifier, specifies the intended top-end configuration.
        visit : `int`
            Exposure identifier.
        dirName : `str`, optional
            Directory from which to read the file. Defaults to the current
            directory.

        Returns
        -------
        self : `PfsConfig`
            Constructed `PfsConfig`.
        """
        filename = os.path.join(dirName, cls.fileNameFormat % (pfsDesignId, visit))
        return cls._readImpl(filename, pfsDesignId=pfsDesignId, visit=visit)

    def copy(self, **kwargs):
        """Copy pfsConfig, optionally changing entries

        Parameters
        -----------
        **kwargs
            Elements to override.

        Returns
        --------
        copy : `PfsConfig`
            Copied pfsConfig.
        """
        keywords = PfsConfig._keywords + PfsConfig._scalars + ['fiberStatus']
        return PfsConfig(**{key: kwargs.get(key, getattr(self, key)) for key in keywords})

    def extractCenters(self, fiberId):
        """Extract centers for fibers

        Parameters
        ----------
        fiberId : iterable of `int`
            Fiber identifiers.

        Returns
        -------
        centers : `numpy.ndarray` of shape ``(N, 2)``
            Center of each fiber.
        """
        index = np.nonzero(np.isin(self.fiberId, fiberId))[0]
        return self.pfiCenter[index]


PFSCONFIG_FILENAME_REGEX: str = r"^pfsConfig-(0x[0-9a-f]+)-([0-9]+)\.fits.*"
"""Regular expression to identify the pfsDesignId and visit number"""


def parsePfsConfigFilename(path: str) -> SimpleNamespace:
    """Parse path from the data butler

    We need to determine the ``pfsConfigId`` to pass to the
    `pfs.datamodel.PfsConfig` I/O methods.

    Parameters
    ----------
    path : `str`
        Path name from the LSST data butler. Besides the usual directory and
        filename with extension, this may include a suffix with additional
        characters added by the butler.

    Returns
    -------
    dirName : `str`
        Directory name.
    fileName : `str`
        Filename without directory.
    pfsDesignId : `int`
        PFS fiber configuration.
    visit : `int`
        PFS visit exposure identifier.
    """
    dirName, fileName = os.path.split(path)
    matches = re.search(PFSCONFIG_FILENAME_REGEX, fileName)

    if not matches:
        raise RuntimeError("Unable to parse filename: %s" % (fileName,))
    pfsDesignId = int(matches.group(1), 16)
    visit = int(matches.group(2))
    return SimpleNamespace(dirName=dirName, fileName=fileName, pfsDesignId=pfsDesignId, visit=visit)


def checkPfsConfigHeader(filename: str, allowFix: bool = False, log: Optional[Logger] = None) -> bool:
    """Check that a pfsConfig header includes the appropriate keywords

    These keywords include:
    - ``W_VISIT``: PFS exposure visit number
    - ``W_PFDSGN``: PFS fiber targeting

    If ``allowFix=True`` and the header requires fixing, we will back up the
    original contents (this is done by ``astropy``; it usually adds a ``.bak``
    to the end of the filename).

    Parameters
    ----------
    filename : `str`
        Name of file to check.
    allowFix : `bool`, optional
        Allow fixing the header if it is non-conformant; by default ``False``.
    log : `Logger`, optional
        Logger to use, or ``None`` for no logging.

    Returns
    -------
    modified : `bool`
        Did we modify the header?
    """
    from astropy import log
    try:
        data = parsePfsConfigFilename(filename)
    except RuntimeError:
        log.warning("Unable to parse filename: %s", filename)
    if log:
        log.info(f"Checking {filename}")
    with pyfits.open(filename, "update" if allowFix else "readonly", save_backup=True) as fits:
        header = fits[0].header
        modified = False
        try:
            modified |= checkHeaderKeyword(
                header, "W_VISIT", data.visit, "PFS exposure visit number", allowFix, log=log
            )
            modified |= checkHeaderKeyword(
                header, "W_PFDSGN", data.pfsDesignId, "PFS fiber configuration identifier", allowFix, log=log
            )
        except ValueError as exc:
            raise ValueError(f"Bad header for {filename}") from exc

        if modified and log:
            log.warning(f"Updated {filename}")
    return modified
