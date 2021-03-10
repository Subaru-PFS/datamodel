import numpy as np
import os
import enum

try:
    import astropy.io.fits as pyfits
except ImportError:
    pyfits = None

from .utils import spectrographFromFiberId, fiberHoleFromFiberId


__all__ = ("TargetType", "FiberStatus", "PfsDesign", "PfsConfig")


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


class TargetType(DocEnum):
    """Enumerated options for what a fiber is targeting"""
    SCIENCE = 1, "science target"
    SKY = 2, "blank sky; used for sky subtraction"
    FLUXSTD = 3, "flux standard; used for fluxcal"
    UNASSIGNED = 4, "no particular target"
    ENGINEERING = 5, "engineering fiber"
    SUNSS_IMAGING = 6, "fiber goes to the SuNSS imaging leg"
    SUNSS_DIFFUSE = 7, "fiber goes to the SuNSS diffuse leg"


class FiberStatus(DocEnum):
    """Enumerated options for the status of a fiber"""
    GOOD = 1, "working normally"
    BROKENFIBER = 2, "broken; ignore any flux"
    BLOCKED = 3, "temporarily blocked; ignore any flux"
    BLACKSPOT = 4, "hidden behind spot; ignore any flux"
    UNILLUMINATED = 5, "not illuminated; ignore any flux"


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
        Intended target position (2-vector) of each fiber on the PFI, microns.
    """
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
               "pfiNominal": "2E",
               }
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
            If there are inconsistent lengths.
        ValueError:
            If the ``targetType`` is not recognised.
        """
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

    def __init__(self, pfsDesignId, raBoresight, decBoresight,
                 posAng,
                 arms,
                 fiberId, tract, patch, ra, dec, catId, objId,
                 targetType, fiberStatus,
                 fiberFlux,
                 psfFlux,
                 totalFlux,
                 fiberFluxErr,
                 psfFluxErr,
                 totalFluxErr,
                 filterNames, pfiNominal):
        self.pfsDesignId = pfsDesignId
        self.raBoresight = raBoresight
        self.decBoresight = decBoresight
        self.posAng = posAng
        self.arms = arms
        self.fiberId = np.array(fiberId)
        self.tract = np.array(tract)
        self.patch = patch
        self.ra = np.array(ra)
        self.dec = np.array(dec)
        self.catId = np.array(catId)
        self.objId = np.array(objId)
        self.targetType = np.array(targetType)
        self.fiberStatus = np.array(fiberStatus)
        self.fiberFlux = [np.array(flux) for flux in fiberFlux]
        self.psfFlux = [np.array(pflux) for pflux in psfFlux]
        self.totalFlux = [np.array(tflux) for tflux in totalFlux]
        self.fiberFluxErr = [np.array(ffErr) for ffErr in fiberFluxErr]
        self.psfFluxErr = [np.array(pfErr) for pfErr in psfFluxErr]
        self.totalFluxErr = [np.array(tfErr) for tfErr in totalFluxErr]
        self.filterNames = filterNames
        self.pfiNominal = np.array(pfiNominal)
        self.validate()

    def __len__(self):
        """Number of fibers"""
        return len(self.fiberId)

    def __str__(self):
        """String representation"""
        return "PfsDesign(%d, ...)" % (self.pfsDesignId)

    def __getitem__(self, index):
        """Get target by index

        Parameters
        ----------
        index : `int`
            Index of interest.

        Returns
        -------
        target : `pfs.datamodel.Target`
            Target data.
        """
        from pfs.datamodel.target import Target  # noqa: prevent circular import dependency
        return Target.fromPfsConfig(self, index)

    @property
    def filename(self):
        """Usual filename"""
        return self.fileNameFormat % (self.pfsDesignId)

    @property
    def spectrograph(self):
        """Return spectrograph number"""
        return spectrographFromFiberId(self.fiberId)

    @property
    def fiberHole(self):
        """Return fiber hole number"""
        return fiberHoleFromFiberId(self.fiberId)

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

            # If POSANG does not exist, use default.
            # This default should be removed when the
            # relevant test datasets have this keyword
            # populated.
            posAng = phu.get('POSANG', cls._POSANG_DEFAULT)

            # If ARM does not exist, use default.
            # This action should be removed once the
            # relevant test datasets have this keyword
            # populated.
            arms = phu.get('ARMS', 'brn')

            data = fd[cls._hduName].data

            for nn in cls._fields:
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

        return cls(**kwargs, raBoresight=raBoresight, decBoresight=decBoresight,
                   posAng=posAng,
                   arms=arms,
                   fiberFlux=[np.array(fiberFlux[ii]) for ii in fiberId],
                   psfFlux=[np.array(psfFlux[ii]) for ii in fiberId],
                   totalFlux=[np.array(totalFlux[ii]) for ii in fiberId],
                   fiberFluxErr=[np.array(fiberFluxErr[ii]) for ii in fiberId],
                   psfFluxErr=[np.array(psfFluxErr[ii]) for ii in fiberId],
                   totalFluxErr=[np.array(totalFluxErr[ii]) for ii in fiberId],
                   filterNames=[filterNames[ii] for ii in fiberId])

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

    def _writeImpl(self, filename):
        # NOTE: When making any changes to this method that modify the output
        # format, increment the DAMD_VER header value and record the change in
        # the versions.txt file.
        if not pyfits:
            raise RuntimeError("I failed to import astropy.io.fits, so cannot write to disk")

        fits = pyfits.HDUList()

        hdr = pyfits.Header()
        hdr['RA'] = (self.raBoresight, "Telescope boresight RA, degrees")
        hdr['DEC'] = (self.decBoresight, "Telescope boresight Dec, degrees")
        hdr['POSANG'] = (self.posAng, "PFI position angle, degrees")
        hdr['ARMS'] = (self.arms, "Exposed arms")
        hdr['DAMD_VER'] = (1, "PfsDesign/PfsConfig datamodel version")
        hdr.update(TargetType.getFitsHeaders())
        hdr.update(FiberStatus.getFitsHeaders())
        hdu = pyfits.PrimaryHDU(header=hdr)
        hdr.update()
        fits.append(hdu)

        maxLength = max(len(pp) for pp in self.patch)
        columns = []
        for name in self._fields:
            format = self._fields[name]
            if format == "A":
                maxLength = max(len(ss) for ss in getattr(self, name))
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

        # clobber=True in writeto prints a message, so use open instead
        with open(filename, "wb") as fd:
            fits.writeto(fd)

    def write(self, dirName=".", fileName=None):
        """Write to file

        Requires pyfits.

        Parameters
        ----------
        dirName : `str`, optional
            Directory to which to write the file. Defaults to the current
            directory.
        fileName : `str`, optional
            Filename to which to write. Defaults to using the filename template.
        """
        if fileName is None:
            fileName = self.filename
        self._writeImpl(os.path.join(dirName, fileName))

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
        selected = set(self.fiberId[select])
        return np.array([ii for ii, ff in enumerate(fiberId) if ff in selected])

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
        selected = set(self.fiberId[select])
        return np.array([ii for ii, ff in enumerate(fiberId) if ff in selected])

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

        Returns the index for the provided fiber identifier.

        Parameters
        ----------
        fiberId : iterable of `int`
            Fiber identifiers to select.

        Returns
        -------
        index : array-like of `int`
            Indices for fiber.
        """
        def impl(fiberId):
            """Implementation: get index of fiber"""
            return np.nonzero(self.fiberId == fiberId)[0]

        try:
            return np.array([impl(ff) for ff in fiberId])
        except TypeError:  # fiberId is not iterable
            return impl(fiberId)

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
        index = np.array([np.argwhere(self.fiberId == ff)[0][0] for ff in fiberId])
        return self.pfiNominal[index]


class PfsConfig(PfsDesign):
    """The configuration of the PFS top-end for one or more observations

    The realised version of a `PfsDesign`.

    Parameters
    ----------
    pfsDesignId : `int`
        PFI design identifier, specifies the intended top-end configuration.
    visit0 : `int`
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
        Actual position (2-vector) of each fiber on the PFI, microns.
    pfiNominal : `numpy.ndarray` of `float`
        Intended target position (2-vector) of each fiber on the PFI, microns.
    """
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

    def __init__(self, pfsDesignId, visit0, raBoresight, decBoresight,
                 posAng,
                 arms,
                 fiberId, tract, patch, ra, dec, catId, objId,
                 targetType, fiberStatus,
                 fiberFlux,
                 psfFlux,
                 totalFlux,
                 fiberFluxErr,
                 psfFluxErr,
                 totalFluxErr,
                 filterNames, pfiCenter, pfiNominal):
        self.visit0 = visit0
        self.pfiCenter = np.array(pfiCenter)
        super().__init__(pfsDesignId, raBoresight, decBoresight,
                         posAng,
                         arms,
                         fiberId, tract, patch, ra, dec,
                         catId, objId, targetType, fiberStatus,
                         fiberFlux,
                         psfFlux,
                         totalFlux,
                         fiberFluxErr,
                         psfFluxErr,
                         totalFluxErr,
                         filterNames, pfiNominal)

    def __str__(self):
        """String representation"""
        return "PfsConfig(%d, %d, ...)" % (self.pfsDesignId, self.visit0)

    @property
    def filename(self):
        """Usual filename"""
        return self.fileNameFormat % (self.pfsDesignId, self.visit0)

    @classmethod
    def fromPfsDesign(cls, pfsDesign, visit0, pfiCenter):
        """Construct from a ``PfsDesign``

        Parameters
        ----------
        pfsDesign : `pfs.datamodel.PfsDesign`
            ``PfsDesign`` to use as the base for this ``PfsConfig``.
        visit0 : `int`
            Exposure identifier.
        pfiCenter : `numpy.ndarray` of `float`
            Actual position (2-vector) of each fiber on the PFI, microns.

        Returns
        -------
        self : `PfsConfig`
            Constructed ``PfsConfig`.
        """
        keywords = ["pfsDesignId", "raBoresight", "decBoresight", "posAng", "arms"]
        kwargs = {kk: getattr(pfsDesign, kk) for kk in pfsDesign._keywords + keywords}
        kwargs["fiberStatus"] = pfsDesign.fiberStatus
        kwargs["visit0"] = visit0
        kwargs["pfiCenter"] = pfiCenter
        return PfsConfig(**kwargs)

    @classmethod
    def read(cls, pfsDesignId, visit0, dirName="."):
        """Construct from file

        Requires pyfits.

        Parameters
        ----------
        pfsDesignId : `int`
            PFI design identifier, specifies the intended top-end configuration.
        visit0 : `int`
            Exposure identifier.
        dirName : `str`, optional
            Directory from which to read the file. Defaults to the current
            directory.

        Returns
        -------
        self : `PfsConfig`
            Constructed `PfsConfig`.
        """
        filename = os.path.join(dirName, cls.fileNameFormat % (pfsDesignId, visit0))
        return cls._readImpl(filename, pfsDesignId=pfsDesignId, visit0=visit0)

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
        index = np.array([np.argwhere(self.fiberId == ff)[0][0] for ff in fiberId])
        return self.pfiCenter[index]
