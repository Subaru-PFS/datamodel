import numpy as np
import os
import enum

try:
    import pyfits
except ImportError:
    pyfits = None


__all__ = ["TargetType", "PfsDesign", "PfsConfig"]

@enum.unique
class TargetType(enum.IntEnum):
    """Enumerated options for what a fiber is targeting

    * ``SCIENCE``: the fiber is intended to be on a science target.
    * ``SKY``: the fiber is intended to be on blank sky, and used for sky
      subtraction.
    * ``FLUXSTD``: the fiber is intended to be on a flux standard, and used for
      flux calibration.
    * ``BROKEN``: the fiber is broken, and any flux should be ignored.
    * ``BLOCKED``: the fiber is hidden behind its spot, and any flux should be
      ignored.
    """
    SCIENCE = 1
    SKY = 2
    FLUXSTD = 3
    BROKEN = 4
    BLOCKED = 5


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
    fiberMag : `list` of `numpy.ndarray` of `float`
        Array of fiber magnitudes for each fiber.
    filterNames : `list` of `list` of `str`
        List of filters used to measure the fiber magnitudes for each filter.
    pfiNominal : `numpy.ndarray` of `float`
        Intended target position (2-vector) of each fiber on the PFI, microns.
    """
    # List of fields required, and their FITS type
    # Some elements of the code expect the following to be present:
    #     fiberId, targetType
    _fields = {"fiberId": "J",
               "tract": "K",
               "patch": "A",
               "ra": "D",
               "dec": "D",
               "catId": "J",
               "objId": "K",
               "targetType": "J",
               "pfiNominal": "2E",
               }
    _pointFields = ["pfiNominal"]  # List of point fields; should be in _fields too
    _photometry = ["fiberMag", "filterNames"]  # List of photometry fields
    _keywords = list(_fields) + _photometry
    _hduName = "DESIGN"

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
        for ii, (mag, names) in enumerate(zip(self.fiberMag, self.filterNames)):
            if len(mag) != len(names):
                raise RuntimeError("Inconsistent lengths between fiberMag (%d) and filterNames (%d) "
                                   "for fiberId=%d" % (len(mag), len(names), self.fiberId[ii]))
        for nn in self._pointFields:
            matrix = getattr(self, nn)
            if matrix.shape != (len(self.fiberId), 2):
                raise RuntimeError("Wrong shape for %s: %s vs (%d,2)" % (nn, matrix.shape, len(self.fiberId)))

    def __init__(self, pfsDesignId, raBoresight, decBoresight,
                 fiberId, tract, patch, ra, dec, catId, objId,
                 targetType, fiberMag, filterNames, pfiNominal):
        self.pfsDesignId = pfsDesignId
        self.raBoresight = raBoresight
        self.decBoresight = decBoresight

        self.fiberId = np.array(fiberId)
        self.tract = np.array(tract)
        self.patch = patch
        self.ra = np.array(ra)
        self.dec = np.array(dec)
        self.catId = np.array(catId)
        self.objId = np.array(objId)
        self.targetType = np.array(targetType)
        self.fiberMag = [np.array(mags) for mags in fiberMag]
        self.filterNames = filterNames
        self.pfiNominal = np.array(pfiNominal)
        self.validate()

    def __len__(self):
        """Number of fibers"""
        return len(self.fiberId)

    def __str__(self):
        """String representation"""
        return "PfsDesign(%d, ...)" % (self.pfsDesignId)

    @property
    def filename(self):
        """Usual filename"""
        return self.fileNameFormat % (self.pfsDesignId)

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
            raise RuntimeError("I failed to import pyfits, so cannot read from disk")

        with pyfits.open(filename) as fd:
            phu = fd[0].header
            raBoresight = phu['RA']
            decBoresight = phu['DEC']
            data = fd[cls._hduName].data

            for nn in cls._fields:
                assert nn not in kwargs
                kwargs[nn] = data[nn]

            photometry = fd["PHOTOMETRY"].data

            fiberId = kwargs["fiberId"]
            fiberMag = {ii: [] for ii in fiberId}
            filterNames = {ii: [] for ii in fiberId}
            for row in photometry:
                fiberMag[row['fiberId']].append(row['fiberMag'])
                filterNames[row['fiberId']].append(row['filterName'])

        return cls(**kwargs, raBoresight=raBoresight, decBoresight=decBoresight,
                   fiberMag=[np.array(fiberMag[ii]) for ii in fiberId],
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
        if not pyfits:
            raise RuntimeError("I failed to import pyfits, so cannot write to disk")

        fits = pyfits.HDUList()

        hdr = pyfits.Header()
        hdr['RA'] = (self.raBoresight, "Telescope boresight RA, degrees")
        hdr['DEC'] = (self.decBoresight, "Telescope boresight Dec, degrees")
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
        fits.append(pyfits.BinTableHDU.from_columns(columns, hdr, name=self._hduName))

        numRows = sum(len(mag) for mag in self.fiberMag)
        fiberId = np.array(sum(([ii]*len(mags) for ii, mags in zip(self.fiberId, self.fiberMag)), []))
        fiberMag = np.array(sum((mag.tolist() for mag in self.fiberMag), []))
        filterNames = sum(self.filterNames, [])
        assert(len(fiberId) == numRows)
        assert(len(fiberMag) == numRows)
        assert(len(filterNames) == numRows)
        maxLength = max(len(ff) for ff in filterNames) if filterNames else 1

        fits.append(pyfits.BinTableHDU.from_columns([
            pyfits.Column(name='fiberId', format='J', array=fiberId),
            pyfits.Column(name='fiberMag', format='E', array=fiberMag),
            pyfits.Column(name='filterName', format='A%d' % maxLength, array=filterNames),
        ], hdr, name='PHOTOMETRY'))

        # clobber=True in writeto prints a message, so use open instead
        with open(filename, "w") as fd:
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
    visit : `int`
        Exposure identifier.
    raBoresight : `float`, degrees
        Right Ascension of telescope boresight.
    decBoresight : `float`, degrees
        Declination of telescope boresight.
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
    fiberMag : `list` of `numpy.ndarray` of `float`
        Array of fiber magnitudes for each fiber.
    filterNames : `list` of `list` of `str`
        List of filters used to measure the fiber magnitudes for each filter.
    pfiCenter : `numpy.ndarray` of `float`
        Actual position (2-vector) of each fiber on the PFI, microns.
    pfiNominal : `numpy.ndarray` of `float`
        Intended target position (2-vector) of each fiber on the PFI, microns.
    """
    # List of fields required, and their FITS type
    # Some elements of the code expect the following to be present:
    #     fiberId, targetType
    _fields = {"fiberId": "J",
               "tract": "K",
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
    _photometry = ["fiberMag", "filterNames"]  # List of photometry fields
    _keywords = list(_fields) + _photometry
    _hduName = "CONFIG"

    fileNameFormat = "pfsConfig-0x%016x-%06d.fits"

    def __init__(self, pfsDesignId, visit, raBoresight, decBoresight,
                 fiberId, tract, patch, ra, dec, catId, objId,
                 targetType, fiberMag, filterNames, pfiCenter, pfiNominal):
        self.visit = visit
        self.pfiCenter = np.array(pfiCenter)
        super().__init__(pfsDesignId, raBoresight, decBoresight, fiberId, tract, patch, ra, dec,
                         catId, objId, targetType, fiberMag, filterNames, pfiNominal)

    def __str__(self):
        """String representation"""
        return "PfsConfig(%d, %d, ...)" % (self.pfsDesignId, self.visit)

    @property
    def filename(self):
        """Usual filename"""
        return self.fileNameFormat % (self.pfsDesignId, self.visit)

    @classmethod
    def fromPfsDesign(cls, pfsDesign, visit, pfiCenter):
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
        keywords = ["pfsDesignId", "raBoresight", "decBoresight"]
        kwargs = {kk: getattr(pfsDesign, kk) for kk in pfsDesign._keywords + keywords}
        kwargs["visit"] = visit
        kwargs["pfiCenter"] = pfiCenter
        return PfsConfig(**kwargs)

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
