import numpy as np
import os
import enum

try:
    import pyfits
except ImportError:
    pyfits = None


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


class PfsConfig:
    """The configuration of the PFS top-end for one or more observations

    Parameters
    ----------
    pfiDesignId : `int`
        PFI design identifier, specifies the intended top-end configuration.
    expId : `int`
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

    fileNameFormat = "pfsConfig-0x%016x-%06d.fits"

    def __init__(self, pfiDesignId, expId, raBoresight, decBoresight,
                 fiberId, tract, patch, ra, dec, catId, objId,
                 targetType, fiberMag, filterNames, pfiCenter, pfiNominal):
        if len(set([
            len(fiberId),
            len(tract),
            len(patch),
            len(ra),
            len(dec),
            len(catId),
            len(objId),
            len(targetType),
            len(fiberMag),
            len(filterNames),
            len(pfiCenter),
            len(pfiNominal),
        ])) != 1:
            raise RuntimeError("Inconsistent lengths: fiberId %d, tract %d, patch %d, ra %d, dec %d, "
                               "catId %d, objId %d, targetType %d, fiberMag %d, filterNames %d, "
                               "pfiCenter %d, pfiNominal %d" %
                               (len(fiberId), len(tract), len(patch), len(ra), len(dec),
                                len(catId), len(objId), len(targetType), len(fiberMag), len(filterNames),
                                len(pfiCenter), len(pfiNominal)))
        for ii, tt in enumerate(targetType):
            try:
                TargetType(tt)
            except ValueError as exc:
                raise ValueError("targetType[%d] = %d is not a recognized TargetType" % (ii, tt)) from exc
        for ii, (mag, names) in enumerate(zip(fiberMag, filterNames)):
            if len(mag) != len(names):
                raise RuntimeError("Inconsistent lengths between fiberMag (%d) and filterNames (%d) "
                                   "for fiberId=%d" % (len(mag), len(names), fiberId[ii]))
        if pfiCenter.shape != (len(fiberId), 2):
            raise RuntimeError("Wrong shape for pfiCenter: %s vs (%d,2)" % (pfiCenter.shape, len(fiberId)))
        if pfiNominal.shape != (len(fiberId), 2):
            raise RuntimeError("Wrong shape for pfiNominal: %s vs (%d,2)" % (pfiNominal.shape, len(fiberId)))

        self.pfiDesignId = pfiDesignId
        self.expId = expId
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
        self.pfiCenter = np.array(pfiCenter)
        self.pfiNominal = np.array(pfiNominal)

    def __len__(self):
        """Number of fibers"""
        return len(self.fiberId)

    def __str__(self):
        """String representation"""
        return "PfsConfig(%d, %d, ...)" % (self.pfiDesignId, self.expId)

    @property
    def filename(self):
        """Usual filename"""
        return self.fileNameFormat % (self.pfiDesignId, self.expId)

    @classmethod
    def read(cls, pfiDesignId, expId, dirName="."):
        """Construct from file

        Requires pyfits.

        Parameters
        ----------
        pfiDesignId : `int`
            PFI design identifier, specifies the intended top-end configuration.
        expId : `int`
            Exposure identifier.
        dirName : `str`, optional
            Directory from which to read the file. Defaults to the current
            directory.

        Returns
        -------
        self : `PfsConfig`
            Constructed `PfsConfig`.
        """
        if not pyfits:
            raise RuntimeError("I failed to import pyfits, so cannot read from disk")

        filename = os.path.join(dirName, cls.fileNameFormat % (pfiDesignId, expId))
        with pyfits.open(filename) as fd:
            phu = fd[0].header
            raBoresight = phu['RA']
            decBoresight = phu['DEC']
            data = fd["CONFIG"].data

            fiberId = data['fiberId']
            tract = data['tract']
            patch = data['patch']
            ra = data['ra']
            dec = data['dec']
            catId = data['catId']
            objId = data['objId']
            targetType = data['targetType']
            pfiCenter = data['pfiCenter']
            pfiNominal = data['pfiNominal']

            photometry = fd["PHOTOMETRY"].data

            fiberMag = {ii: [] for ii in fiberId}
            filterNames = {ii: [] for ii in fiberId}
            for row in photometry:
                fiberMag[row['fiberId']].append(row['fiberMag'])
                filterNames[row['fiberId']].append(row['filterName'])

        return cls(pfiDesignId=pfiDesignId, expId=expId, tract=tract, raBoresight=raBoresight,
                   decBoresight=decBoresight, patch=patch, fiberId=fiberId,
                   ra=ra, dec=dec, catId=catId, objId=objId, targetType=targetType,
                   fiberMag=[np.array(fiberMag[ii]) for ii in fiberId],
                   filterNames=[filterNames[ii] for ii in fiberId],
                   pfiCenter=pfiCenter, pfiNominal=pfiNominal)

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
        fits.append(pyfits.BinTableHDU.from_columns([
            pyfits.Column(name='fiberId', format='J', array=self.fiberId),
            pyfits.Column(name='tract', format='K', array=self.tract),
            pyfits.Column(name='patch', format='A%d' % maxLength, array=self.patch),
            pyfits.Column(name='ra', format='D', array=self.ra),
            pyfits.Column(name='dec', format='D', array=self.dec),
            pyfits.Column(name='catId', format='J', array=self.catId),
            pyfits.Column(name='objId', format='K', array=self.objId),
            pyfits.Column(name='targetType', format='J', array=self.targetType),
            pyfits.Column(name='pfiCenter', format='2E', array=self.pfiCenter),
            pyfits.Column(name='pfiNominal', format='2E', array=self.pfiNominal),
        ], hdr, name='CONFIG'))

        numRows = sum(len(mag) for mag in self.fiberMag)
        fiberId = np.array(sum(([ii]*len(mag) for ii, mag in enumerate(self.fiberMag)), []))
        fiberMag = np.array(sum((mag.tolist() for mag in self.fiberMag), []))
        filterNames = np.array(sum(self.filterNames, []))
        assert(len(fiberId) == numRows)
        assert(len(fiberMag) == numRows)
        assert(len(filterNames) == numRows)
        maxLength = max(len(ff) for ff in filterNames)

        fits.append(pyfits.BinTableHDU.from_columns([
            pyfits.Column(name='fiberId', format='J', array=fiberId),
            pyfits.Column(name='fiberMag', format='E', array=fiberMag),
            pyfits.Column(name='filterName', format='A%d' % maxLength, array=filterNames),
        ], hdr, name='PHOTOMETRY'))

        # clobber=True in writeto prints a message, so use open instead
        if fileName is None:
            fileName = self.fileNameFormat % (self.pfiDesignId, self.expId)
        with open(os.path.join(dirName, fileName), "w") as fd:
            fits.writeto(fd)
