import os
import re
from types import SimpleNamespace
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type
import warnings

import numpy as np
import astropy.io.fits
from scipy.interpolate import CubicSpline

from .utils import astropyHeaderFromDict, astropyHeaderToDict
from .identity import CalibIdentity

__all__ = (
    "Box",
    "Spline",
    "PfsDetectorMap",
    "SplinedDetectorMap",
    "GlobalDetectorModelScaling",
    "DifferentialDetectorMap",
    "DistortedDetectorMap",
    "DoubleDetectorMap",
    "PolynomialDetectorMap",
    "PfsDistortion",
    "DoubleDistortion",
    "MultipleDistortionsDetectorMap",
)


class Box:
    """Struct for a 2-dimensional box

    This is a less-functional version of LSST's ``lsst.geom.Box2I`, intended to
    simply group the parameters together. Methods are provided for conversion
    to/from LSST's ``Box2I``.

    Parameters
    ----------
    xMin : `int`
        Minimum value for x.
    yMin : `int`
        Minimum value for y.
    xMax : `int`
        Maximum value for x.
    yMax : `int`
        Maximum value for y.
    """
    def __init__(self, xMin, yMin, xMax, yMax):
        self.xMin = xMin
        self.yMin = yMin
        self.xMax = xMax
        self.yMax = yMax

    @classmethod
    def fromLsst(cls, box):
        """Convert from an LSST Box object

        Parameters
        ----------
        box : `lsst.geom.Box2I`
            LSST Box object.

        Returns
        -------
        self : `Box`
            Box struct.
        """
        return cls(box.getMinX(), box.getMinY(), box.getMaxX(), box.getMaxY())

    def toLsst(self):
        """Convert to an LSST Box object

        Returns
        -------
        box : `lsst.geom.Box2I`
            LSST Box object.
        """
        from lsst.geom import Box2I, Point2I
        return Box2I(Point2I(self.xMin, self.yMin), Point2I(self.xMax, self.yMax))

    @classmethod
    def fromFitsHeader(cls, header):
        """Construct from FITS header values

        Parameters
        ----------
        header : `dict`
            FITS header.

        Returns
        -------
        self : `Box`
            Box struct.
        """
        return cls(header["MINX"], header["MINY"], header["MAXX"], header["MAXY"])

    def toFitsHeader(self):
        """"Generate FITS header values

        Returns
        -------
        header : `dict`
            FITS header.
        """
        header = {}
        header["MINX"] = self.xMin
        header["MINY"] = self.yMin
        header["MAXX"] = self.xMax
        header["MAXY"] = self.yMax
        return header


class Spline:
    """Struct for a 1-dimensional spline

    This is a less-functional version of `pfs.drp.stella.SplineF, intended to
    simply group the knots and values together. An evaluation method is provided
    as a convenience.

    Parameters
    ----------
    knots : array-like, shape ``(N,)``
        Spline knot positions.
    values : array-like, shape ``(N,)``
        Values at the knots.
    """
    def __init__(self, knots, values):
        self.knots = knots
        self.values = values
        self._impl = CubicSpline(self.knots, self.values, bc_type='not-a-knot', extrapolate=False)

    def __call__(self, x):
        """Evaluate the spline

        Parameters
        ----------
        x : array-like
            Positions at which to evaluate the spline.

        Returns
        -------
        y : array-like
            Spline values.
        """
        return self._impl(x)


class PfsDetectorMap(ABC):
    """Base class for mapping between fiberId,wavelength and x,y on the detector

    The detectorMap implementations in the datamodel package handle I/O only.
    For fully-functional implementations that include evaluation of the
    mappings, see the drp_stella package.

    Subclasses must implement the ``__init__``, ``_readImpl`` and ``_writeImpl``
    methods.
    """
    filenameFormat = "pfsDetectorMap-%(visit0)06d-%(arm)1s%(spectrograph)1d.fits"
    """Format for filename (`str`)

    Should include formatting directives for the ``identity`` dict.
    """

    filenameRegex = (r"pfsDetectorMap-(?P<visit0>\d{6})-(?P<arm>\S)(?P<spectrograph>\d).fits")
    """Regex for extracting dataId from filename (`str`)

    Should capture the regex capture directives for the ``identity`` dict.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("This abstract base class cannot be instantiated directly")

    def __str__(self):
        """Stringify"""
        return f"{self.__class__.__name__}{{{len(self)} fibers}}"

    @property
    def filename(self):
        """Filename, without directory"""
        return self.getFilename(self.identity)

    @classmethod
    def getFilename(cls, identity):
        """Calculate filename

        Parameters
        ----------
        identity : `pfs.datamodel.CalibIdentity`
            Identity of the data.

        Returns
        -------
        filename : `str`
            Filename, without directory.
        """
        return cls.filenameFormat % identity.getDict()

    @classmethod
    def parseFilename(cls, path):
        """Parse filename to get the file's identity

        Uses the class attributes ``filenameRegex`` and ``filenameKeys`` to
        construct the identity from the filename.

        Parameters
        ----------
        path : `str`
            Path to the file of interest.

        Returns
        -------
        identity : `pfs.datamodel.CalibIdentity`
            Identity of the data of interest.
        """
        dirName, fileName = os.path.split(path)
        matches = re.search(cls.filenameRegex, fileName)
        if not matches:
            raise RuntimeError("Unable to parse filename: %s" % (fileName,))
        identity = matches.groupdict()
        identity["obsDate"] = None  # Not exposed in the detectorMap filename
        return CalibIdentity.fromDict(identity)

    @classmethod
    def readFits(cls, filename):
        """Read from FITS file

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
        identity = cls.parseFilename(filename)
        import astropy.io.fits
        with astropy.io.fits.open(filename) as fits:
            return cls._readImpl(fits, identity)

    @classmethod
    def read(cls, identity, dirName="."):
        """Read file given an identity

        This API is intended for use by science users, as it allows selection
        of the correct file by identity (e.g., visit, arm, spectrograph),
        without knowing the file naming convention.

        Parameters
        ----------
        identity : `pfs.datamodel.CalibIdentity`
            Identification of the calib data of interest.
        dirName : `str`, optional
            Directory from which to read.

        Returns
        -------
        self : `PfsFiberArraySet`
            Spectra read from file.
        """
        import astropy.io.fits
        filename = os.path.join(dirName, cls.getFilename(identity))
        with astropy.io.fits.open(filename) as fits:
            return cls._readImpl(fits, identity)

    @classmethod
    @abstractmethod
    def _readImpl(cls, fits, identity):
        """Implementation of reading from a FITS file in memory

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file in memory.
        identity : `pfs.datamodel.CalibIdentity`
            Identity of the calib data.

        Returns
        -------
        self : `SplinedDetectorMap`
            DetectorMap read from FITS file.
        """
        subclasses = {ss.__name__: ss for ss in cls.__subclasses__()}  # ignores sub-sub-classes
        name = fits[0].header.get("pfs_detectorMap_class", "SplinedDetectorMap")
        if name not in subclasses:
            raise RuntimeError(f"Unrecognised pfs_detectorMap_class value: {name}")
        return subclasses[name]._readImpl(fits, identity)

    def writeFits(self, filename):
        """Write to FITS file

        This API is intended for use by the LSST data butler, which handles
        translating the desired identity into a filename.

        Parameters
        ----------
        filename : `str`
            Filename of FITS file.
        """
        fits = self._writeImpl()
        fits.writeto(filename, overwrite=True)

    def write(self, dirName="."):
        """Write to file

        This API is intended for use by science users, as it allows setting the
        correct filename from parameters that make sense, such as which
        exposure, spectrograph, etc.

        Parameters
        ----------
        dirName : `str`, optional
            Directory to which to write.
        """
        filename = os.path.join(dirName, self.filename)
        self.writeFits(filename)

    @abstractmethod
    def _writeImpl(self):
        """Implementation of writing to FITS file

        Returns
        -------
        fits : `astropy.io.fits.HDUList`
            FITS file representation.
        """
        raise NotImplementedError("Subclasses must override this method")


class SplinedDetectorMap(PfsDetectorMap):
    """DetectorMap implemented with splines for individual fibers

    This implementation handles I/O only. For a fully-functional implementation
    that includes evaluation of the mappings, see the drp_stella package.

    Parameters
    ----------
    identity : `pfs.datamodel.CalibIdentity`
        Identity of the data of interest.
    box : `Box`
        Bounding box for detector.
    fiberId : `numpy.ndarray` of `int`, shape ``(N,)``
        Fiber identifiers.
    xSplines : iterable of `Spline`
        Spline for each fiber with trace center as a function of row.
    wavelengthSplines : iterable of `Spline`
        Spline for each fiber with wavelength as a function of row.
    spatialOffsets : `numpy.ndarray` of `float`, shape ``(N,)``
        Slit offsets in the spatial dimension for each fiber.
    spectralOffsets : `numpy.ndarray` of `float`, shape ``(N,)``
        Slit offsets in the spectral dimension for each fiber.
    metadata : `dict`
        Keyword-value pairs to put in the header.
    """
    def __init__(self, identity, box, fiberId, xSplines, wavelengthSplines,
                 spatialOffsets, spectralOffsets, metadata):
        self.identity = identity
        self.box = box
        self.fiberId = fiberId
        self.xSplines = xSplines
        self.wavelengthSplines = wavelengthSplines
        self.spatialOffsets = spatialOffsets
        self.spectralOffsets = spectralOffsets
        self.metadata = metadata
        self.validate()

    def validate(self):
        """Ensure that array lengths are as expected

        Raises
        ------
        AssertionError
            When an array length doesn't match that expected.
        """
        length = len(self.fiberId)
        assert len(self.xSplines) == length
        assert len(self.wavelengthSplines) == length
        assert len(self.spatialOffsets) == length
        assert len(self.spectralOffsets) == length

    def __len__(self):
        """Number of fibers"""
        return len(self.fiberId)

    @classmethod
    def _readImpl(cls, fits, identity):
        """Implementation of reading from a FITS file in memory

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file in memory.
        identity : `pfs.datamodel.CalibIdentity`
            Identity of the calib data.

        Returns
        -------
        self : `SplinedDetectorMap`
            DetectorMap read from FITS file.
        """
        header = astropyHeaderToDict(fits[0].header)
        box = Box.fromFitsHeader(header)
        fiberId = fits["FIBERID"].data.astype(np.int32)   # astype() forces machine-native byte order
        numFibers = len(fiberId)

        slitOffsets = fits["SLITOFF"].data
        spatialOffsets = slitOffsets[0].astype(float)
        spectralOffsets = slitOffsets[1].astype(float)

        # array.astype() required to force byte swapping (dtype('>f4') --> np.float32)
        # otherwise pybind doesn't recognise them as the proper type.
        centerTable = fits["CENTER"].data
        centerIndexData = centerTable["index"]
        centerKnotsData = centerTable["knot"].astype(float)
        centerValuesData = centerTable["value"].astype(float)
        centerSplines = []
        for ii in range(numFibers):
            select = centerIndexData == ii
            centerSplines.append(Spline(centerKnotsData[select], centerValuesData[select]))

        wavelengthTable = fits["WAVELENGTH"].data
        wavelengthIndexData = wavelengthTable["index"]
        wavelengthKnotsData = wavelengthTable["knot"].astype(float)
        wavelengthValuesData = wavelengthTable["value"].astype(float)
        wavelengthSplines = []
        for ii in range(numFibers):
            select = wavelengthIndexData == ii
            wavelengthSplines.append(Spline(wavelengthKnotsData[select], wavelengthValuesData[select]))

        return cls(identity, box, fiberId, centerSplines, wavelengthSplines,
                   spatialOffsets, spectralOffsets, header)

    def _writeImpl(self):
        """Implementation of writing to FITS file

        Returns
        -------
        fits : `astropy.io.fits.HDUList`
            FITS file representation.
        """
        # NOTE: When making any changes to this method that modify the output
        # format, increment the DAMD_VER header value and record the change in
        # the versions.txt file.

        numFibers = len(self)
        slitOffsets = np.zeros((3, numFibers))
        slitOffsets[0] = self.spatialOffsets
        slitOffsets[1] = self.spectralOffsets
        # slitOffsets[2] is focus (backward compatibility), but we're not using that

        centerKnots = [ss.knots for ss in self.xSplines]
        centerValues = [ss.values for ss in self.xSplines]
        wavelengthKnots = [ss.knots for ss in self.wavelengthSplines]
        wavelengthValues = [ss.values for ss in self.wavelengthSplines]

        centerIndex = np.array(sum(([ii]*len(vv) for ii, vv in enumerate(centerKnots)), []))
        centerKnots = np.concatenate(centerKnots)
        centerValues = np.concatenate(centerValues)
        wavelengthIndex = np.array(sum(([ii]*len(vv) for ii, vv in enumerate(wavelengthKnots)), []))
        wavelengthKnots = np.concatenate(wavelengthKnots)
        wavelengthValues = np.concatenate(wavelengthValues)

        #
        # OK, we've unpacked the DetectorMap; time to write the contents to disk
        #
        fits = astropy.io.fits.HDUList()
        header = self.metadata.copy()
        header.update(self.box.toFitsHeader())
        if "pfs_detectorMap_class" in header:
            del header["pfs_detectorMap_class"]
        header = astropyHeaderFromDict(header)
        header["OBSTYPE"] = "detectorMap"
        header["HIERARCH pfs_detectorMap_class"] = "SplinedDetectorMap"

        # NOTE: The datamodel version also gets incremented here for the DifferentialDetectorMap
        header['DAMD_VER'] = (2, "SplinedDetectorMap datamodel version")

        phu = astropy.io.fits.PrimaryHDU(header=header)
        fits.append(phu)

        hdu = astropy.io.fits.ImageHDU(self.fiberId, name="FIBERID")
        hdu.header["INHERIT"] = True
        fits.append(hdu)

        hdu = astropy.io.fits.ImageHDU(slitOffsets, name="SLITOFF")
        hdu.header["INHERIT"] = True
        fits.append(hdu)

        hdu = astropy.io.fits.BinTableHDU.from_columns([
            astropy.io.fits.Column(name="index", format="K", array=centerIndex),
            astropy.io.fits.Column(name="knot", format="D", array=centerKnots),
            astropy.io.fits.Column(name="value", format="D", array=centerValues),
        ], name="CENTER")
        hdu.header["INHERIT"] = True
        fits.append(hdu)

        hdu = astropy.io.fits.BinTableHDU.from_columns([
            astropy.io.fits.Column(name="index", format="K", array=wavelengthIndex),
            astropy.io.fits.Column(name="knot", format="D", array=wavelengthKnots),
            astropy.io.fits.Column(name="value", format="D", array=wavelengthValues),
        ], name="WAVELENGTH")
        hdu.header["INHERIT"] = True
        fits.append(hdu)
        return fits


class GlobalDetectorModelScaling(SimpleNamespace):
    """Struct for parameters that set scaling of a global detector model

    Parameters
    ----------
    fiberPitch : `float`
        Distance between fibers (pixels).
    dispersion : `float`
        Linear wavelength dispersion (nm per pixel)
    wavelengthCenter : `float`
        Central wavelength (nm).
    minFiberId : `int`
        Minimum fiberId value.
    maxFiberId : `int`
        Maximum fiberId value.
    height : `int`
        Height of detector (wavelength dimension; pixel).
    buffer : `float`
        Fraction of expected wavelength range by which to expand the wavelength
        range in the polynomials; this accounts for distortion or small
        inaccuracies in the dispersion.
    """
    def __init__(self, fiberPitch, dispersion, wavelengthCenter, minFiberId, maxFiberId, height, buffer):
        super().__init__(fiberPitch=fiberPitch, dispersion=dispersion, wavelengthCenter=wavelengthCenter,
                         minFiberId=minFiberId, maxFiberId=maxFiberId, height=height, buffer=buffer)

    @classmethod
    def fromFitsHeader(cls, header):
        """Construct from FITS header

        Parameters
        ----------
        header : `dict`
            FITS header.

        Returns
        -------
        self : `GlobalDetectorModelScaling`
            Constructed object.
        """
        def lookup(header, keyword):
            """Look up value of ``keyword`` in ``header``

            Tries uppercase version of keyword if the original is not present,
            for backwards compatibility.
            """
            return header[keyword] if keyword in header else header[keyword.upper()]

        return cls(
            fiberPitch=lookup(header, "SCALING.fiberPitch"),
            dispersion=lookup(header, "SCALING.dispersion"),
            wavelengthCenter=lookup(header, "SCALING.wavelengthCenter"),
            minFiberId=int(lookup(header, "SCALING.minFiberId")),
            maxFiberId=int(lookup(header, "SCALING.maxFiberId")),
            height=int(lookup(header, "SCALING.height")),
            buffer=lookup(header, "SCALING.buffer"),
        )

    def toFitsHeader(self):
        """Write to FITS header

        Returns
        -------
        header : `dict`
            FITS header.
        """
        return {
            "SCALING.fiberPitch": self.fiberPitch,
            "SCALING.dispersion": self.dispersion,
            "SCALING.wavelengthCenter": self.wavelengthCenter,
            "SCALING.minFiberId": self.minFiberId,
            "SCALING.maxFiberId": self.maxFiberId,
            "SCALING.height": self.height,
            "SCALING.buffer": self.buffer,
        }


class DifferentialDetectorMap(PfsDetectorMap):
    """DetectorMap implemented as a model relative to another detectorMap

    This implementation handles I/O only. For a fully-functional implementation
    that includes evaluation of the mappings, see the drp_stella package.

    Parameters
    ----------
    identity : `pfs.datamodel.CalibIdentity`
        Identity of the data of interest.
    box : `Box`
        Bounding box for detector.
    base : `pfs.datamodel.SplinedDetectorMap`
        Base detectorMap.
    order : `int`
        Polynomial order.
    scaling : `GlobalDetectorModelScaling`
        Scaling parameters.
    fiberCenter : `float`
        Central fiberId, separating low- and high-fiberId CCDs.
    xCoeff : `numpy.ndarray` of `float`, shape ``(M,)``
        Coefficients for x distortion polynomial.
    yCoeff : `numpy.ndarray` of `float`, shape ``(M,)``
        Coefficients for y distortion polynomial.
    highCcdCoeff : `numpy.ndarray` of `float`, shape ``(6,)``
        Coefficients for high-fiberId CCD affine transform.
    metadata : `dict`
        Keyword-value pairs to put in the header.
    """
    def __init__(self, identity, box, base, order, scaling, fiberCenter,
                 xCoeff, yCoeff, highCcdCoeff, metadata):
        self.identity = identity
        self.box = box
        self.base = base
        self.order = order
        self.scaling = scaling
        self.fiberCenter = fiberCenter
        self.xCoeff = xCoeff
        self.yCoeff = yCoeff
        self.highCcdCoeff = highCcdCoeff
        self.metadata = metadata
        self.validate()

    def validate(self):
        """Ensure that array lengths are as expected

        Raises
        ------
        AssertionError
            When an array length doesn't match that expected.
        """
        numCoeff = (self.order + 1)*(self.order + 2)//2
        assert len(self.xCoeff) == numCoeff
        assert len(self.yCoeff) == numCoeff
        assert len(self.highCcdCoeff) == 6

    def __len__(self):
        """Number of fibers"""
        return len(self.fiberId)

    @classmethod
    def _readImpl(cls, fits, identity):
        """Implementation of reading from a FITS file in memory

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file in memory.
        identity : `pfs.datamodel.CalibIdentity`
            Identity of the calib data.

        Returns
        -------
        self : `DifferentialDetectorMap`
            DetectorMap read from FITS file.
        """
        header = astropyHeaderToDict(fits[0].header)
        box = Box.fromFitsHeader(header)
        order = header["ORDER"]

        base = SplinedDetectorMap._readImpl(fits, identity)
        scaling = GlobalDetectorModelScaling.fromFitsHeader(header)
        fiberCenter = header["FIBERCENTER"]

        xCoeff = fits["COEFFICIENTS"].data["x"].astype(float)
        yCoeff = fits["COEFFICIENTS"].data["y"].astype(float)
        rightCcd = fits["HIGHCCD"].data["coefficients"].astype(float)

        return cls(identity, box, base, order, scaling, fiberCenter,
                   xCoeff, yCoeff, rightCcd, header)

    def _writeImpl(self):
        """Implementation of writing to FITS file

        Returns
        -------
        fits : `astropy.io.fits.HDUList`
            FITS file representation.
        """
        # NOTE: When making any changes to this method that modify the output
        # format, increment the DAMD_VER header value in the
        # SplinedDetectorMap._writeImpl method, and record the change in
        # the versions.txt file.
        fits = self.base._writeImpl()

        header = self.metadata.copy()
        header.update(self.box.toFitsHeader())
        header.update(astropyHeaderFromDict(self.scaling.toFitsHeader()))
        if "pfs_detectorMap_class" in header:
            del header["pfs_detectorMap_class"]
        header["OBSTYPE"] = "detectorMap"
        header["HIERARCH pfs_detectorMap_class"] = "DifferentialDetectorMap"
        header["ORDER"] = self.order
        header["HIERARCH FIBERCENTER"] = self.fiberCenter
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=astropy.io.fits.verify.VerifyWarning)
            fits[0].header.update(astropyHeaderFromDict(header))

        tableHeader = astropy.io.fits.Header()
        tableHeader["INHERIT"] = True

        table = astropy.io.fits.BinTableHDU.from_columns([
            astropy.io.fits.Column(name="x", format="D", array=self.xCoeff),
            astropy.io.fits.Column(name="y", format="D", array=self.yCoeff),
        ], header=tableHeader, name="COEFFICIENTS")
        fits.append(table)

        table = astropy.io.fits.BinTableHDU.from_columns([
            astropy.io.fits.Column(name="coefficients", format="D", array=self.highCcdCoeff),
        ], header=tableHeader, name="HIGHCCD")
        fits.append(table)

        return fits


class DistortedDetectorMap(PfsDetectorMap):
    """DetectorMap implemented as a distortion on top of a SplinedDetectorMap

    This will replace DifferentialDetectorMap in the future.

    This implementation handles I/O only. For a fully-functional implementation
    that includes evaluation of the mappings, see the drp_stella package.

    Parameters
    ----------
    identity : `pfs.datamodel.CalibIdentity`
        Identity of the data of interest.
    box : `Box`
        Bounding box for detector.
    base : `pfs.datamodel.SplinedDetectorMap`
        Base detectorMap.
    order : `int`
        Polynomial order.
    xCoeff : `numpy.ndarray` of `float`, shape ``(M,)``
        Coefficients for x distortion polynomial.
    yCoeff : `numpy.ndarray` of `float`, shape ``(M,)``
        Coefficients for y distortion polynomial.
    rightCcdCoeff : `numpy.ndarray` of `float`, shape ``(6,)``
        Coefficients for right CCD affine transform.
    metadata : `dict`
        Keyword-value pairs to put in the header.
    """
    def __init__(self, identity, box, base, order, xCoeff, yCoeff, rightCcdCoeff, metadata):
        self.identity = identity
        self.box = box
        self.base = base
        self.order = order
        self.xCoeff = xCoeff
        self.yCoeff = yCoeff
        self.rightCcdCoeff = rightCcdCoeff
        self.metadata = metadata
        self.validate()

    def validate(self):
        """Ensure that array lengths are as expected

        Raises
        ------
        AssertionError
            When an array length doesn't match that expected.
        """
        numCoeff = (self.order + 1)*(self.order + 2)//2
        assert len(self.xCoeff) == numCoeff
        assert len(self.yCoeff) == numCoeff
        assert len(self.rightCcdCoeff) == 6

    def __len__(self):
        """Number of fibers"""
        return len(self.fiberId)

    @classmethod
    def _readImpl(cls, fits, identity):
        """Implementation of reading from a FITS file in memory

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file in memory.
        identity : `pfs.datamodel.CalibIdentity`
            Identity of the calib data.

        Returns
        -------
        self : `DifferentialDetectorMap`
            DetectorMap read from FITS file.
        """
        header = astropyHeaderToDict(fits[0].header)
        box = Box.fromFitsHeader(header)
        order = header["ORDER"]

        base = SplinedDetectorMap._readImpl(fits, identity)

        xCoeff = fits["COEFFICIENTS"].data["x"].astype(float)
        yCoeff = fits["COEFFICIENTS"].data["y"].astype(float)
        rightCcd = fits["RIGHTCCD"].data["coefficients"].astype(float)

        return cls(identity, box, base, order, xCoeff, yCoeff, rightCcd, header)

    def _writeImpl(self):
        """Implementation of writing to FITS file

        Returns
        -------
        fits : `astropy.io.fits.HDUList`
            FITS file representation.
        """
        # NOTE: When making any changes to this method that modify the output
        # format, increment the DAMD_VER header value in the
        # SplinedDetectorMap._writeImpl method, and record the change in
        # the versions.txt file.
        fits = self.base._writeImpl()

        header = self.metadata.copy()
        header.update(self.box.toFitsHeader())
        if "pfs_detectorMap_class" in header:
            del header["pfs_detectorMap_class"]
        header["OBSTYPE"] = "detectorMap"
        header["HIERARCH pfs_detectorMap_class"] = "DistortedDetectorMap"
        header["ORDER"] = self.order
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=astropy.io.fits.verify.VerifyWarning)
            fits[0].header.update(astropyHeaderFromDict(header))

        tableHeader = astropy.io.fits.Header()
        tableHeader["INHERIT"] = True

        table = astropy.io.fits.BinTableHDU.from_columns([
            astropy.io.fits.Column(name="x", format="D", array=self.xCoeff),
            astropy.io.fits.Column(name="y", format="D", array=self.yCoeff),
        ], header=tableHeader, name="COEFFICIENTS")
        fits.append(table)

        table = astropy.io.fits.BinTableHDU.from_columns([
            astropy.io.fits.Column(name="coefficients", format="D", array=self.rightCcdCoeff),
        ], header=tableHeader, name="RIGHTCCD")
        fits.append(table)

        return fits


class DoubleDetectorMap(PfsDetectorMap):
    """DetectorMap implemented as two distortions on top of a SplinedDetectorMap

    This implementation handles I/O only. For a fully-functional implementation
    that includes evaluation of the mappings, see the drp_stella package.

    Parameters
    ----------
    identity : `pfs.datamodel.CalibIdentity`
        Identity of the data of interest.
    box : `Box`
        Bounding box for detector.
    base : `pfs.datamodel.SplinedDetectorMap`
        Base detectorMap.
    order : `int`
        Polynomial order.
    xLeft : `numpy.ndarray` of `float`, shape ``(M,)``
        Coefficients for x distortion polynomial of left CCD.
    yLeft : `numpy.ndarray` of `float`, shape ``(M,)``
        Coefficients for y distortion polynomial of left CCD.
    xRight : `numpy.ndarray` of `float`, shape ``(M,)``
        Coefficients for x distortion polynomial of right CCD.
    yRight : `numpy.ndarray` of `float`, shape ``(M,)``
        Coefficients for y distortion polynomial of right CCD.
    metadata : `dict`
        Keyword-value pairs to put in the header.
    """
    def __init__(self, identity, box, base, order, xLeft, yLeft, xRight, yRight, metadata):
        self.identity = identity
        self.box = box
        self.base = base
        self.order = order
        self.xLeft = xLeft
        self.yLeft = yLeft
        self.xRight = xRight
        self.yRight = yRight
        self.metadata = metadata
        self.validate()

    def validate(self):
        """Ensure that array lengths are as expected

        Raises
        ------
        AssertionError
            When an array length doesn't match that expected.
        """
        numCoeff = (self.order + 1)*(self.order + 2)//2
        assert len(self.xLeft) == numCoeff
        assert len(self.yLeft) == numCoeff
        assert len(self.xRight) == numCoeff
        assert len(self.yRight) == numCoeff

    def __len__(self):
        """Number of fibers"""
        return len(self.fiberId)

    @classmethod
    def _readImpl(cls, fits, identity):
        """Implementation of reading from a FITS file in memory

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file in memory.
        identity : `pfs.datamodel.CalibIdentity`
            Identity of the calib data.

        Returns
        -------
        self : `DifferentialDetectorMap`
            DetectorMap read from FITS file.
        """
        header = astropyHeaderToDict(fits[0].header)
        box = Box.fromFitsHeader(header)
        order = header["ORDER"]

        base = SplinedDetectorMap._readImpl(fits, identity)

        xLeft = fits["COEFFICIENTS"].data["xLeft"].astype(float)
        yLeft = fits["COEFFICIENTS"].data["yLeft"].astype(float)
        xRight = fits["COEFFICIENTS"].data["xRight"].astype(float)
        yRight = fits["COEFFICIENTS"].data["yRight"].astype(float)

        return cls(identity, box, base, order, xLeft, yLeft, xRight, yRight, header)

    def _writeImpl(self):
        """Implementation of writing to FITS file

        Returns
        -------
        fits : `astropy.io.fits.HDUList`
            FITS file representation.
        """
        # NOTE: When making any changes to this method that modify the output
        # format, increment the DAMD_VER header value in the
        # SplinedDetectorMap._writeImpl method, and record the change in
        # the versions.txt file.
        fits = self.base._writeImpl()

        header = self.metadata.copy()
        header.update(self.box.toFitsHeader())
        if "pfs_detectorMap_class" in header:
            del header["pfs_detectorMap_class"]
        header["OBSTYPE"] = "detectorMap"
        header["HIERARCH pfs_detectorMap_class"] = "DoubleDetectorMap"
        header["ORDER"] = self.order
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=astropy.io.fits.verify.VerifyWarning)
            fits[0].header.update(astropyHeaderFromDict(header))

        tableHeader = astropy.io.fits.Header()
        tableHeader["INHERIT"] = True

        table = astropy.io.fits.BinTableHDU.from_columns([
            astropy.io.fits.Column(name="xLeft", format="D", array=self.xLeft),
            astropy.io.fits.Column(name="yLeft", format="D", array=self.yLeft),
            astropy.io.fits.Column(name="xRight", format="D", array=self.xRight),
            astropy.io.fits.Column(name="yRight", format="D", array=self.yRight),
        ], header=tableHeader, name="COEFFICIENTS")
        fits.append(table)

        return fits


class PolynomialDetectorMap(PfsDetectorMap):
    """DetectorMap implemented as polynomial distortion over SplinedDetectorMap

    This implementation handles I/O only. For a fully-functional implementation
    that includes evaluation of the mappings, see the drp_stella package.

    Parameters
    ----------
    identity : `pfs.datamodel.CalibIdentity`
        Identity of the data of interest.
    box : `Box`
        Bounding box for detector.
    base : `pfs.datamodel.SplinedDetectorMap`
        Base detectorMap.
    order : `int`
        Polynomial order.
    xCoeff : `numpy.ndarray` of `float`, shape ``(M,)``
        Coefficients for x distortion polynomial.
    yCoeff : `numpy.ndarray` of `float`, shape ``(M,)``
        Coefficients for y distortion polynomial.
    metadata : `dict`
        Keyword-value pairs to put in the header.
    """
    def __init__(self, identity, box, base, order, xCoeff, yCoeff, metadata):
        self.identity = identity
        self.box = box
        self.base = base
        self.order = order
        self.xCoeff = xCoeff
        self.yCoeff = yCoeff
        self.metadata = metadata
        self.validate()

    def validate(self):
        """Ensure that array lengths are as expected

        Raises
        ------
        AssertionError
            When an array length doesn't match that expected.
        """
        numCoeff = (self.order + 1)*(self.order + 2)//2
        assert len(self.xCoeff) == numCoeff
        assert len(self.yCoeff) == numCoeff

    def __len__(self):
        """Number of fibers"""
        return len(self.fiberId)

    @classmethod
    def _readImpl(cls, fits, identity):
        """Implementation of reading from a FITS file in memory

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file in memory.
        identity : `pfs.datamodel.CalibIdentity`
            Identity of the calib data.

        Returns
        -------
        self : `DifferentialDetectorMap`
            DetectorMap read from FITS file.
        """
        header = astropyHeaderToDict(fits[0].header)
        box = Box.fromFitsHeader(header)
        order = header["ORDER"]

        base = SplinedDetectorMap._readImpl(fits, identity)

        xCoeff = fits["COEFFICIENTS"].data["xCoeff"].astype(float)
        yCoeff = fits["COEFFICIENTS"].data["yCoeff"].astype(float)

        return cls(identity, box, base, order, xCoeff, yCoeff, header)

    def _writeImpl(self):
        """Implementation of writing to FITS file

        Returns
        -------
        fits : `astropy.io.fits.HDUList`
            FITS file representation.
        """
        # NOTE: When making any changes to this method that modify the output
        # format, increment the DAMD_VER header value in the
        # SplinedDetectorMap._writeImpl method, and record the change in
        # the versions.txt file.
        fits = self.base._writeImpl()

        header = self.metadata.copy()
        header.update(self.box.toFitsHeader())
        if "pfs_detectorMap_class" in header:
            del header["pfs_detectorMap_class"]
        header["OBSTYPE"] = "detectorMap"
        header["HIERARCH pfs_detectorMap_class"] = "PolynomialDetectorMap"
        header["ORDER"] = self.order
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=astropy.io.fits.verify.VerifyWarning)
            fits[0].header.update(astropyHeaderFromDict(header))

        tableHeader = astropy.io.fits.Header()
        tableHeader["INHERIT"] = True

        table = astropy.io.fits.BinTableHDU.from_columns([
            astropy.io.fits.Column(name="xCoeff", format="D", array=self.xCoeff),
            astropy.io.fits.Column(name="yCoeff", format="D", array=self.yCoeff),
        ], header=tableHeader, name="COEFFICIENTS")
        fits.append(table)

        return fits


class PfsDistortion(ABC):
    """Base class for distortion models for the MultipleDistortionsDetectorMap

    The distortion implementations in the datamodel package handle I/O only.
    For fully-functional implementations that include evaluation of the
    mappings, see the drp_stella package.

    Subclasses must implement the ``readFits`` and ``writeFits`` methods.

    Multiple distortions may be written to a single FITS file, so a single
    extension name is not sufficient to identify them; the index is added to
    disambiguate and order them.
    """
    @classmethod
    def getName(cls) -> str:
        """Name of distortion

        Returns
        -------
        name : `str`
            Name of distortion.
        """
        return cls.__name__

    @classmethod
    def getDistortion(cls, name: str) -> Type["PfsDistortion"]:
        """Get distortion subclass by name

        Parameters
        ----------
        name : `str`
            Distortion subclass name.

        Returns
        -------
        subclass : `PfsDistortion`
            Distortion subclass.
        """
        for ss in cls.__subclasses__():
            if ss.getName() == name:
                return ss
        raise RuntimeError(f"Unrecognised distortion class: {name}")

    @classmethod
    @abstractmethod
    def readFits(cls, fits: astropy.io.fits.HDUList, index: int) -> "PfsDistortion":
        """Read from FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file in memory.
        index : `int`
            Index of distortion to read.

        Returns
        -------
        self : ``cls``
            Constructed distortion, from FITS file.
        """
        raise NotImplementedError("Subclass must implement readFits()")

    @abstractmethod
    def writeFits(self, fits: astropy.io.fits.HDUList, index: int):
        """Write to FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file in memory; modified.
        index : `int`
            Index of distortion to write.
        """
        raise NotImplementedError("Subclass must implement writeFits()")


class DoubleDistortion(PfsDistortion):
    """Distortion fields in x and y for left and right CCDs

    Parameters
    ----------
    order : `int`
        Order of the polynomials.
    box : `Box`
        Range of the polynomials.
    xLeft : `numpy.ndarray` of `float`
        Coefficients for the xLeft polynomial.
    yLeft : `numpy.ndarray` of `float`
        Coefficients for the yLeft polynomial.
    xRight : `numpy.ndarray` of `float`
        Coefficients for the xRight polynomial.
    yRight : `numpy.ndarray` of `float`
        Coefficients for the yRight polynomial.
    """
    def __init__(
        self,
        order: int,
        box: Box,
        xLeft: np.ndarray,
        yLeft: np.ndarray,
        xRight: np.ndarray,
        yRight: np.ndarray,
    ):
        self.order = order
        self.box = box
        self.xLeft = xLeft
        self.yLeft = yLeft
        self.xRight = xRight
        self.yRight = yRight

    @staticmethod
    def getExtName(index: int) -> str:
        """Return FITS extension name

        Parameters
        ----------
        index : `int`
            Index of distortion.
        """
        return f"DoubleDistortion_{index}"

    @classmethod
    def readFits(cls, fits: astropy.io.fits.HDUList, index: int) -> "PfsDistortion":
        """Read from FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file in memory.
        index : `int`
            Index of distortion to read.

        Returns
        -------
        self : ``cls``
            Constructed distortion, from FITS file.
        """
        hdu = fits[cls.getExtName(index)]
        order = hdu.header["ORDER"]
        box = Box.fromFitsHeader(hdu.header)
        xLeft = hdu.data["xLeft"].astype(float)
        yLeft = hdu.data["yLeft"].astype(float)
        xRight = hdu.data["xRight"].astype(float)
        yRight = hdu.data["yRight"].astype(float)
        return cls(order, box, xLeft, yLeft, xRight, yRight)

    def writeFits(self, fits: astropy.io.fits.HDUList, index: int):
        """Write to a FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file in memory; modified.
        index : `int`
            Index of distortion to write.
        """
        header = astropy.io.fits.Header()
        header["ORDER"] = self.order
        header["INHERIT"] = True
        header.update(self.box.toFitsHeader())

        table = astropy.io.fits.BinTableHDU.from_columns([
            astropy.io.fits.Column(name="xLeft", format="D", array=self.xLeft),
            astropy.io.fits.Column(name="yLeft", format="D", array=self.yLeft),
            astropy.io.fits.Column(name="xRight", format="D", array=self.xRight),
            astropy.io.fits.Column(name="yRight", format="D", array=self.yRight),
        ], header=header, name=self.getExtName(index))
        fits.append(table)


class MultipleDistortionsDetectorMap(PfsDetectorMap):
    """Multiple distortions on top of a SplinedDetectorMap

    This implementation handles I/O only. For a fully-functional implementation
    that includes evaluation of the mappings, see the drp_stella package.

    Parameters
    ----------
    identity : `pfs.datamodel.CalibIdentity`
        Identity of the data of interest.
    box : `Box`
        Bounding box for detector.
    base : `pfs.datamodel.SplinedDetectorMap`
        Base detectorMap.
    distortions : `list` of `pfs.datamodel.Distortion`
        Distortions to apply.
    metadata : `dict`
        Keyword-value pairs to put in the header.
    """
    def __init__(
        self,
        identity: CalibIdentity,
        box: Box,
        base: SplinedDetectorMap,
        distortions: List[PfsDistortion],
        metadata: Dict[str, Any] = None
    ):
        self.identity = identity
        self.box = box
        self.base = base
        self.distortions = distortions
        self.metadata = metadata
        self.validate()

    def validate(self):
        """Ensure that array lengths are as expected

        Raises
        ------
        AssertionError
            When an array length doesn't match that expected.
        """
        pass

    def __len__(self) -> int:
        """Number of fibers"""
        return len(self.base.fiberId)

    @classmethod
    def _readImpl(
        cls, fits: astropy.io.fits.HDUList, identity: CalibIdentity
    ) -> "MultipleDistortionsDetectorMap":
        """Implementation of reading from a FITS file in memory

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file in memory.
        identity : `pfs.datamodel.CalibIdentity`
            Identity of the calib data.

        Returns
        -------
        self : `DifferentialDetectorMap`
            DetectorMap read from FITS file.
        """
        header = astropyHeaderToDict(fits[0].header)
        box = Box.fromFitsHeader(header)
        base = SplinedDetectorMap._readImpl(fits, identity)

        names = ("".join(nn.tolist()) for nn in fits["DISTORTIONS"].data["name"])
        classes = (PfsDistortion.getDistortion(nn) for nn in names)
        distortions = [DistortionClass.readFits(fits, ii) for ii, DistortionClass in enumerate(classes)]

        return cls(identity, box, base, distortions, header)

    def _writeImpl(self):
        """Implementation of writing to FITS file

        Returns
        -------
        fits : `astropy.io.fits.HDUList`
            FITS file representation.
        """
        # NOTE: When making any changes to this method that modify the output
        # format, increment the DAMD_VER header value in the
        # SplinedDetectorMap._writeImpl method, and record the change in
        # the versions.txt file.
        fits = self.base._writeImpl()

        header = self.metadata.copy()
        header.update(self.box.toFitsHeader())
        if "pfs_detectorMap_class" in header:
            del header["pfs_detectorMap_class"]
        header["OBSTYPE"] = "detectorMap"
        header["HIERARCH pfs_detectorMap_class"] = "MultipleDistortionsDetectorMap"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=astropy.io.fits.verify.VerifyWarning)
            fits[0].header.update(astropyHeaderFromDict(header))

        tableHeader = astropy.io.fits.Header()
        tableHeader["INHERIT"] = True
        names = [distortion.getName() for distortion in self.distortions]
        table = astropy.io.fits.BinTableHDU.from_columns([
            astropy.io.fits.Column("name", format="PA()", array=names),
        ], header=tableHeader, name="DISTORTIONS")
        fits.append(table)
        for ii, distortion in enumerate(self.distortions):
            distortion.writeFits(fits, ii)

        return fits
