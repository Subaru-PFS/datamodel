import os

import numpy as np
try:
    import astropy.io.fits as pyfits
except ImportError:
    pyfits = None

import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.daf.base as dafBase
import lsst.afw.fits as afwFits


class PfsFiberTrace:
    """A class corresponding to a single fiberTrace file"""
    fileNameFormat = "pfsFiberTrace-%10s-%06d-%1s%1d.fits"

    def __init__(self, obsDate, spectrograph, arm, visit0, metadata=None):
        self.obsDate = obsDate
        self.spectrograph = spectrograph
        self.arm = arm
        self.visit0 = visit0
        self.metadata = metadata

        self.fiberId = []
        self.traces = []

    def read(self, dirName="."):
        """Read self's pfsFiberTrace file from directory dirName"""
        if not pyfits:
            raise RuntimeError("I failed to import astropy.io.fits, so cannot read from disk")

        fileName = PfsFiberTrace.fileNameFormat % (self.obsDate, self.visit0, self.arm, self.spectrograph)

        self.metadata = afwFits.readMetadata(os.path.join(dirName, fileName), 0, True)
        self.metadata.remove("COMMENT")  # Added by FITS writer, not stripped (!)
        allTracesMI = afwImage.MaskedImageF(os.path.join(dirName, fileName))

        with pyfits.open(os.path.join(dirName, fileName)) as fd:
            hdu = fd["ID_BOX"]
            self.fiberId = hdu.data['FIBERID']
            minX = hdu.data['MINX']
            minY = hdu.data['MINY']
            maxX = hdu.data['MAXX']
            maxY = hdu.data['MAXY']

        self.traces = []
        x0 = 0
        for i in range(len(self.fiberId)):
            # bbox: BBox in full (i.e. data) image
            bbox = afwGeom.BoxI(afwGeom.PointI(minX[i], minY[i]), afwGeom.PointI(maxX[i], maxY[i]))

            # bboxAllTMI: BBox in allTracesMI
            bboxAllTMI = afwGeom.BoxI(afwGeom.PointI(x0, bbox.getMinY()), bbox.getDimensions())
            
            trace = allTracesMI[bboxAllTMI].clone()
            trace.setXY0(bbox.getBegin())
            
            self.traces.append(trace)
            x0 += bbox.getWidth()

    def write(self, dirName=".", fileName=None, metadata=None):
        if not pyfits:
            raise RuntimeError("I failed to import astropy.io.fits, so cannot write to disk")

        if fileName is None:
            fileName = self.fileNameFormat % (self.obsDate, self.visit0, self.arm, self.spectrograph)
        fullFileName = os.path.join(dirName, fileName)
        #
        # We'll pack all the traces into a single masked image, so figure out how large it needs to be
        #
        # Start by unpacking the traces' BBoxes; we need to do this anyway for the fits I/O
        #
        minX = []
        minY = []
        maxX = []
        maxY = []
        width = 0
        for i in range(len(self.traces)):
            bbox = self.traces[i].getBBox()

            minX.append(bbox.getMinX())
            minY.append(bbox.getMinY())
            maxX.append(bbox.getMaxX())
            maxY.append(bbox.getMaxY())

            width += bbox.getWidth()

        height = max(maxY) + 1
        allTracesMI = afwImage.MaskedImageF(width, height)

        # Copy trace's MaskedImages to allTracesMI
        x0 = 0
        origin = afwGeom.PointI(0, 0)
        for i in range(len(self.traces)):
            trace = self.traces[i]

            xy0 = afwGeom.Point2I(x0, minY[i]) # origin in allTracesMI
            allTracesMI[afwGeom.BoxI(xy0, trace.getDimensions())] = \
                        trace.Factory(trace, afwGeom.BoxI(origin, trace.getDimensions()), afwImage.LOCAL)

            x0 += trace.getWidth()
        #
        # Time to actually write the data
        #
        if metadata is None:
            hdr = dafBase.PropertySet()
        else:
            hdr = metadata

        hdr.set('OBSTYPE', 'FIBERTRACE')

        # Write fits file from MaskedImage
        allTracesMI.writeFits(fullFileName, hdr)

        # append the additional HDUs
        hdu = pyfits.BinTableHDU.from_columns([
            pyfits.Column(name = 'FIBERID', format = 'J', array=np.array(self.fiberId, dtype=np.int32)),
            pyfits.Column(name = 'MINX', format = 'J', array=np.array(minX, dtype=np.int32)),
            pyfits.Column(name = 'MINY', format = 'J', array=np.array(minY, dtype=np.int32)),
            pyfits.Column(name = 'MAXX', format = 'J', array=np.array(maxX, dtype=np.int32)),
            pyfits.Column(name = 'MAXY', format = 'J', array=np.array(maxY, dtype=np.int32)),
        ])

        hdu.name = "ID_BOX"
        hdu.header["INHERIT"] = True

        # clobber=True in writeto prints a message, so use open instead
        
        with pyfits.open(fullFileName, "update") as fd:
            fd[1].name = "IMAGE"
            fd[2].name = "MASK"
            fd[3].name = "VARIANCE"
            fd.append(hdu)
