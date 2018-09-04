import numpy as np
import os

try:
    import astropy.io.fits as pyfits
except ImportError:
    try:
        import pyfits
    except ImportError:
        pyfits = None

from .utils import calculate_pfiDesignId
from .pfiDesign import PfiDesign

class pfiConfig(object):
    """ A class corresponding to a single pfiConfig file. """

    fileNameFormat = "pfiConfig-%06d-0x%08x.fits"

    def __init__(self, pfiDesignId=None, visit0=None,
                 mcsX=None, mcsY=None, pfiX=None, pfiY=None,
                 fpsTransformId=None, fpsFlags=None):
        self.pfiDesignId = pfiDesignId
        self.visit0 = visit0
        self.mcsX = mcsX
        self.mcsY = mcsY
        self.pfiX = pfiX
        self.pfiY = pfiY
        self.fpsTransformId = fpsTransformId
        self.fpsFlags = fpsFlags

        self.pfiDesign = None

    def readPfiDesign(self, dirName='.', fileName=None):
        fileName = self.fileNameFormat % (self.visit0, self.pfiConfigId)

        # Read in our design.
        self.pfiDesign = PfiDesign(self.pfiDesignId)
        self.pfiDesign.read(dirName=dirName, fileName=fileName)

    def read(self, dirName="."):
        """ Read our persisted file from dirName. """
        if not pyfits:
            raise RuntimeError("I failed to import pyfits, so cannot read from disk")

        if self.visit0 is None or self.pfiDesignId is None:
            raise RuntimeError("Need both visit0 and pfiDesignId to read a pfiConfig file.")
        fileName = self.fileNameFormat % (self.visit0, self.pfiConfigId)

        # Read in our design.
        self.pfiDesign = PfiDesign(self.pfiDesignId)
        self.pfiDesign.read(dirName=dirName, fileName=fileName)

        fd = pyfits.open(os.path.join(dirName, fileName))

        hdu = fd["CONFIG"]
        hdr, data = hdu.header, hdu.data

        self.mcsX = data['mcsX']
        self.mcsY = data['mcsY']
        self.pfiX = data['pfiX']
        self.pfiY = data['pfiY']
        self.fpsFlags = data['fpsFlags']

        assert self.pfiDesignId == calculate_pfiDesignId(self.fiberId, self.ra, self.dec)
