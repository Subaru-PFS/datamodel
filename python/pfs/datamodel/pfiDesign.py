import numpy as np
import os

try:
    import astropy.io.fits as pyfits
except ImportError:
    try:
        import pyfits
    except ImportError:
        pyfits = None

from pfs.datamodel.utils import calculate_pfiDesignId

class PfiDesign(object):
    """A class corresponding to a single pfiDesign file"""

    fileNameFormat = "pfiDesign-0x%08x.fits"

    def __init__(self, pfiDesignId=None, tract=None, patch=None,
                 fiberId=None, ra=None, dec=None, catId=None, objId=None,
                 fiberMag=None, filterNames=["g", "r", "i", "z", "y"]):
        self.pfiDesignId = pfiDesignId
        self.tract = tract
        self.patch = patch

        self.fiberId = fiberId
        self.ra = ra
        self.dec = dec
        self.catId = catId
        self.objId = objId
        self.fiberMag = fiberMag
        self.filterNames = filterNames

        _pfiDesignId = calculate_pfiDesignId(self.fiberId, self.ra, self.dec)

        if self.pfiDesignId is None:
            self.pfiDesignId = _pfiDesignId
        elif _pfiDesignId != 0x0:
            if self.pfiDesignId != _pfiDesignId:
                raise RuntimeError("Mismatch between pfiDesignId == 0x%08x and fiberId/ra/dec -> 0x%08x" %
                                   (self.pfiDesignId, _pfiDesignId))

    def read(self, dirName=".", fileName=None):
        """Read self's pfiDesign file from directory dirName

        Args
        ----
        dirName : str
          the directory to search in
        fileName : str
          if set, the (non-path) filename to read. This is used
          when reading a pfiConfig file.
        """

        if not pyfits:
            raise RuntimeError("I failed to import pyfits, so cannot read from disk")

        if fileName is None:
            fileName = self.fileNameFormat % self.pfiDesignId

        fd = pyfits.open(os.path.join(dirName, fileName))
        hdr, data = fd["DESIGN"].header, fd["DESIGN"].data

        self.filterNames = []
        i = -1
        while True:
            i += 1
            key = "FILTER%d" % i
            if key in hdr:
                self.filterNames.append(hdr[key])
            else:
                break

        if False:
            for k, v in hdr.items():
                print("%8s %s" % (k, v))

        self.fiberId = data['fiberId']
        self.tract = data['tract']
        self.patch = data['patch']
        self.objId = data['objId']
        self.ra = data['ra']
        self.dec = data['dec']
        self.fiberMag = data['fiberMag']

        assert self.pfiDesignId == calculate_pfiDesignId(self.fiberId, self.ra, self.dec)

    def write(self, dirName=".", fileName=None):
        if not pyfits:
            raise RuntimeError("I failed to import pyfits, so cannot read from disk")

        for name in ["fiberId", "ra", "dec"]:
            if getattr(self, name, None) is None:
                if name == "fiberId" or self.pfiDesignId != 0x0:
                    raise RuntimeError("I cannot write a pfiDesign file unless %s is provided" % name)

                setattr(self, name, np.zeros_like(self.fiberId, dtype=np.float32))

        # even if set in __init__ it might be invalid by now
        _pfiDesignId = calculate_pfiDesignId(self.fiberId, self.ra, self.dec)

        if self.pfiDesignId is None:
            self.pfiDesignId = _pfiDesignId
        else:
            if self.pfiDesignId != _pfiDesignId:
                raise RuntimeError("Mismatch between pfiDesignId == 0x%08x and fiberId/ra/dec -> 0x%08x" %
                                   (self.pfiDesignId, _pfiDesignId))

        hdus = pyfits.HDUList()

        hdr = pyfits.Header()
        hdu = pyfits.PrimaryHDU(header=hdr)
        hdr.update()
        hdus.append(hdu)

        # catId, objId, ra, dec, fiber flux
        hdr = pyfits.Header()
        for i, b in enumerate(self.filterNames):
            hdr["FILTER%d" % i] = b
        hdr.update(INHERIT=True)

        hdu = pyfits.BinTableHDU.from_columns([
            pyfits.Column(name = 'fiberId', format = 'J', array=self.fiberId),
            pyfits.Column(name = 'catId', format = 'J', array=self.catId),
            pyfits.Column(name = 'tract', format = 'J', array=self.tract),
            pyfits.Column(name = 'patch', format = 'A3', array=self.patch),
            pyfits.Column(name = 'objId', format = 'K', array=self.objId),
            pyfits.Column(name = 'ra', format = 'E', array=self.ra),
            pyfits.Column(name = 'dec', format = 'E', array=self.dec),
            pyfits.Column(name = 'fiberMag', format = '%dE' % len(self.filterNames), array=self.fiberMag),
        ], hdr)
        hdu.name = 'DESIGN'
        hdus.append(hdu)

        # clobber=True in writeto prints a message, so use open instead
        if fileName is None:
            fileName = self.fileNameFormat % (self.pfiDesignId)
        with open(os.path.join(dirName, fileName), "w") as fd:
            hdus.writeto(fd)
