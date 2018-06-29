import os

import numpy as np
try:
    import astropy.io.fits as pyfits
except ImportError:
    pyfits = None

class PfsSimObject(object):
    """A class corresponding to a single pfsSimObject file"""

    fileNameFormat = "pfsSimObject-%03d-0x%08x.fits"

    def __init__(self, objId, catId=0,
                 lam=None, flux=None, lines=None):
        self.catId = catId
        self.objId = objId

        self.lam = lam
        self.flux = flux
        self.lines = lines

    @property
    def fileName(self):
        return self.fileNameFormat % (self.catId, self.objId)

    def read(self, dirName="."):
        """Read self's pfsSimObject file from directory dirName"""

        if not pyfits:
            raise RuntimeError("I failed to import pyfits, so cannot read from disk")

        fd = pyfits.open(os.path.join(dirName, self.fileName))

        phdr = fd[0].header
        for name, value in [("catId", self.catId),
                            ("objId", self.objId),
                            ]:
            if value != phdr[name]:
                raise RuntimeError("Header keyword %s is %s; expected %s" % (name, phdr[name], value))

        for hduName in ["LAM", "FLUX"]:
            hdu = fd[hduName]
            hdr, data = hdu.header, hdu.data

            if hduName == "FLUX":
                self.flux = data
            elif hduName == "LAM":
                self.lam = data
            else:
                raise RuntimeError("Unexpected HDU %s reading %s" % (hduName, self.fileName))

    def write(self, dirName=".", fileName=None):
        if not pyfits:
            raise RuntimeError("I failed to import pyfits, so cannot write to disk")

        if self.lam is None or self.flux is None:
            raise RuntimeError("cannot write pfsSimObject file without lam or flux")

        if fileName is None:
            fileName = self.fileName

        hdus = pyfits.HDUList()

        hdr = pyfits.Header()
        hdr.update(catId=self.catId,
                   objId=self.objId,
        )
        hdus.append(pyfits.PrimaryHDU(header=hdr))

        for hduName, data in [("LAM", self.lam),
                              ("FLUX", self.flux),
                              ("LINES", self.lines),
                              ]:
            hdr = pyfits.Header()
            hdr.update(INHERIT=True)

            if hduName == "LINES":
                if data is None:
                    continue
                raise NotImplementedError("Have not worked on line table yet")
            else:
                hdu = pyfits.ImageHDU(data, hdr)

            hdu.name = hduName
            hdus.append(hdu)

        # clobber=True in writeto prints a message, so use open instead
        with open(os.path.join(dirName, fileName), "w") as fd:
            hdus.writeto(fd)

