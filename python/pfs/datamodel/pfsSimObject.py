import os

import logging
import numpy as np
try:
    import astropy.io.fits as pyfits
except ImportError:
    pyfits = None

class PfsSimObject(object):
    """A class corresponding to a single pfsSimObject file"""

    fileNameFormat = "pfsSimObject-%03d-0x%08x.fits"

    def __init__(self, objId, catId=0):
        """ Create an _empty_ pfsSimObject. Must call .read() or .setData() """

        self.catId = catId
        self.objId = objId

        self._lam = None
        self._flux = None
        self.lines = None
        self.wcs = None

    def __str__(self):
        return f"pfsSimObject(catId={self.catId}, objId=0x{self.objId:08x}) loaded={self.lam is not None}"

    @property
    def fileName(self):
        return self.fileNameFormat % (self.catId, self.objId)

    # Vaguely signify that we do not want .lam and .flux written to directly.
    # I guess we could copy and .setflags(write=False)
    @property
    def lam(self):
        return self._lam

    @property
    def flux(self):
        return self._flux

    def setData(self, flux, lam=None, logSpacing=False, lamRange=None, lines=None):
        """ Set all data

        Args
        ----

        flux : 1d ndarray
          The flux, in nanoJanskys. Converted to float32 if necessary.
        lam : 1d ndarray, same length as flux
          The wavelengths of flux, in nanometers. Converted to float32 if necessary.
          Interpretation depends on logSpacing
          If None, then created from lamRange and logSpacing
        logSpacing : bool
          If True, lam are logarithmically _spaced_.
        lamRange : (nm of flux[0], nm of flux[-1])
          If set, the full range of lam. This can be used in two ways:
           - if lam is None, lam will be filled using lamRange, len(flux), logSpacing
           - if lam is set, lam is checked that it conforms to CRVAL1 and CDELT1,
             with CUNIT1 being either 'WAVE' or 'WAVE-LOG' depending on logSpacing.

        Exceptions
        ----------
        ValueError if lam and logRange are both passed in, but do not agree.
        """

        flux = np.asarray(flux, dtype='f4')
        if lam is not None:
            lam = np.asarray(lam, dtype='f4')

        if lam is None and lamRange is None:
            raise RuntimeError("either lam or lamRange must be passed in.")

        if lamRange is None:
            self.wcs = None
        else:
            lamStart, lamEnd = lamRange

            if logSpacing:
                lam1ll = np.logspace(np.log10(lamStart), np.log10(lamEnd),
                                     num=len(flux), endpoint=True, dtype='f4')
                lam1 = np.log10(lam1ll)
                self.wcs = ('WAVE-LOG',
                            np.log10(lamStart)*np.log(10),
                            np.diff(np.log10(lam1ll[:2]))[0] * np.log(10))
            else:
                lam1 = np.linspace(lamStart, lamEnd, num=len(flux),
                                   endpoint=True, dtype='f4')
                self.wcs = ('WAVE',
                            lamStart,
                            (lamEnd-lamStart)/(len(flux)-1))

            if lam is not None:
                ok = np.allclose(lam, lam1)
                if not ok:
                    raise ValueError("lam array and lamRange do not match")
            else:
                lam1.setflags(write=False)
                lam = lam1

        self._lam = lam
        self._flux = flux

        self.lines = lines      # CPLXXX check this

    def read(self, dirName="."):
        """Read self's pfsSimObject file from directory dirName"""

        if not pyfits:
            raise RuntimeError("I failed to import pyfits, so cannot read from disk")

        if self._lam is not None:
            raise RuntimeError(f"{self} has already been initialized'")

        path = os.path.join(dirName, self.fileName)
        fd = pyfits.open(path)

        phdr = fd[0].header
        for name, value in [("catId", self.catId),
                            ("objId", self.objId),
                            ]:
            if value != phdr[name]:
                raise RuntimeError("Header keyword %s is %s; expected %s" % (name, phdr[name], value))

        for hduName in ["FLUX", "LAM"]:
            hdu = fd[hduName]
            hdr, data = hdu.header, hdu.data

            if hduName == "FLUX":
                self._flux = data
                if 'CTYPE1' in hdr:
                    try:
                        self.wcs = hdr['CTYPE1'], hdr['CD1_1'], hdr['CDELT1']
                    except KeyError:
                        logging.warn(f'ignored incomplete WCS in f{path}')

            elif hduName == "LAM":
                self._lam = data
            else:
                raise RuntimeError("Unexpected HDU %s reading %s" % (hduName, self.fileName))

    def _addWcs(self, hdr):
        """ Add WCS cards if we know enough.  """

        if self.wcs is not None:
            comment = 'linear' if self.wcs[0] == 'WAVE' else 'log base e -- see CTYPE1 reference'
            hdr.append(('CTYPE1', self.wcs[0], 'See Griesen et al. DOI: 10.1051/0004-6361:20053818'))
            hdr.append(('CUNIT1', 'nm'))
            hdr.append(('CRVAL1', self.wcs[1], comment))
            hdr.append(('CRPIX1', 1))
            hdr.append(('CDELT1', self.wcs[2], comment))

    def write(self, dirName=".", fileName=None):
        if not pyfits:
            raise RuntimeError("I failed to import pyfits, so cannot write to disk")

        if self._lam is None or self._flux is None:
            raise RuntimeError("cannot write pfsSimObject file without both wavelengths and flux")

        if fileName is None:
            fileName = self.fileName

        hdus = pyfits.HDUList()

        hdr = pyfits.Header()
        hdr['catId'] = self.catId
        hdr['objId'] = self.objId

        self._addWcs(hdr)

        hdus.append(pyfits.PrimaryHDU(header=hdr))

        for hduName, data in [("LAM", self._lam),
                              ("FLUX", self._flux),
                              ("LINES", self.lines),
                              ]:
            hdr = pyfits.Header()
            hdr['INHERIT'] = True  # Cannot depend on this, sadly.
            self._addWcs(hdr)

            if hduName == "LINES":
                if data is None:
                    continue
                raise NotImplementedError("Have not worked on line table yet")
            else:
                hdu = pyfits.ImageHDU(data, hdr)

            hdu.name = hduName
            hdus.append(hdu)

        # clobber=True in writeto prints a message, so use open instead
        with open(os.path.join(dirName, fileName), "wb") as fd:
            hdus.writeto(fd)
