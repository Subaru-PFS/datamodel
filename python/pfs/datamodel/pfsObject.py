import collections
import os

import numpy as np
try:
    import pyfits
except ImportError:
    pyfits = None
from scipy.interpolate import interp1d

from pfs.datamodel.pfsArm import PfsArm
from pfs.datamodel.utils import calculate_pfsVisitHash


class PfsObject:
    """A class corresponding to a single pfsObject file"""
    NCOARSE = 10    # number of elements in coarse-grained covariance
    fileNameFormat = "pfsObject-%05d-%s-%03d-%08x-%02d-0x%08x.fits"

    class FluxTbl:
        def __init__(self, lam=None):
            if lam is None:
                self.lam = None
                self.flux = None
                self.fluxVariance = None
                self.mask = None
            else:
                self.lam = np.array(lam, dtype=np.float32)
                self.flux = np.zeros_like(lam)
                self.fluxVariance = np.zeros_like(lam)
                self.mask = np.zeros_like(lam, dtype=np.int32)

    def __init__(self, tract, patch, objId, catId=0, visits=None, pfsConfigIds=None,
                 nVisit=None, pfsVisitHash=0x0):
        if visits is not None:
            self.visits = visits
            self.nVisit = len(visits)
            self.pfsVisitHash = calculate_pfsVisitHash(visits)

            if nVisit and nVisit != self.nVisit:
                raise RuntimeError("Number of visits provided (== %d) != nVisit (== %d)" %
                                   (nVisit, self.nVisit))
            if pfsVisitHash and pfsVisitHash != self.pfsVisitHash:
                raise RuntimeError("pfsVisitHash provided (== 0x%08x) != nVisit (== 0x%08x)" %
                                   (pfsVisitHash, self.pfsVisitHash))
        else:
            self.nVisit = nVisit if nVisit else 1
            self.visits = None
            self.pfsVisitHash = pfsVisitHash

        self.pfsConfigIds = pfsConfigIds

        self.tract = tract
        self.patch = patch
        self.catId = catId
        self.objId = objId

        self.lam = None
        self.flux = None
        self.fluxTbl = self.FluxTbl(None)
        self.mask = None
        self.sky = None

        self.covar = None
        self.covar2 = None

    def read(self, dirName="."):
        """Read self's pfsObject file from directory dirName"""

        if not pyfits:
            raise RuntimeError("I failed to import pyfits, so cannot read from disk")

        fileName = self.fileNameFormat % (self.tract, self.patch, self.catId, self.objId,
                                          self.nVisit % 100, self.pfsVisitHash)

        fd = pyfits.open(os.path.join(dirName, fileName))

        phdr = fd[0].header
        for name, value in [("tract", self.tract),
                            ("patch", self.patch),
                            ("catId", self.catId),
                            ("objId", self.objId),
                            ("pfsVHash", self.pfsVisitHash)
                            ]:
            if value != phdr[name]:
                raise RuntimeError("Header keyword %s is %s; expected %s" % (name, phdr[name], value))

        for hduName in ["FLUX", "FLUXTBL", "COVAR", "COVAR2", "MASK", "SKY"]:
            hdu = fd[hduName]
            hdr, data = hdu.header, hdu.data

            if hdu.data is None:
                continue

            if self.lam is None:
                self.lam = 1 + np.arange(data.shape[-1], dtype=float)
                self.lam = hdr["CRVAL1"] + (self.lam - hdr["CRPIX1"])*hdr["CD1_1"]

            if False:
                for k, v in hdr.items():
                    print("%8s %s" % (k, v))

            if data.ndim == 1:
                if hduName == "FLUX":
                    self.flux = data
                elif hduName == "FLUXTBL":
                    self.fluxTbl.lam = data["lambda"]
                    self.fluxTbl.flux = data["flux"]
                    self.fluxTbl.fluxVariance = data["fluxVariance"]
                    self.fluxTbl.mask = data["mask"]
                elif hduName == "MASK":
                    self.mask = data
                elif hduName == "SKY":
                    self.sky = data
                else:
                    raise RuntimeError("Unexpected HDU %s reading %s" % (hduName, fileName))
            else:
                if hduName not in ("COVAR", "COVAR2"):
                    raise RuntimeError("Unexpected HDU %s reading %s" % (hduName, fileName))

                if hduName == "COVAR":
                    self.covar = data
                else:
                    self.covar2 = data

        hdu = fd["CONFIG"]
        hdr, data = hdu.header, hdu.data

        self.visits = data["visit"]
        self.pfsConfigIds = data["pfsConfigId"]

    def write(self, dirName=".", fileName=None):
        if not pyfits:
            raise RuntimeError("I failed to import pyfits, so cannot read from disk")

        hdus = pyfits.HDUList()

        hdr = pyfits.Header()
        hdr.update(tract=self.tract,
                   patch=self.patch,
                   catId=self.catId,
                   objId=self.objId,
                   pfsVHash=self.pfsVisitHash,
                   )
        hdus.append(pyfits.PrimaryHDU(header=hdr))

        for hduName, data in [("FLUX", self.flux),
                              ("FLUXTBL", self.fluxTbl),
                              ("COVAR", self.covar),
                              ("COVAR2", self.covar2),
                              ("MASK", self.mask),
                              ("SKY", self.sky),
                              ]:
            hdr = pyfits.Header()
            hdr.update(INHERIT=True)

            if hduName == "FLUX":     # Add WCS
                hdr.update(CRVAL1=self.lam[0], CRPIX1=0, CD1_1=(self.lam[1] - self.lam[0]))

            if not hdus:
                hdu = pyfits.PrimaryHDU(data, header=hdr)
            elif hduName == "FLUXTBL":
                hdu = pyfits.BinTableHDU.from_columns([
                        pyfits.Column('lambda',       'E', array=data.lam),
                        pyfits.Column('flux',         'E', array=data.flux),
                        pyfits.Column('fluxVariance', 'E', array=data.fluxVariance),
                        pyfits.Column('mask',         'J', array=data.mask),
                                                  ])
            else:
                hdu = pyfits.ImageHDU(data, hdr)

            hdu.name = hduName
            hdus.append(hdu)
        #
        # Now the config table
        #
        hdu = pyfits.BinTableHDU.from_columns([pyfits.Column('visit', 'J',
                                                             array=self.visits),
                                               pyfits.Column('pfsConfigId', 'K',
                                                             array=self.pfsConfigIds),
                                               ], nrows=self.nVisit)
        hdr.update(INHERIT=True)
        hdu.name = "CONFIG"
        hdus.append(hdu)

        if fileName is None:
            fileName = self.fileNameFormat % (self.tract, self.patch, self.catId, self.objId,
                                              self.nVisit % 100, self.pfsVisitHash)

        # clobber=True in writeto prints a message, so use open instead
        with open(os.path.join(dirName, fileName), "w") as fd:
            hdus.writeto(fd)

    def plot(self, showFlux=None, showFluxTbl=False, showMask=False, showSky=False,
             showCovar=False, showCovar2=False, showFluxVariance=False, showPlot=True):
        """Plot some or all of the contents of the PfsObject

        Default is to show the flux from the resampled arrays.

        If showFluxTbl is true, take the values from the non-resampled (fluxTbl) arrays
        """
        import matplotlib.pyplot as plt
        xlabel = "Wavelength (micron)"
        title = "%d %s 0x%08x %08d" % (self.tract, self.patch, self.pfsVisitHash, self.objId)

        show = dict(mask=showMask, sky=showSky,
                    covar=showCovar, covar2=showCovar2, fluxVariance=showFluxVariance)
        show.update(flux=not sum(show.values()) if showFlux is None else showFlux)
        show.update(fluxTbl=showFluxTbl)

        if not show["fluxTbl"]:
            for name, data in (["flux", self.flux],
                               ["mask", self.mask],
                               ["sky",  self.sky]):
                if not show[name]:
                    continue

                plt.plot(self.lam, data)
                plt.xlabel(xlabel)
                plt.title("%s %s" % (title, name))

                if name in ("flux"):
                    plt.axhline(0, ls=':', color='black')

                    if showPlot:
                        plt.show()

            if show["covar"]:
                for i in range(self.covar.shape[0]):
                    plt.plot(self.lam, self.covar[i], label="covar[%d]" % (i))
                    plt.legend(loc='best')

                plt.xlabel(xlabel)
                plt.title("%s %s" % (title, "covar"))
                if showPlot:
                    plt.show()

            if show["covar2"]:
                sc = plt.imshow(self.covar2, interpolation='nearest', vmin=0)
                plt.colorbar(sc)

                lab = "wavelength bin"
                plt.xlabel(lab)
                plt.ylabel(lab)
                plt.title("%s %s" % (title, "covar2"))
                if showPlot:
                    plt.show()
        else:
            for name, data in (["flux", self.fluxTbl.flux],
                               ["fluxVariance", self.fluxTbl.fluxVariance],
                               ["mask",  self.fluxTbl.mask]):
                if not show[name]:
                    continue

                plt.plot(self.fluxTbl.lam, data, alpha=0.3 if name == 'flux' else 1.0,
                         label="fluxTbl.%s" % name)

                if name in ("flux"):
                    plt.plot(self.lam, self.flux, label="flux")
                    plt.legend(loc='best')

                plt.xlabel(xlabel)
                plt.title("%s fluxTbl.%s" % (title, name))
                if showPlot:
                    plt.show()

