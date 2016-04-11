import collections
import os
import pyfits
import matplotlib.pyplot as plt

from pfs.datamodel.pfsConfig import PfsConfig

class PfsArm(object):
    """A class corresponding to a single pfsArm file"""
    #
    # Flags for MASKs
    #
    flags = dict(NODATA = 0x1,          # this pixel contains no data
                 )

    fileNameFormat = "pfsArm-%06d-%1d%1s.fits"

    def __init__(self, visit, spectrograph, arm):
        self.pfsConfigId = None
        self.visit = visit
        self.spectrograph = spectrograph
        self.arm = arm
        
    def read(self, dirName=".", pfsConfigs=None):
        """Read self's pfsArm file from directory dirName

        If provided, pfsConfigs is a dict of pfsConfig objects, indexed by pfsConfigId
        """
        fileName = PfsArm.fileNameFormat % (self.visit, self.spectrograph, self.arm)
        fd = pyfits.open(os.path.join(dirName, fileName)) 
            
        for hduName in ["WAVELENGTH", "FLUX", "COVAR", "MASK", "SKY"]:
            hdu = fd[hduName]
            hdr, data = hdu.header, hdu.data
        
            if False:
                for k, v in hdr.items():
                    print "%8s %s" % (k, v)
            
            if data.ndim == 2:
                if hduName == "WAVELENGTH":
                    self.lam = data
                elif hduName == "FLUX":
                    self.flux = data
                elif hduName == "MASK":
                    self.mask = data
                elif hduName == "SKY":
                    self.sky = data
                else:
                    raise RuntimeError("Unexpected HDU %s reading %s" % (hduName, fileName))
            else:
                if hduName != "COVAR":
                    raise RuntimeError("Unexpected HDU %s reading %s" % (hduName, fileName))

                self.covar = data
                
            #print hdr["EXTNAME"], hdr["XTENSION"], data.dtype, data.shape
            
        hdu = fd["CONFIG"]
        hdr, data = hdu.header, hdu.data
        
        assert 'pfsConfigId' in data.names
        assert 'visit' in data.names
        assert len(data['visit']) == 1   # only one row in the table
        
        if data['visit'][0] != self.visit:
            raise RuntimeError("Filename corresponds to visit %d, but config gives %d" %
                               self.visit, data['visit'][0])
            
        self.pfsConfigId = data['pfsConfigId'][0]
        
        if pfsConfigs is None:
            self.pfsConfig = None
        else:
            if self.pfsConfigId not in pfsConfigs:
                pfsConfigs[self.pfsConfigId] = PfsConfig(self.pfsConfigId)
                pfsConfigs[self.pfsConfigId].read(dirName)
            
            self.pfsConfig = pfsConfigs[self.pfsConfigId]
            
            if len(self.flux) != len(self.pfsConfig.ra):
                raise RuntimeError("Mismatch between pfsArm and pfsConfig files")
            if False:
                print "%d%s 0x%x %d" % \
                   (self.spectrograph, self.arm, self.pfsConfigId, self.visit),  \
                    pfsConfig.ra, pfsConfig.dec
        
    def getFiberIdx(self, fiberId):
        """Convert a fiberId to a fiber index (checking the range)"""
        if fiberId <= 0 or fiberId > len(self.lam):
            raise IndexError("fiberId %d is out of range %d..%d" % (fiberId, 1, len(self.lam)))

        return fiberId - 1

    def plot(self, fiberId=1, showFlux=None, showMask=False, showSky=False, showCovar=False):
        """Plot some or all of the contents of the PfsArm

        Default is to show the flux
        """
        show = dict(mask=showMask, sky=showSky, covar=showCovar)
        show.update(flux = not sum(show.values()) if showFlux is None else showFlux)

        fiberIdx = self.getFiberIdx(fiberId)
        
        xlabel = "Wavelength (micron)"
        title = "%06d %d%s fiber %d" % (self.visit, self.spectrograph, self.arm, fiberId)
        
        for name, data in (["flux", self.flux],
                           ["mask", self.mask],
                           ["sky", self.sky]):
            if not show[name]:
                continue

            plt.plot(self.lam[fiberIdx], data[fiberIdx])
            plt.xlabel(xlabel)
            plt.title("%s %s" % (title, name))

            if name in ("flux"):
                plt.axhline(0, ls=':', color='black')

            plt.show()

        if show["covar"]:
            for i in range(self.covar.shape[1]):
                plt.plot(self.lam[fiberIdx], self.covar[fiberIdx][i], label="covar[%d]" % (i))
            plt.legend(loc='best')

            plt.title("%s %s" % (title, "covar"))
            plt.show()

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class PfsArmSet(object):
    """Manipulate a set of pfsArms corresponding to a single visit"""
    def __init__(self, visits, spectrograph, arms=['b', 'r', 'n'], pfsConfigs={}):
        try:
            visits[0]
        except TypeError:
            visits = [visits]

        self.pfsConfigs = pfsConfigs
        self.visits = visits        
        self.spectrograph = spectrograph
        self.arms = arms
        
        self.data = collections.OrderedDict()
        for visit in self.visits:
            self.data[visit] = collections.OrderedDict()
            for arm in self.arms:
                self.data[visit][arm] = PfsArm(visit, spectrograph, arm)
                
    def read(self, dirName="."):
        for visit in self.visits:
            for arm in self.arms:
                self.data[visit][arm].read(dirName, pfsConfigs=self.pfsConfigs)

    def getFiberIdx(self, fiberId):
        return self.data.values()[0].values()[0].getFiberIdx(fiberId)

    def plot(self, fiberId=1, showFlux=None, showMask=False, showSky=False, showCovar=False):
        """Plot some or all of the contents of the PfsArms

        Default is to show the flux
        """
        show = dict(mask=showMask, sky=showSky, covar=showCovar)
        show.update(flux = not sum(show.values()) if showFlux is None else showFlux)

        fiberIdx = self.getFiberIdx(fiberId)

        xlabel = "Wavelength (micron)"
        for visit in self.visits:
            title = "%06d %d fiber %d" % (visit, self.spectrograph, fiberId)
            
            for name in ["flux", "mask", "sky"]:
                if not show[name]:
                    continue

                for arm in self.data[visit].values():
                    plt.plot(arm.lam[fiberIdx], getattr(arm, name)[fiberIdx], label=arm.arm,)

                plt.title("%s %s" % (title, name))
                plt.xlabel(xlabel)
                plt.legend(loc='best')

                if name in ("flux"):
                    plt.axhline(0, ls=':', color='black')

                plt.show()

            if show["covar"]:
                for arm in self.data[visit].values():
                    for i in range(arm.covar.shape[1]):
                        plt.plot(arm.lam[fiberIdx], arm.covar[fiberIdx][i], 
                                 label="%s covar[%d]" % (arm.arm, i))
                plt.xlabel(xlabel)
                plt.legend(loc='best')
    
                plt.title("%s %s" % (title, "covar"))
                plt.show()
