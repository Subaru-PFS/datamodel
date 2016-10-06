import collections
import os
import re
import numpy as np
try:
    import pyfits
except ImportError:
    pyfits = None
#import matplotlib.pyplot as plt

from pfs.datamodel.pfsConfig import PfsConfig

class PfsFiberTrace(object):
    """A class corresponding to a single fiberTrace file"""
    #
    # Flags for MASKs
    #
    flags = dict(NODATA = 0x1,          # this pixel contains no data
                 )

    fileNameFormat = "pfsFiberTrace-%06d-%1d%1s.fits"

    def __init__(self, visit, spectrograph, arm):#, pfsConfigId=None, pfsConfig=None):
        #self.pfsConfigId = pfsConfigId
        self.visit = visit
        self.spectrograph = spectrograph
        self.arm = arm

        self.fwhm = 0.
        self.threshold = 0.
        self.nTerms = 0
        self.saturationLevel = 0.
        self.minLength = 0
        self.maxLength = 0
        self.nLost = 0
        self.traceFunction = 'POLYNOMIAL'
        self.order = 0
        self.xLow = 0.
        self.xHigh = 0.
        self.nCutLeft = 0
        self.nCutRight = 0
        self.interpol = 'SPLINE3'
#        READNOISE WHY????
        self.swathLength = 0
        self.overSample = 0
        self.maxIterSF = 0
        self.maxIterSig = 0
        self.lambdaSF = 0.
        self.lambdaSP = 0.
        self.lambdaWing = 0.
        self.lSigma = 0.
        self.uSigma = 0.
        
        self.xCenter = []
        self.yCenter = []
        self.yLow = []
        self.yHigh = []
        self.coeffs = []
        self.profiles = []

#        self.pfsConfig = pfsConfig
#        if self.pfsConfig and self.pfsConfigId != self.pfsConfig.pfsConfigId:
#            raise RuntimeError("pfsConfigId == 0x%08x != pfsConfig.pfsConfigId == 0x%08x" %
#                               (self.pfsConfigId, self.pfsConfig.pfsConfigId))

    @staticmethod
    def readFits(fileName):#, pfsConfigs=None):
        dirName = os.path.dirname( fileName )
        
        info = PfsFiberTrace.getInfo( fileName )
        visit = info[0]['visit']
        spectrograph = info[0]['spectrograph']
        arm = info[0]['arm']
        pfsFiberTrace = PfsFiberTrace( visit, spectrograph, arm )
        
        pfsFiberTrace.read(dirName=dirName)#, pfsConfigs=pfsConfigs)
        return pfsFiberTrace
    
    @staticmethod
    def getInfo(fileName):
        """Get information about the image from the filename and its contents

        @param filename    Name of file to inspect
        @return File properties; list of file properties for each extension
        """
        minSpectrograph = 1
        maxSpectrograph = 4
        arms = ['b', 'r', 'n', 'm']
        armsRe = '[b,r,n,m]'
        path, filename = os.path.split(fileName)
        matches = re.search("pfsFiberTrace-(\d{6})-(\d{1})("+armsRe+").fits", filename)
        visit, spectrograph, arm = matches.groups()
        if int(spectrograph) < minSpectrograph or int(spectrograph) > maxSpectrograph:
            message = 'spectrograph (=',spectrograph,') out of bounds'
            raise Exception(message)
        if arm not in arms:
            message = 'arm (=',arm,') not a valid arm'
            raise Exception(message)

        info = dict(visit=int(visit, base=10), arm=arm, spectrograph=int(spectrograph))
        if os.path.exists(filename):
            header = afwImage.readMetadata(filename)
            info = self.getInfoFromMetadata(header, info=info)
        return info, [info]

    def read(self, dirName="."):#, pfsConfigs=None):
        """Read self's pfsFiberTrace file from directory dirName

#        If provided, pfsConfigs is a dict of pfsConfig objects, indexed by pfsConfigId
        """
        import pdb; pdb.set_trace()
        if not pyfits:
            raise RuntimeError("I failed to import pyfits, so cannot read from disk")

        fileName = PfsFiberTrace.fileNameFormat % (self.visit, self.spectrograph, self.arm)
        fd = pyfits.open(os.path.join(dirName, fileName)) 
        
        prihdr = fd[0].header
        self.fwhm = prihdr['FWHM']
        self.threshold = prihdr['THRESH']
        self.nTerms = prihdr['NTERMS']
        self.saturationLevel = prihdr['SATLEVEL']
        self.minLength = prihdr['MINLEN']
        self.maxLength = prihdr['MAXLEN']
        self.nLost = prihdr['NLOST']
        self.traceFunction = prihdr['FUNC']
        self.order = prihdr['ORDER']
        self.xLow = prihdr['XLOW']
        self.xHigh = prihdr['XHIGH']
        self.nCutLeft = prihdr['NCUTL']
        self.nCutRight = prihdr['NCUTR']
        self.interpol = prihdr['INTERPOL']
#        READNOISE WHY????
        self.swathLength = prihdr['SWATHLEN']
        self.overSample = prihdr['OSAMPLE']
        self.maxIterSF = prihdr['MAXITSF']
        self.maxIterSig = prihdr['MAXITSIG']
        self.lambdaSF = prihdr['LAMBDASF']
        self.lambdaSP = prihdr['LAMBDASP']
        self.lambdaWing = prihdr['LAMBWING']
        self.lSigma = prihdr['LSIGMA']
        self.uSigma = prihdr['USIGMA']

        for hduName in ["FUNCTION", "PROFILE"]:
            hdu = fd[hduName]
            print 'reading hdu ',hdu
            hdr, data = hdu.header, hdu.data
        
            if False:
                for k, v in hdr.items():
                    print "%8s %s" % (k, v)
            
            if hduName == "FUNCTION":
                print 'data = ',len(data),': ',data
                self.xCenter = data['XCENTER']
                self.yCenter = data['YCENTER']
                self.yLow = data['YLOW']
                self.yHigh = data['YHIGH']
                self.coeffs = data['COEFFS']
            elif hduName == "PROFILE":
                self.profiles = data
            else:
                raise RuntimeError("Unexpected HDU %s reading %s" % (hduName, fileName))
        
    def writeFits(self, fileName, flags=None):
        print 'writing <',fileName,'>'
        dirName, fName = os.path.split(fileName)
        self.write(dirName=dirName, fileName = fName)
        
    def write(self, dirName=".", fileName=None):
        if not pyfits:
            raise RuntimeError("I failed to import pyfits, so cannot read from disk")

        hdus = pyfits.HDUList()

        hdr = pyfits.Header()
        hdr['FWHM'] = self.fwhm
        hdr['THRESH'] = self.threshold
        hdr['NTERMS'] = self.nTerms
        hdr['SATLEVEL'] = self.saturationLevel
        hdr['MINLEN'] = self.minLength
        hdr['MAXLEN'] = self.maxLength
        hdr['NLOST'] = self.nLost
        hdr['FUNC'] = self.traceFunction
        hdr['ORDER'] = self.order
        hdr['XLOW'] = self.xLow
        hdr['XHIGH'] = self.xHigh
        hdr['NCUTL'] = self.nCutLeft
        hdr['NCUTR'] = self.nCutRight
        hdr['INTERPOL'] = self.interpol
#        READNOISE WHY????
        hdr['SWATHLEN'] = self.swathLength
        hdr['OSAMPLE'] = self.overSample
        hdr['MAXITSF'] = self.maxIterSF
        hdr['MAXITSIG'] = self.maxIterSig
        hdr['LAMBDASF'] = self.lambdaSF
        hdr['LAMBDASP'] = self.lambdaSP
        hdr['LAMBWING'] = self.lambdaWing
        hdr['LSIGMA'] = self.lSigma
        hdr['USIGMA'] = self.uSigma
        hdr.update()
        hdus.append(pyfits.PrimaryHDU(header=hdr))
        
        hdu = pyfits.BinTableHDU.from_columns([
            pyfits.Column(name = 'XCENTER', format = 'E',
                          array=np.array(self.xCenter, dtype=np.float32)),
            pyfits.Column(name = 'YCENTER', format = 'I',
                          array=np.array(self.yCenter, dtype=np.int16)),
            pyfits.Column(name = 'YLOW', format = 'I',
                          array=np.array(self.yLow, dtype=np.int16)),
            pyfits.Column(name = 'YHIGH', format = 'I',
                          array=np.array(self.yHigh, dtype=np.int16)),
            pyfits.Column(name = 'COEFFS', format = 'E',
                          array=np.array(self.coeffs, dtype=np.float32)),
        ])
        hdu.name = 'FUNCTION'
        hdus.append(hdu)

        hdu = pyfits.ImageHDU(self.profiles)
        hdu.name = "PROFILE"
        hdus.append(hdu)

        # clobber=True in writeto prints a message, so use open instead
        if fileName == None:
            fileName = self.fileNameFormat % (self.visit, self.spectrograph, self.arm)
        print 'fileName = <',os.path.join(dirName, fileName),'>'
        with open(os.path.join(dirName, fileName), "w") as fd:
            hdus.writeto(fd)            

    def getFiberIdx(self, fiberId):
        """Convert a fiberId to a fiber index (checking the range)"""
        if fiberId <= 0 or fiberId > len(self.lam):
            raise IndexError("fiberId %d is out of range %d..%d" % (fiberId, 1, len(self.lam)))

        return fiberId - 1

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class PfsFiberTraceSet(object):
    """Manipulate a set of pfsFiberTraces corresponding to a single visit"""
    def __init__(self, visit, spectrograph, arms=['b', 'r', 'n']):#, pfsConfigId=None, pfsConfig=None,
                 #pfsConfigs={}):
#        self.pfsConfigs = pfsConfigs
#        self.pfsConfigId = pfsConfigId
        self.visit = visit        
        self.spectrograph = spectrograph
        self.arms = arms
      
#        if pfsConfig:
#            if self.pfsConfigId:
#                if self.pfsConfigId != pfsConfig.pfsConfigId:
#                    raise RuntimeError("pfsConfigId == 0x%08x != pfsConfig.pfsConfigId == 0x%08x" %
#                                       (self.pfsConfigId, pfsConfig.pfsConfigId))
#            else:
#                self.pfsConfigId = pfsConfig.pfsConfigId
#
#            self.pfsConfigs[self.pfsConfigId] = pfsConfig
#
#        if self.pfsConfigId in self.pfsConfigs:
#            self.pfsConfig = self.pfsConfigs[self.pfsConfigId]
#        else:
#            self.pfsConfig = None

        self.data = collections.OrderedDict()
        for arm in self.arms:
            self.data[arm] = PfsFiberTrace(visit, spectrograph, arm)#, self.pfsConfigId, self.pfsConfig)
                
    def read(self, dirName="."):
        for arm in self.arms:
            self.data[arm].read(dirName)#, pfsConfigs=self.pfsConfigs)

#            if not self.pfsConfig:
#                self.pfsConfig = self.data[arm].pfsConfig

    def getFiberIdx(self, fiberId):
        return self.data.values()[0].getFiberIdx(fiberId)
