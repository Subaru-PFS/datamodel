import collections
import numpy as np
import os
try:
    import pyfits
except ImportError:
    pyfits = None
import re

class PfsFiberTrace(object):
    """A class corresponding to a single fiberTrace file"""
    #
    # Flags for MASKs
    #
    flags = dict(NODATA = 0x1,          # this pixel contains no data
                 )

    fileNameFormat = "pfsFiberTrace-%10s-0-%1d%1s.fits"

    def __init__(self, calibDate, spectrograph, arm):
        self.calibDate = calibDate
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
        
        self.fiberId = []
        self.xCenter = []
        self.yCenter = []
        self.yLow = []
        self.yHigh = []
        self.coeffs = []
        self.profiles = []

    @staticmethod
    def readFits(fileName, hdu, flags):
        dirName = os.path.dirname( fileName )
        
        info = PfsFiberTrace.getInfo( fileName )
        calibDate = info[0]['calibDate']
        spectrograph = info[0]['spectrograph']
        arm = info[0]['arm']
        pfsFiberTrace = PfsFiberTrace( calibDate, spectrograph, arm )
        
        pfsFiberTrace.read(dirName=dirName)
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
        matches = re.search("pfsFiberTrace-(\d{4})-(\d{2})-(\d{2})-0-(\d{1})("+armsRe+").fits", filename)
        if not matches:
            message = 'pfsFiberTrace.getInfo: Cannot interpret filename <',filename,'>'
            raise Exception(message)
        year, month, day, spectrograph, arm = matches.groups()
        if int(spectrograph) < minSpectrograph or int(spectrograph) > maxSpectrograph:
            message = 'spectrograph (=',spectrograph,') out of bounds'
            raise Exception(message)
        if arm not in arms:
            message = 'arm (=',arm,') not a valid arm'
            raise Exception(message)

        calibDate = '%04d-%02d-%02d' % (int(year), int(month), int(day))
        info = dict(calibDate=calibDate, arm=arm, spectrograph=int(spectrograph))
        if os.path.exists(filename):
            header = afwImage.readMetadata(filename)
            info = self.getInfoFromMetadata(header, info=info)
        return info, [info]

    def read(self, dirName="."):
        """Read self's pfsFiberTrace file from directory dirName"""
        if not pyfits:
            raise RuntimeError("I failed to import pyfits, so cannot read from disk")

        fileName = PfsFiberTrace.fileNameFormat % (self.calibDate, self.spectrograph, self.arm)
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
            hdr, data = hdu.header, hdu.data
            
            if hduName == "FUNCTION":
                self.fiberId = data['FIBERID']
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
        
        coeffArr = np.array(self.coeffs, dtype=np.float32)
        hdu = pyfits.BinTableHDU.from_columns([
            pyfits.Column(name = 'FIBERID', format = 'J',
                          array=np.array(self.fiberId, dtype=np.int32)),
            pyfits.Column(name = 'XCENTER', format = 'E',
                          array=np.array(self.xCenter, dtype=np.float32)),
            pyfits.Column(name = 'YCENTER', format = 'J',
                          array=np.array(self.yCenter, dtype=np.int32)),
            pyfits.Column(name = 'YLOW', format = 'J',
                          array=np.array(self.yLow, dtype=np.int32)),
            pyfits.Column(name = 'YHIGH', format = 'J',
                          array=np.array(self.yHigh, dtype=np.int32)),
            pyfits.Column(name = 'COEFFS', format = '%dE'%(self.order+1),
                          array=coeffArr)
        ])
        hdu.name = 'FUNCTION'
        hdus.append(hdu)

        hdu = pyfits.ImageHDU(self.profiles)
        hdu.name = "PROFILE"
        hdus.append(hdu)

        # clobber=True in writeto prints a message, so use open instead
        if fileName == None:
            fileName = self.fileNameFormat % (self.calibDate, self.spectrograph, self.arm)
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
    def __init__(self, visit, spectrograph, arms=['b', 'r', 'n']):
        self.visit = visit        
        self.spectrograph = spectrograph
        self.arms = arms

        self.data = collections.OrderedDict()
        for arm in self.arms:
            self.data[arm] = PfsFiberTrace(visit, spectrograph, arm)
                
    def read(self, dirName="."):
        for arm in self.arms:
            self.data[arm].read(dirName)

    def getFiberIdx(self, fiberId):
        return self.data.values()[0].getFiberIdx(fiberId)
