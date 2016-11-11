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
    fileNameFormat = "pfsFiberTrace-%10s-0-%1s%1d.fits"

    def __init__(self, obsDate, spectrograph, arm):
        self.obsDate = obsDate
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

    def read(self, dirName="."):
        """Read self's pfsFiberTrace file from directory dirName"""
        if not pyfits:
            raise RuntimeError("I failed to import pyfits, so cannot read from disk")

        fileName = PfsFiberTrace.fileNameFormat % (self.obsDate, self.arm, self.spectrograph)
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
        if fileName is None:
            fileName = self.fileNameFormat % (self.obsDate, self.arm, self.spectrograph)
        with open(os.path.join(dirName, fileName), "w") as fd:
            hdus.writeto(fd)            
