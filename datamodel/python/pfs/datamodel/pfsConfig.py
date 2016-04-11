import os
import pyfits

from pfs.datamodel.utils import calculate_pfsConfigId

class PfsConfig(object):
    """A class corresponding to a single pfsConfig file"""
    
    fileNameFormat = "pfsConfig-0x%08x.fits"

    def __init__(self, pfsConfigId):
        self.pfsConfigId = pfsConfigId
        
    def read(self, dirName="."):
        """Read self's pfsConfig file from directory dirName"""
        
        fd = pyfits.open(os.path.join(dirName, self.fileNameFormat % self.pfsConfigId))

        hdu = fd["CONFIG"]
        hdr, data = hdu.header, hdu.data
    
        if False:
            for k, v in hdr.items():
                print "%8s %s" % (k, v)
            
        self.fiberId = data['fiberId']
        self.catId = data['catId']
        self.objId = data['objId']
        self.ra = data['ra']
        self.dec = data['dec']
        self.fiberFlux = data['fiber flux']
        self.mpsCentroid = data['mps centroid']
        
        assert self.pfsConfigId == calculate_pfsConfigId(self.fiberId, self.ra, self.dec)        
