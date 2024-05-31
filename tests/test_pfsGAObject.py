import os
import re
import numpy as np
from unittest import TestCase

from pfs.datamodel import Target, TargetType
from pfs.datamodel import Observations
from pfs.datamodel import MaskHelper
from pfs.datamodel import GAFluxTable
from pfs.datamodel import PfsGAObject, StellarParams, Abundances, VelocityCorrections

class PfsGAObjectTestCase(TestCase):
    """ Check the format of example datamodel files are
        consistent with that specified in the corresponding
        datamodel classes.
    """

    def makePfsGAObject(self):
        """Construct a PfsGAObject with dummy values for testing."""

        catId = 12345
        tract = 1
        patch = '1,1'
        objId = 123456789
        ra = -100.63654
        dec = -68.591576
        targetType = TargetType.SCIENCE

        target = Target(catId, tract, patch, objId, ra, dec, targetType)

        visit = np.array([ 83219, 83220 ])
        arm = np.array([ 'b', 'm', ])
        spectrograph = np.array([1, 1])
        pfsDesignId = np.array([8854764194165386399, 8854764194165386400])
        fiberId = np.array([476, 476])
        pfiNominal = np.array([[ra, dec], [ra, dec]])
        pfiCenter = np.array([[ra, dec], [ra, dec]])

        observations = Observations(visit, arm, spectrograph, pfsDesignId, fiberId, pfiNominal, pfiCenter)

        npix = 4096
        wavelength = np.concatenate([
            np.linspace(380, 650, npix, dtype=np.float32),
            np.linspace(710, 885, npix, dtype=np.float32)
        ])
        flux = np.zeros_like(wavelength)
        error = np.zeros_like(wavelength)
        model = np.zeros_like(wavelength)
        cont = np.zeros_like(wavelength)
        norm_flux = np.zeros_like(wavelength)
        norm_error = np.zeros_like(wavelength)
        norm_model = np.zeros_like(wavelength)
        mask = np.zeros_like(wavelength, dtype=np.int32)
        sky = np.zeros_like(wavelength)
        covar = np.zeros((3, wavelength.size), dtype=np.float32)    # Tridiagonal covariance matrix of flux
        covar2 = np.zeros((1, 1), dtype=np.float32)                 # ?

        flags = MaskHelper()                                        # {'BAD': 0, 'BAD_FIBERTRACE': 11, 'BAD_FLAT': 9, 'BAD_FLUXCAL': 13, 'BAD_SKY': 12, 'CR': 3, 'DETECTED': 5, 'DETECTED_NEGATIVE': 6, 'EDGE': 4, 'FIBERTRACE': 10, 'INTRP': 2, 'IPC': 14, 'NO_DATA': 8, 'REFLINE': 15, 'SAT': 1, 'SUSPECT': 7, 'UNMASKEDNAN': 16})
        metadata = {}                                               # Key-value pairs to put in the header
        fluxTable = GAFluxTable(wavelength, flux, error, model, cont,
                                norm_flux, norm_error, norm_model,
                                mask, flags)

        stellarParams = StellarParams(
            method=np.array(['rvfit', 'rvfit', 'rvfit', 'rvfit', 'rvfit']),
            frame=np.array(['helio', '', '', '', '']),
            param=np.array(['v_los', 'Fe_H', 'T_eff', 'log_g', 'a_Fe']),
            covarId=np.array([0, 1, 2, 3, 4]),
            unit=np.array(['km s-1', 'dex', 'K', '', 'dex']),
            value=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            valueErr=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            flag=np.array([False, False, False, False, False]),
            status=np.array(['', '', '', '', '']),
        )

        velocityCorrections = VelocityCorrections(
            visit=visit,
            JD=np.zeros_like(visit, dtype=np.float32),
            helio=np.zeros_like(visit, dtype=np.float32),
            bary=np.zeros_like(visit, dtype=np.float32),
        )

        abundances = Abundances(
            method=np.array(['chemfit', 'chemfit', 'chemfit']),
            element=np.array(['Mg', 'Ti', 'Si']),
            covarId=np.array([0, 1, 2]),
            value=np.array([0.0, 0.0, 0.0]),
            valueErr=np.array([0.0, 0.0, 0.0]),
        )

        paramsCovar = np.eye(3, dtype=np.float32)
        abundCovar = np.eye(4, dtype=np.float32)
        notes = None

        return PfsGAObject(target, observations,
                                  wavelength, flux, mask, sky, covar, covar2,
                                  flags, metadata,
                                  fluxTable,
                                  stellarParams,
                                  velocityCorrections,
                                  abundances,
                                  paramsCovar,
                                  abundCovar,
                                  notes)
    
    def assertPfsGAObject(self, lhs, rhs):
        np.testing.assert_array_equal(lhs.observations.visit, rhs.observations.visit)

        # TODO: add more tests here

    def extractAttributes(self, cls, fileName):
        matches = re.search(cls.filenameRegex, fileName)
        if not matches:
            self.fail(
                "Unable to parse filename: {} using regex {}"
                .format(fileName, cls.filenameRegex))

        # Cannot use algorithm in PfsSpectra._parseFilename(),
        # specifically cls.filenameKeys, due to ambiguity in parsing
        # integers in hex format (eg objId). Need to parse cls.filenameFormat
        ff = re.search(r'^[a-zA-Z]+(.*)\.fits', cls.filenameFormat)[1]
        cmps = re.findall(r'-{0,1}(0x){0,1}\%\((\w+)\)\d*(\w)', ff)
        fmts = [(kk, tt) for ox, kk, tt in cmps]

        d = {}
        for (kk, tt), vv in zip(fmts, matches.groups()):
            if tt == 'd':
                ii = int(vv)
            elif tt == 'x':
                ii = int(vv, 16)
            elif tt == 's':
                ii = vv
            d[kk] = ii
        return d
    
    def test_filenameRegex(self):
        d = self.extractAttributes(
                PfsGAObject,
                'pfsGAObject-07621-01234-2,2-02468ace1234abcd-003-0x0123456789abcdef.fits')
        self.assertEqual(d['catId'], 7621)
        self.assertEqual(d['tract'], 1234)
        self.assertEqual(d['patch'], '2,2')
        self.assertEqual(d['objId'], 163971054118939597)
        self.assertEqual(d['nVisit'], 3)
        self.assertEqual(d['pfsVisitHash'], 81985529216486895)

    def test_getIdentity(self):
        """Construct a PfsGAObject and get its identity."""

        pfsGAObject = self.makePfsGAObject()
        identity = pfsGAObject.getIdentity()
        filename = pfsGAObject.filenameFormat % identity

        self.assertEqual('pfsGAObject-12345-00001-1,1-00000000075bcd15-002-0x05a95bc24d8ce16f.fits', filename)

    def test_validate(self):
        """Construct a PfsGAObject and run validation."""

        pfsGAObject = self.makePfsGAObject()
        pfsGAObject.validate()

    def test_writeFits_fromFits(self):
        """Construct a PfsGAObject and save it to a FITS file."""

        pfsGAObject = self.makePfsGAObject()

        dirName = os.path.splitext(__file__)[0]
        if not os.path.exists(dirName):
            os.makedirs(dirName)

        id = pfsGAObject.getIdentity()
        filename = os.path.join(dirName, pfsGAObject.filenameFormat % id)
        if os.path.exists(filename):
            os.unlink(filename)

        try:
            pfsGAObject.writeFits(filename)
            other = PfsGAObject.readFits(filename)
            self.assertPfsGAObject(pfsGAObject, other)
        except Exception as e:
            raise
        finally:
            os.unlink(filename)
