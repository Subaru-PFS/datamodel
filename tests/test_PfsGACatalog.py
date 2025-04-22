import os
import re
import numpy as np
from unittest import TestCase

from pfs.datamodel import TargetType, Observations
from pfs.datamodel.ga import PfsGACatalog, PfsGACatalogNotes, GACatalogTable


class PfsGACatalogTestCase(TestCase):
    """ Check the format of example datamodel files are
        consistent with that specified in the corresponding
        datamodel classes.
    """

    def makePfsGACatalog(self):
        """Construct a PfsGACatalog with dummy values for testing."""

        catId = 12345

        observations = Observations(
            visit=np.array([10000, 10001, 10002, 10005]),
            arm=np.array(['bmn', 'bmn', 'bmn', 'bmn']),
            spectrograph=np.array(4 * [-1]),
            pfsDesignId=np.array([100001, 100002, 100003, 100004]),
            fiberId=np.array(4 * [-1]),
            pfiNominal=np.array(4 * [2 * [np.nan]]),
            pfiCenter=np.array(4 * [2 * [np.nan]]),
            obsTime=np.array(['2025-01-10 12:00:00', '2025-01-10 12:15:00', '2025-01-10 12:30:00', '2025-01-10 13:00:00']),
            expTime=np.array([1800.0, 1800.0, 1800.0, 1800.0]),
        )

        catalog = GACatalogTable(
            catId=np.array([10001, 10001, 10001]),
            objId=np.array([1, 2, 3]),
            gaiaId=np.array([11, 12, 13]),
            ps1Id=np.array([21, 22, 23]),
            hscId=np.array([21, 22, 23]),
            miscId=np.array([-1, -1, 33]),
            ra=np.array([210.01, 210.02, 210.03]),
            dec=np.array([67.1, 67.2, 67.3]),
            epoch=np.array([2016.0, 2016.0, 2016.0]),
            pmRa=np.array([0.0, 0.0, 0.0]),
            pmDec=np.array([0.0, 0.0, 0.0]),
            parallax=np.array([0.0, 0.0, 0.0]),
            targetType=np.array([TargetType.SCIENCE, TargetType.SCIENCE, TargetType.SCIENCE]),
            proposalId=np.array(['A', 'A', 'A']),
            obCode=np.array(['A', 'B', 'C']),
            
            nVisit_b=np.array([3, 3, 3]),
            nVisit_m=np.array([3, 3, 3]),
            nVisit_r=np.array([3, 3, 3]),
            nVisit_n=np.array([3, 3, 3]),

            expTimeEff_b=np.array([5400, 5400, 5400]),
            expTimeEff_m=np.array([5400, 5400, 5400]),
            expTimeEff_r=np.array([5400, 5400, 5400]),
            expTimeEff_n=np.array([5400, 5400, 5400]),

            rv=np.array([0.0, 0.0, 0.0]),
            rvErr=np.array([0.0, 0.0, 0.0]),
            tEff=np.array([0.0, 0.0, 0.0]),
            tEffErr=np.array([0.0, 0.0, 0.0]),
            m_h=np.array([0.0, 0.0, 0.0]),
            m_hErr=np.array([0.0, 0.0, 0.0]),
            logg=np.array([0.0, 0.0, 0.0]),
            loggErr=np.array([0.0, 0.0, 0.0]),

            flag=np.array([False, False, False]),
            status=np.array(['', '', '']),
        )

        return PfsGACatalog(
            catId, observations, catalog, metadata=None, notes=None
        )

    def assertPfsGACatalog(self, lhs, rhs):
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
            PfsGACatalog,
            'pfsGACatalog-07621-003-0x0123456789abcdef.fits')
        self.assertEqual(d['catId'], 7621)
        self.assertEqual(d['nVisit'], 3)
        self.assertEqual(d['pfsVisitHash'], 81985529216486895)

    def test_validate(self):
        """Construct a PfsGACatalog and run validation."""

        pfsGACatalog = self.makePfsGACatalog()
        pfsGACatalog.validate()

    def test_writeFits_fromFits(self):
        """Construct a PfsGACatalog and save it to a FITS file."""

        pfsGACatalog = self.makePfsGACatalog()

        dirName = os.path.splitext(__file__)[0]
        if not os.path.exists(dirName):
            os.makedirs(dirName)

        id = pfsGACatalog.getIdentity()
        filename = os.path.join(dirName, pfsGACatalog.filenameFormat % id)
        if os.path.exists(filename):
            os.unlink(filename)

        try:
            pfsGACatalog.writeFits(filename)
            other = PfsGACatalog.readFits(filename)
            self.assertPfsGACatalog(pfsGACatalog, other)
        except Exception as e:
            raise e
        finally:
            if os.path.exists(filename):
                os.unlink(filename)
