import unittest
from pfs.datamodel.drp import PfsObject
from pfs.datamodel.drp import PfsReference
from pfs.datamodel.drp import PfsMerged
from pfs.datamodel.drp import PfsSingle
from pfs.datamodel.drp import PfsArm
from pfs.datamodel.pfsFluxReference import PfsFluxReference
import re


class FileFormatTestCase(unittest.TestCase):
    """ Checks the format of example datamodel files are
        consistent with that specified in the corresponding
        datamodel classes.
    """

    def extractAttributes(self, cls, fileName):
        matches = re.search(cls.filenameRegex, fileName)
        if not matches:
            raise RuntimeError(f"Unable to parse filename: {fileName} using regex {cls.filenameRegex}")

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

    def testBadHashInFileName(self):
        with self.assertRaises(RuntimeError):
            self.extractAttributes(
                PfsObject,
                'pfsObject-07621-2,2-001-02468ace1234abcd-003-1234abcddeadbeef.fits'
            )

    def testPfsArm(self):
        d = self.extractAttributes(PfsArm, 'pfsArm-123450-b1.fits')
        self.assertEqual(d['visit'], 123450)
        self.assertEqual(d['arm'], 'b')
        self.assertEqual(d['spectrograph'], 1)

    def testPfsMerged(self):
        d = self.extractAttributes(PfsMerged, 'pfsMerged-012345.fits')
        self.assertEqual(d['visit'], 12345)

    def testPfsReference(self):
        d = self.extractAttributes(
            PfsReference, 'pfsReference-00100-07621-2,2-02468ace1234abcd.fits')
        self.assertEqual(d['tract'], 7621)
        self.assertEqual(d['patch'], '2,2')
        self.assertEqual(d['catId'], 100)
        self.assertEqual(d['objId'], 0x02468ace1234abcd)

    def testPfsSingle(self):
        d = self.extractAttributes(
            PfsSingle, 'pfsSingle-12300-76210-2,2-02468ace1234abcd-123450.fits')
        self.assertEqual(d['tract'], 76210)
        self.assertEqual(d['patch'], '2,2')
        self.assertEqual(d['catId'], 12300)
        self.assertEqual(d['objId'], 0x02468ace1234abcd)
        self.assertEqual(d['visit'], 123450)

    def testPfsObject(self):
        d = self.extractAttributes(
            PfsObject,
            'pfsObject-12345-07621-2,2-02468ace1234abcd-003-0x1234abcddeadbeef.fits')
        self.assertEqual(d['tract'], 7621)
        self.assertEqual(d['patch'], '2,2')
        self.assertEqual(d['objId'], 0x02468ace1234abcd)
        self.assertEqual(d['catId'], 12345)
        self.assertEqual(d['nVisit'], 3)
        self.assertEqual(d['pfsVisitHash'], 0x1234abcddeadbeef)

    def testPfsFluxReference(self):
        d = self.extractAttributes(
            PfsFluxReference, 'pfsFluxReference-654321.fits')
        self.assertEqual(d['visit'], 654321)
