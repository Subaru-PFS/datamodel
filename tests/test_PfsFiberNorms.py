import sys
import unittest

import numpy as np

import lsst.utils.tests

from pfs.datamodel.identity import CalibIdentity
from pfs.datamodel.pfsFiberNorms import PfsFiberNorms


class PfsFiberNormsTestCase(lsst.utils.tests.TestCase):
    """Test for PfsFiberNorms"""
    def setUp(self):
        self.rng = np.random.RandomState(12345)
        self.numCoeffs = 3

        self.identity = CalibIdentity(visit0=12345, arm="r", spectrograph=1, obsDate="2024-04-15")
        self.fiberId = np.arange(1, 100, 3, dtype=int)
        self.numFibers = self.fiberId.size
        self.height = 123
        self.coeff = self.rng.uniform(1, 2, size=(self.numFibers, self.numCoeffs))
        self.metadata = {"KEY": "VALUE", "NUMBER": 67890}

        self.fiberNorms = PfsFiberNorms(self.identity, self.fiberId, self.height, self.coeff, self.metadata)

    def testLen(self):
        """Test PfsFiberNorms.__len__"""
        self.assertEqual(len(self.fiberNorms), self.numFibers)

    def testGetItem(self):
        """Test PfsFiberNorms.__getitem__"""
        for ii, fiberId in enumerate(self.fiberId):
            self.assertFloatsEqual(self.fiberNorms[fiberId], self.coeff[ii])
        self.assertRaises(KeyError, self.fiberNorms.__getitem__, 1234567890)

    def testContains(self):
        """Test PfsFiberNorms.__contains__"""
        for fiberId in self.fiberId:
            self.assertTrue(fiberId in self.fiberNorms)
        self.assertFalse(1234567890 in self.fiberNorms)

    def testIo(self):
        """Test PfsFiberNorms I/O

        Tests that we can round-trip through FITS with the "writeFits" and
        "readFits" methods.
        """
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            self.fiberNorms.writeFits(filename)
            fiberNorms = PfsFiberNorms.readFits(filename)
            self.assertEqual(fiberNorms.identity, self.identity)
            self.assertFloatsEqual(fiberNorms.fiberId, self.fiberId)
            self.assertEqual(fiberNorms.height, self.height)
            self.assertFloatsEqual(fiberNorms.coeff, self.coeff)
            for kk, vv in self.metadata.items():
                self.assertEqual(fiberNorms.metadata[kk], vv)


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    setup_module(sys.modules["__main__"])
    unittest.main()
