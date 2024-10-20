import sys
import unittest

import astropy.io.fits
import numpy as np

import lsst.utils.tests

from pfs.datamodel.identity import CalibIdentity
from pfs.datamodel.pfsFiberNorms import PfsFiberNorms


class PfsFiberNormsTestCase(lsst.utils.tests.TestCase):
    """Test for PfsFiberNorms"""
    def setUp(self):
        self.rng = np.random.RandomState(12345)
        self.length = 123

        self.identity = CalibIdentity(visit0=12345, arm="r", spectrograph=1, obsDate="2024-04-15")
        self.fiberId = np.arange(1, 100, 3, dtype=int)
        self.numFibers = self.fiberId.size
        self.wavelength = self.rng.uniform(size=(self.numFibers, self.length))
        self.values = self.rng.uniform(1, 2, size=(self.numFibers, self.length))
        self.fiberProfilesHash = {1: 0x12345678, 2: 0x23456789}
        self.model = astropy.io.fits.ImageHDU(
            self.rng.uniform(size=1234),
            astropy.io.fits.Header(cards=dict(KEY="VALUE")),
        )
        self.metadata = {"KEY": "VALUE", "NUMBER": 67890}

        self.fiberNorms = PfsFiberNorms(
            self.identity,
            self.fiberId,
            self.wavelength,
            self.values,
            self.fiberProfilesHash,
            self.model,
            self.metadata,
        )

    def testLen(self):
        """Test PfsFiberNorms.__len__"""
        self.assertEqual(len(self.fiberNorms), self.numFibers)

    def testHash(self):
        """Test PfsFiberNorms.hash"""
        fiberNorms = PfsFiberNorms(
            self.identity,
            self.fiberId,
            self.wavelength,
            self.values + 1.0e-6,
            self.fiberProfilesHash,
            self.model,
            self.metadata,
        )
        self.assertNotEqual(fiberNorms.hash, self.fiberNorms.hash)

    def testIo(self):
        """Test PfsFiberNorms I/O

        Tests that we can round-trip through FITS with the "writeFits" and
        "readFits" methods.
        """
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            self.fiberNorms.writeFits(filename)
            fiberNorms = PfsFiberNorms.readFits(filename)
            self.assertEqual(fiberNorms.hash, self.fiberNorms.hash)
            self.assertEqual(fiberNorms.identity, self.identity)
            self.assertFloatsEqual(fiberNorms.fiberId, self.fiberId)
            self.assertFloatsEqual(fiberNorms.wavelength, self.wavelength)
            self.assertFloatsEqual(fiberNorms.values, self.values)
            self.assertEqual(fiberNorms.fiberProfilesHash, self.fiberProfilesHash)
            self.assertFloatsEqual(fiberNorms.model.data, self.model.data)
            for kk, vv in self.model.header.items():
                self.assertEqual(fiberNorms.model.header[kk], vv)
            for kk, vv in self.metadata.items():
                self.assertEqual(fiberNorms.metadata[kk], vv)

    def testSelect(self):
        """Test select"""
        select = slice(None, None, 2)
        fiberId = self.fiberId[select]
        fiberNorms = self.fiberNorms.select(fiberId=fiberId)
        self.assertEqual(len(fiberNorms), fiberId.size)
        self.assertTrue(np.array_equal(fiberNorms.fiberId, fiberId))
        self.assertTrue(np.array_equal(fiberNorms.wavelength, self.wavelength[select]))
        self.assertTrue(np.array_equal(fiberNorms.values, self.values[select]))
        self.assertEqual(fiberNorms.fiberProfilesHash, self.fiberNorms.fiberProfilesHash)

        selection = np.zeros(self.numFibers, dtype=bool)
        selection[select] = True
        fiberNorms = self.fiberNorms[selection]
        self.assertEqual(fiberNorms.fiberId.size, fiberId.size)
        self.assertEqual(len(fiberNorms), fiberId.size)
        self.assertTrue(np.array_equal(fiberNorms.fiberId, fiberId))
        self.assertTrue(np.array_equal(fiberNorms.wavelength, self.wavelength[select]))
        self.assertTrue(np.array_equal(fiberNorms.values, self.values[select]))
        self.assertEqual(fiberNorms.fiberProfilesHash, self.fiberNorms.fiberProfilesHash)


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    setup_module(sys.modules["__main__"])
    unittest.main()
