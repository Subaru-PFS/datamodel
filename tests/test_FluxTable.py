import sys
import unittest

import matplotlib
matplotlib.use("Agg")  # noqa

import numpy as np

import lsst.utils.tests

from pfs.datamodel import FluxTable, MaskHelper

display = None


class FluxTableTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.length = 4000
        self.minWavelength = 350.0
        self.maxWavelength = 1250.0
        numBad = 10
        rng = np.random.RandomState(12345)  # I have the same combination on my luggage
        wavelengthRange = self.maxWavelength - self.minWavelength
        self.wavelength = (self.minWavelength +
                           np.sort(rng.uniform(size=self.length).astype(np.float32))*wavelengthRange)
        self.flux = rng.uniform(size=self.length).astype(np.float32)
        self.error = rng.uniform(size=self.length).astype(np.float32)
        self.mask = np.zeros_like(self.flux, dtype=np.int32)
        self.flags = MaskHelper(missing=1)

        # Add some bad pixels
        bad = rng.choice(range(self.length), numBad, False)
        self.flux[bad] = 12345.6789
        self.mask[bad] = self.flags.get("missing")

        self.fluxTable = FluxTable(self.wavelength, self.flux, self.error, self.mask, self.flags)

    def assertFluxTable(self, ft):
        self.assertEqual(len(ft), self.length)
        self.assertFloatsEqual(ft.wavelength, self.wavelength)
        self.assertFloatsEqual(ft.flux, self.flux)
        self.assertFloatsEqual(ft.error, self.error)
        self.assertFloatsEqual(ft.mask, self.mask)
        self.assertDictEqual(ft.flags.flags, self.flags.flags)

    def testBasic(self):
        """Test basic functions"""
        self.assertFluxTable(self.fluxTable)

    def testFits(self):
        """Test I/O with FITS"""
        from astropy.io.fits import HDUList
        fits = HDUList()
        self.fluxTable.toFits(fits)
        ft = FluxTable.fromFits(fits)
        self.assertFluxTable(ft)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    setup_module(sys.modules["__main__"])
    from argparse import ArgumentParser
    parser = ArgumentParser(__file__)
    parser.add_argument("--display", help="Display backend")
    args, argv = parser.parse_known_args()
    display = args.display
    unittest.main(failfast=True, argv=[__file__] + argv)
