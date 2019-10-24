import sys
import unittest

import numpy as np
import astropy.wcs

import lsst.utils.tests

from pfs.datamodel.wavelengthArray import WavelengthArray

display = None


class WavelengthArrayTestCase(lsst.utils.tests.TestCase):
    def testBasics(self):
        """Test the basic behaviour of WavelenthArray"""
        minWl = 600
        maxWl = 900
        size = 50

        # Creation
        wlArray = WavelengthArray(minWl, maxWl, size)
        linspace = np.linspace(minWl, maxWl, size, dtype=np.float32)
        self.assertEqual(len(wlArray), size)
        self.assertEqual(len(wlArray), len(linspace))
        self.assertFloatsEqual(wlArray, linspace)

        # Iteration
        for xx, yy in zip(wlArray, linspace):
            self.assertFloatsEqual(xx, yy)

        # Indexing
        for ii in range(size):
            self.assertFloatsEqual(wlArray[ii], linspace[ii])
            self.assertFloatsEqual(wlArray[-ii], linspace[-ii])

        # Behaves like a numpy array
        self.assertFloatsEqual(wlArray.mean(), linspace.mean())
        self.assertFloatsEqual(wlArray.std(), linspace.std())

    def testIO(self):
        """Test that we can round-trip a WavelengthArray"""
        size = 50
        wlArray = WavelengthArray(600, 900, size)
        header = wlArray.toFitsHeader()
        copy = WavelengthArray.fromFitsHeader(header, size)
        self.assertEqual(len(copy), size)
        self.assertFloatsEqual(copy, wlArray)
        self.assertEqual(type(copy), type(wlArray))

    def testWcs(self):
        """Test that the WavelengthArray WCS header is correct"""
        size = 50
        wlArray = WavelengthArray(600, 900, size)
        header = wlArray.toFitsHeader()
        wcs = astropy.wcs.WCS(header)
        for ii in range(size):
            self.assertFloatsAlmostEqual(wlArray[ii], wcs.pixel_to_world(ii + 1).to(astropy.units.nm).value,
                                         atol=1.0e-4)


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
