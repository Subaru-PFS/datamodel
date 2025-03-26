import sys
from typing import Callable, TYPE_CHECKING
import unittest

import astropy.io.fits

import lsst.afw.image.testUtils
import lsst.utils.tests
import numpy as np
from pfs.datamodel import Identity, MaskHelper
from pfs.datamodel.drp import PfsMerged

display = None


class PfsFiberArraySetTestCase(lsst.utils.tests.TestCase):
    """We test on PfsMerged, as a subclass of PfsFiberArraySet"""
    if TYPE_CHECKING:
        assertFloatsEqual: Callable[..., None]
        assertFloatsAlmostEqual: Callable[..., None]

    def setUp(self):
        self.rng = np.random.RandomState(12345)  # I have the same combination on my luggage

    def makePfsMerged(self, num: int = 5, length=1000) -> PfsMerged:
        """Make a PfsMerged with random values

        Parameters
        ----------
        num : `int`, optional
            Number of fibers.
        length : `int`, optional
            Number of wavelength points.

        Returns
        -------
        PfsMerged : `PfsMerged`
            PfsMerged with random values.
        """
        identity = Identity(visit=12345)
        fiberId = self.rng.randint(1000, size=num)
        wavelength = np.tile(np.linspace(300, 1300, length, dtype=float), (num, 1))
        flux = self.rng.uniform(size=(num, length)).astype(np.float32)
        mask = self.rng.randint(0xFFFF, size=(num, length)).astype(np.int32)
        sky = self.rng.uniform(size=(num, length)).astype(np.float32)
        norm = self.rng.uniform(size=(num, length)).astype(np.float32)
        covar = self.rng.uniform(size=(num, 3, length)).astype(np.float32)
        flags = MaskHelper(FOO=0, BAR=3, BAZ=13, QIX=7)
        metadata = dict(FOO="bar", BAZ="qux")

        return PfsMerged(
            identity, fiberId, wavelength, flux, mask, sky, norm, covar, flags, metadata
        )

    def assertPfsMergedEqual(self, left: PfsMerged, right: PfsMerged):
        """Assert that two `PfsMerged` are equal"""
        self.assertEqual(left.identity, right.identity)
        self.assertFloatsEqual(left.fiberId, right.fiberId)
        self.assertFloatsEqual(left.wavelength, right.wavelength)
        self.assertFloatsEqual(left.flux, right.flux)
        self.assertFloatsEqual(left.mask, right.mask)
        self.assertFloatsEqual(left.sky, right.sky)
        self.assertFloatsEqual(left.norm, right.norm)
        self.assertFloatsEqual(left.covar, right.covar)
        self.assertDictEqual(left.flags.flags, right.flags.flags)
        for key in right.metadata:
            self.assertIn(key, right.metadata)
            self.assertEqual(left.metadata[key], right.metadata[key])

    def testBasic(self):
        """Test basic functionality"""
        num = 5
        spectra = self.makePfsMerged(num)
        self.assertEqual(len(spectra), num)

    def testIO(self):
        """Test I/O functionality"""
        spectra = self.makePfsMerged()
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            spectra.writeFits(filename)
            copy = PfsMerged.readFits(filename)
            self.assertPfsMergedEqual(copy, spectra)

            # Check that the DAMD_VER is present
            with astropy.io.fits.open(filename) as hdus:
                self.assertIn("DAMD_VER", hdus[0].header)

    def testSameWavelength(self):
        """Test I/O when the wavelengths are the same"""
        spectra = self.makePfsMerged()
        spectra.wavelength = np.tile(np.linspace(300, 1300, spectra.length, dtype=float), (len(spectra), 1))
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            spectra.writeFits(filename)
            copy = PfsMerged.readFits(filename)
            self.assertPfsMergedEqual(copy, spectra)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    setup_module(sys.modules["__main__"])
    unittest.main(failfast=True)
