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

    def testLsf(self):
        """Claude: the LSF is written and read with the spectra, follows subsets, and merges"""
        spectra = self.makePfsMerged()
        self.assertIsNone(spectra.lsf)
        lsf = makeLsf(spectra.fiberId, spectra.length)
        spectra.lsf = FakeLsf(lsf)  # converts itself
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            spectra.writeFits(filename)
            self.assertIsNone(spectra.lsf.calledWith)  # Keyed by fiberId: no keys needed
            copy = PfsMerged.readFits(filename)
        self.assertPfsMergedEqual(copy, spectra)
        self.assertEqual(len(copy.lsf.arms), 1)
        np.testing.assert_array_equal(copy.lsf.arms[0].fiberId, spectra.fiberId)
        np.testing.assert_array_equal(copy.lsf.arms[0].gaussianWidth, lsf.arms[0].gaussianWidth)
        self.assertEqual([ii.spectrum for ii in copy.lsf.inputs], list(spectra.fiberId))

        # A subset keeps the LSF, which is keyed by fiberId
        subset = copy[copy.fiberId > copy.fiberId[1]]
        self.assertIs(subset.lsf, copy.lsf)

        # Merging concatenates the LSFs, with the detector indices adjusted
        other = self.makePfsMerged()
        other.fiberId = other.fiberId + 1000
        other.lsf = makeLsf(other.fiberId, other.length)
        merged = PfsMerged.fromMerge([copy, other])
        self.assertEqual(len(merged.lsf.arms), 2)
        self.assertEqual({ii.armIndex for ii in merged.lsf.inputs if ii.spectrum >= 1000}, {1})

        # Merging spectra of which some have no LSF gives no LSF
        other.lsf = None
        self.assertIsNone(PfsMerged.fromMerge([copy, other]).lsf)


def makeLsf(fiberId, length, combined=True):
    """Claude: a simple LSF for spectra with the given ``fiberId`` (or target indices)"""
    from pfs.datamodel.pfsLsf import PfsArmLsfData, PfsLsf, PfsLsfInput
    arm = PfsArmLsfData(arm="r", spectrograph=1, length=length, kind="gaussian", fiberId=np.asarray(fiberId),
                        gaussianWidth=np.linspace(1.0, 2.0, len(fiberId)))
    inputs = []
    if combined:
        inputs = [PfsLsfInput(spectrum=int(ff), armIndex=0, fiberId=int(ff), order=1,
                              indexPositions=np.array([0, length - 1]), indexSamples=np.array([0.0, 99.0]),
                              indexMaxError=0.0, weightMinIndex=0, weightMaxIndex=length - 1,
                              weightBinSize=50, weightValues=np.ones(3, dtype=np.float32))
                  for ff in fiberId]
    return PfsLsf([arm], inputs)


class FakeLsf:
    """Claude: stands in for drp_stella's LSF, which converts itself with ``toDatamodel``"""
    def __init__(self, lsf, expectKeys=None):
        self.lsf = lsf
        self.expectKeys = expectKeys
        self.calledWith = "not called"

    def toDatamodel(self, keys=None):
        self.calledWith = keys
        return self.lsf


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    setup_module(sys.modules["__main__"])
    unittest.main(failfast=True)
