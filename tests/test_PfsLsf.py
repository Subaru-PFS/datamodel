"""Claude: FITS round trip of the line-spread function containers (PfsLsf)"""
import os
import tempfile
import unittest

import numpy as np

import lsst.utils.tests

from pfs.datamodel.pfsLsf import PfsArmLsfData, PfsLsf, PfsLsfInput, PfsLsfSurface


def makeArm(rng, arm="r", spectrograph=1, kind="disk", numFibers=5):
    """A detector LSF with some of everything"""
    fiberId = np.arange(1, 1 + 3*numFibers, 3)
    knots = np.linspace(0, 4175, 7)
    surfaces = {}
    if kind == "disk":
        for name, degree in (("radius", (2, 3)), ("rms", (3, 4)), ("S", (1, 1))):
            surfaces[name] = PfsLsfSurface(
                name=name,
                degree=degree,
                domain=(0.0, 4096.0, 0.0, 4176.0),
                coeffs=rng.normal(size=(degree[0] + 1)*(degree[1] + 1)),
                offsets={"mtp": {"U3-1": (0.01, 0.001), "U3-2": (-0.02, 0.002)}} if name == "rms" else {},
                reference={"mtp": None, "visit": "148314"} if name == "rms" else {},
                rms=0.01,
                chi2=123.4,
                ndof=100,
                nUsed=110,
            )
    return PfsArmLsfData(
        arm=arm,
        spectrograph=spectrograph,
        length=4176,
        kind=kind,
        fiberId=fiberId,
        surfaces=surfaces,
        extraction="optimal",
        blurOrder=4 if kind == "disk" else 2,
        centroidZeroPoint=0.002,
        xSplit=2048.0 if kind == "disk" else np.inf,
        affine=rng.normal(size=(3, 2)) if kind == "disk" else None,
        affineMaxError=1.0e-6,
        isEngineering=np.arange(numFibers) == 0,
        fiberGroups={"mtp": {int(ff): ("U3-1" if ii % 2 else "U3-2") for ii, ff in enumerate(fiberId)}},
        iisExtraBlur=(0.04, 0.08) if kind == "disk" else None,
        gaussianWidth=rng.uniform(1, 2, size=numFibers) if kind == "gaussian" else None,
        traceKnots=knots,
        traceValues=rng.uniform(0, 4096, size=(numFibers, len(knots))),
        wavelengthKnots=knots if kind == "disk" else None,
        wavelengthValues=rng.uniform(600, 900, size=(numFibers, len(knots))) if kind == "disk" else None,
        metadata={"visits": [148314, 148317], "scatteredLight": {"doWide": True}, "zero": np.float64(0.5)},
    )


class PfsLsfTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12345)

    def assertArmsEqual(self, got, expect):
        for name in ("arm", "spectrograph", "length", "kind", "extraction", "blurOrder", "centroidZeroPoint",
                     "xSplit", "affineMaxError", "fiberGroups", "iisExtraBlur"):
            self.assertEqual(getattr(got, name), getattr(expect, name), name)
        for name in ("fiberId", "isEngineering", "affine", "gaussianWidth", "traceKnots", "traceValues",
                     "wavelengthKnots", "wavelengthValues"):
            gg, ee = getattr(got, name), getattr(expect, name)
            if ee is None:
                self.assertIsNone(gg, name)
            else:
                np.testing.assert_array_equal(gg, ee, err_msg=name)
        self.assertEqual(got.metadata, {"visits": [148314, 148317], "scatteredLight": {"doWide": True},
                                        "zero": 0.5})
        self.assertEqual(set(got.surfaces), set(expect.surfaces))
        for name, ss in expect.surfaces.items():
            gg = got.surfaces[name]
            self.assertEqual((gg.name, gg.degree, gg.domain, gg.rms, gg.chi2, gg.ndof, gg.nUsed),
                             (ss.name, ss.degree, ss.domain, ss.rms, ss.chi2, ss.ndof, ss.nUsed))
            np.testing.assert_array_equal(gg.coeffs, ss.coeffs)
            self.assertEqual(gg.offsets, ss.offsets)
            self.assertEqual(gg.reference, ss.reference)

    def roundTrip(self, lsf):
        with tempfile.TemporaryDirectory() as tempDir:
            filename = os.path.join(tempDir, "lsf.fits")
            lsf.writeFits(filename)
            return PfsLsf.readFits(filename)

    def testArm(self):
        """The LSF of one detector"""
        lsf = PfsLsf([makeArm(self.rng)])
        copy = self.roundTrip(lsf)
        self.assertFalse(copy.isCombined)
        self.assertEqual(len(copy.arms), 1)
        self.assertArmsEqual(copy.arms[0], lsf.arms[0])

    def testCombined(self):
        """The LSF of a combination, with disk and Gaussian detector LSFs"""
        arms = [makeArm(self.rng, "b", 1, "gaussian"), makeArm(self.rng, "r", 1), makeArm(self.rng, "n", 1)]
        inputs = []
        for spectrum in range(4):
            for armIndex in range(3):
                inputs.append(PfsLsfInput(
                    spectrum=spectrum,
                    armIndex=armIndex,
                    fiberId=int(arms[armIndex].fiberId[spectrum]),
                    order=3,
                    indexPositions=np.array([10, 74, 138, 140]),
                    indexSamples=self.rng.uniform(0, 4000, size=4),
                    indexMaxError=1.0e-4,
                    weightMinIndex=10,
                    weightMaxIndex=140,
                    weightBinSize=50,
                    weightValues=self.rng.uniform(size=3).astype(np.float32),
                ))
        lsf = PfsLsf(arms, inputs)
        copy = self.roundTrip(lsf)
        self.assertTrue(copy.isCombined)
        for got, expect in zip(copy.arms, arms):
            self.assertArmsEqual(got, expect)
        self.assertEqual(len(copy.inputs), len(inputs))
        for got, expect in zip(copy.inputs, inputs):
            for name in ("spectrum", "armIndex", "fiberId", "order", "indexMaxError", "weightMinIndex",
                         "weightMaxIndex", "weightBinSize"):
                self.assertEqual(getattr(got, name), getattr(expect, name), name)
            for name in ("indexPositions", "indexSamples", "weightValues"):
                np.testing.assert_array_equal(getattr(got, name), getattr(expect, name), err_msg=name)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
