import sys
import unittest

import numpy as np

import lsst.utils.tests

from pfs.datamodel.pfsThetaPhiModel import (
    PfsThetaPhiModel, ThetaPhiModelFlag, evaluateThetaPhiBasis, makeThetaPhiBasis
)


class PfsThetaPhiModelTestCase(lsst.utils.tests.TestCase):
    """Test for PfsThetaPhiModel"""
    def setUp(self):
        self.rng = np.random.RandomState(12345)
        self.fiberId = np.arange(2, 42, dtype=int)
        self.numFibers = self.fiberId.size
        self.basis = makeThetaPhiBasis(3, 2)
        self.coeffs = self.rng.normal(0, 0.005, size=(self.numFibers, len(self.basis)))
        self.coeffs[5] = np.nan                       # a fiber with no model
        self.homeTheta = 360 + self.rng.uniform(5, 40, self.numFibers)
        self.homePhi = np.zeros(self.numFibers)
        self.flags = np.zeros(self.numFibers, dtype=int)
        self.flags[5] = ThetaPhiModelFlag.NO_MODEL
        self.flags[9] = ThetaPhiModelFlag.BROKEN_COBRA
        self.runs = ["Run 28 (2026-05)", "Run 30 (2026-09)"]
        self.runStart = ["2026-05-06", "2026-09-01"]
        self.homeOffsets = self.rng.normal(0, 0.006, size=(self.numFibers, len(self.runs)))
        self.alpha = dict(b=0.95, r=0.93, m=0.91, n=0.85)
        self.alphaSample = dict(r="Runs 28-31, scanned positions", b="Runs 28-31, scanned positions")
        self.metadata = dict(ORDER=3, PURE_HARMONICS=2, THETA_RAMP=False, BAND_LO=707.0, BAND_HI=790.0,
                             CALIB_DATE="2026-11-20", FIT_FORM="ln")

    def makeModel(self, **kwargs):
        args = dict(
            fiberId=self.fiberId, basis=self.basis, coeffs=self.coeffs, homeTheta=self.homeTheta,
            homePhi=self.homePhi, flags=self.flags, numPoints=np.full(self.numFibers, 300),
            phiMin=np.zeros(self.numFibers), phiMax=np.full(self.numFibers, 180.0),
            thetaCoverage=np.ones(self.numFibers), rms=np.full(self.numFibers, 0.002),
            modelError=np.zeros(self.numFibers), runs=self.runs, runStart=self.runStart,
            homeOffsets=self.homeOffsets, alpha=self.alpha, alphaSample=self.alphaSample,
            metadata=self.metadata,
        )
        args.update(kwargs)
        return PfsThetaPhiModel(**args)

    def direct(self, index, theta, phi):
        """s from the formula in datamodel.txt, written out independently"""
        total = 0.0
        for (it, tf, ip, pf), cc in zip(self.basis, self.coeffs[index]):
            def func(name, harmonic, angle):
                angle = np.deg2rad(angle)
                return dict(cos=np.cos(harmonic*angle), sin=np.sin(harmonic*angle),
                            one=np.ones_like(angle), lin=harmonic*angle/(2*np.pi))[name]
            total += cc*(func(tf, it, theta)*func(pf, ip, phi)
                         - func(tf, it, self.homeTheta[index])*func(pf, ip, self.homePhi[index]))
        return total

    def testBasis(self):
        """The standard basis has the expected size and terms"""
        self.assertEqual(len(makeThetaPhiBasis(3, 2)), 20)
        self.assertEqual(len(makeThetaPhiBasis(2, 1)), 8)
        self.assertEqual(makeThetaPhiBasis(1, 1, thetaRamp=True)[-1], (1, "lin", 0, "one"))
        values = evaluateThetaPhiBasis(self.basis, [10.0, 20.0], 30.0)
        self.assertEqual(values.shape, (2, len(self.basis)))

    def testEvaluate(self):
        """evaluate agrees with the formula, is zero at home, and NaN where there's no model"""
        model = self.makeModel()
        theta = self.rng.uniform(0, 400, 50)
        phi = self.rng.uniform(0, 180, 50)
        for ii in (0, 3, 17):
            got = model.evaluate(self.fiberId[ii], theta, phi)
            self.assertFloatsAlmostEqual(got, self.direct(ii, theta, phi), atol=1e-15, rtol=0)
            atHome = model.evaluate(self.fiberId[ii], self.homeTheta[ii], self.homePhi[ii])
            self.assertFloatsAlmostEqual(atHome, 0.0, atol=1e-15, rtol=0)
        self.assertTrue(np.all(np.isnan(model.evaluate(self.fiberId[5], theta, phi))))
        self.assertTrue(np.isnan(model.evaluate(12345, 100.0, 90.0)))
        # Vectorised over fibers too
        fibers = self.fiberId[[0, 3, 17]]
        got = model.evaluate(fibers, theta[:3], phi[:3])
        self.assertFloatsAlmostEqual(got, [self.direct(ii, tt, pp) for ii, tt, pp in
                                           zip([0, 3, 17], theta[:3], phi[:3])], atol=1e-15, rtol=0)

    def testHomeAngles(self):
        """getHomeAngles returns the home angles, NaN for unknown fibers"""
        model = self.makeModel()
        theta, phi = model.getHomeAngles([self.fiberId[3], 12345])
        self.assertEqual(theta[0], self.homeTheta[3])
        self.assertEqual(phi[0], self.homePhi[3])
        self.assertTrue(np.isnan(theta[1]) and np.isnan(phi[1]))

    def testRamp(self):
        """A theta ramp tells theta from theta + 360"""
        basis = makeThetaPhiBasis(1, 1, thetaRamp=True)
        coeffs = np.zeros((1, len(basis)))
        coeffs[0, -1] = 0.004
        model = PfsThetaPhiModel(
            [5], basis, coeffs, [381.6], [0.0], [0], [100], [0.0], [180.0], [1.0], [0.002], [0.0],
            [], [], np.zeros((1, 0)), dict(r=1.0), {}, dict(ORDER=1, PURE_HARMONICS=1, THETA_RAMP=True),
        )
        got = model.evaluate(5, [381.6, 21.6, 200.0], 0.0)
        expected = 0.004*(np.array([381.6, 21.6, 200.0]) - 381.6)/360
        self.assertFloatsAlmostEqual(got, expected, atol=1e-15, rtol=0)

    def testCorrection(self):
        """The correction is (1 + alpha s)/(1 + delta), and 1 for cobraHome"""
        model = self.makeModel()
        self.assertEqual(model.getRun("2026-05-01"), None)
        self.assertEqual(model.getRun("2026-05-06"), "Run 28 (2026-05)")
        self.assertEqual(model.getRun("2026-11-19"), "Run 30 (2026-09)")   # Run 31 uses Run 30's delta
        theta = self.rng.uniform(0, 400, self.numFibers)
        phi = self.rng.uniform(0, 180, self.numFibers)
        isHome = np.zeros(self.numFibers, dtype=bool)
        isHome[:4] = True
        got = model.getCorrection(self.fiberId, theta, phi, "r", "2026-09-03", isHome)
        expected = (1 + 0.93*model.evaluate(self.fiberId, theta, phi))/(1 + self.homeOffsets[:, 1])
        good = ~isHome & model.isGood(self.fiberId)
        self.assertFloatsAlmostEqual(got[good], expected[good], atol=1e-15, rtol=0)
        self.assertFloatsEqual(got[isHome], 1.0)
        self.assertTrue(np.isnan(got[5]))                                  # no model
        self.assertTrue(np.isfinite(got[9]))                               # BROKEN_COBRA is advisory
        self.assertFalse(model.isGood(self.fiberId[9], ThetaPhiModelFlag.BROKEN_COBRA))
        with self.assertRaises(KeyError):
            model.getCorrection(self.fiberId, theta, phi, "x", "2026-09-03", isHome)
        before = model.getCorrection(self.fiberId, theta, phi, "r", "2026-01-01", isHome)
        self.assertTrue(np.all(np.isnan(before[~isHome])))                 # no delta before the first run
        self.assertFloatsEqual(before[isHome], 1.0)

    def testIO(self):
        """Write and read back"""
        model = self.makeModel()
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            model.writeFits(filename)
            copy = PfsThetaPhiModel.readFits(filename)
        for name in ("fiberId", "coeffs", "homeTheta", "homePhi", "flags", "numPoints", "phiMin", "phiMax",
                     "thetaCoverage", "rms", "modelError", "homeOffsets"):
            np.testing.assert_array_equal(getattr(copy, name), getattr(model, name), err_msg=name)
        self.assertEqual(copy.basis, model.basis)
        self.assertEqual(copy.runs, model.runs)
        self.assertEqual(copy.runStart, model.runStart)
        self.assertEqual(copy.alpha, model.alpha)
        self.assertEqual(copy.alphaSample["r"], self.alphaSample["r"])
        self.assertEqual(copy.metadata["CALIB_DATE"], "2026-11-20")
        self.assertEqual(copy.hash, model.hash)
        changed = self.makeModel(coeffs=self.coeffs*1.001)
        self.assertNotEqual(changed.hash, model.hash)
        theta = self.rng.uniform(0, 400, 20)
        self.assertFloatsEqual(copy.evaluate(self.fiberId[3], theta, 45.0),
                               model.evaluate(self.fiberId[3], theta, 45.0))

    def testBadBasis(self):
        """A BASIS table that disagrees with ORDER is refused"""
        model = self.makeModel(metadata=dict(self.metadata, ORDER=2))
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            model.writeFits(filename)
            with self.assertRaises(RuntimeError):
                PfsThetaPhiModel.readFits(filename)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    setup_module(sys.modules["__main__"])
    unittest.main(failfast=True)
