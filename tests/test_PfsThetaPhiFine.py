import sys
import unittest

import numpy as np

import lsst.utils.tests

from pfs.datamodel.pfsThetaPhiFine import PfsThetaPhiFine, ThetaPhiFineFlag, ThetaPhiFineQa
from pfs.datamodel.pfsThetaPhiModel import PfsThetaPhiModel, makeThetaPhiBasis


class PfsThetaPhiFineTestCase(lsst.utils.tests.TestCase):
    """Test for PfsThetaPhiFine"""
    def setUp(self):
        self.rng = np.random.RandomState(12345)
        self.fiberId = np.array([3, 17, 42, 99])
        self.numFibers = len(self.fiberId)
        self.numPoints = np.array([60, 80, 0, 50])           # fiber 42 has no points
        self.pointStart = np.r_[0, np.cumsum(self.numPoints)[:-1]]
        total = self.numPoints.sum()
        self.theta = self.rng.uniform(0, 400, total)
        self.phi = self.rng.uniform(0, 180, total)
        self.amplitude = np.tile([0.0015, 0.0015], (self.numFibers, 1))
        self.lTheta = np.tile([2.1, 21.0], (self.numFibers, 1))
        self.lPhi = np.tile([2.2, 15.0], (self.numFibers, 1))
        self.flags = np.zeros(self.numFibers, dtype=int)
        self.flags[2] = ThetaPhiFineFlag.NO_LAYER
        # Weights from a GP fitted to synthetic residuals, computed here independently of the class
        self.weight = np.zeros(total)
        for ii in range(self.numFibers):
            pts = slice(self.pointStart[ii], self.pointStart[ii] + self.numPoints[ii])
            if self.numPoints[ii] == 0:
                continue
            kernel = self.kernel(ii, self.theta[pts], self.phi[pts])
            y = self.rng.normal(0, 0.002, self.numPoints[ii])
            self.weight[pts] = np.linalg.solve(kernel + 0.002**2*np.eye(self.numPoints[ii]), y)
        self.errDistance = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
        self.errScale = np.array([0.4, 0.6, 0.8, 0.95, 1.0])
        self.metadata = dict(ARM="r", BAND_LO=771.0, BAND_HI=808.0, SOURCE="sky", RUNS="Run 28-31")

    def kernel(self, ii, theta, phi):
        """The kernel between (theta, phi) and fiber ii's points, as in datamodel.txt"""
        pts = slice(self.pointStart[ii], self.pointStart[ii] + self.numPoints[ii])
        dTheta = (theta[:, None] - self.theta[pts][None, :] + 180) % 360 - 180
        dPhi = phi[:, None] - self.phi[pts][None, :]
        result = 0.0
        for cc in range(2):
            rr = np.hypot(dTheta/self.lTheta[ii, cc], dPhi/self.lPhi[ii, cc])
            result = result + self.amplitude[ii, cc]**2*(1 + np.sqrt(3)*rr)*np.exp(-np.sqrt(3)*rr)
        return result

    def makeLayer(self, **kwargs):
        args = dict(
            fiberId=self.fiberId, pointStart=self.pointStart, numPoints=self.numPoints,
            amplitude=self.amplitude,
            lTheta=self.lTheta, lPhi=self.lPhi, flags=self.flags, repeatCorr=np.full(self.numFibers, 0.5),
            varRemoved=np.full(self.numFibers, 0.25), theta=self.theta, phi=self.phi, weight=self.weight,
            errDistance=self.errDistance, errScale=self.errScale, metadata=self.metadata,
        )
        args.update(kwargs)
        return PfsThetaPhiFine(**args)

    def testEvaluate(self):
        """evaluate agrees with the formula;  theta is periodic;  no layer gives 0"""
        layer = self.makeLayer()
        theta = self.rng.uniform(0, 400, 30)
        phi = self.rng.uniform(0, 180, 30)
        for ii in (0, 1, 3):
            pts = slice(self.pointStart[ii], self.pointStart[ii] + self.numPoints[ii])
            expected = self.kernel(ii, theta, phi) @ self.weight[pts]
            got = layer.evaluate(self.fiberId[ii], theta, phi)
            self.assertFloatsAlmostEqual(got, expected, atol=1e-15, rtol=0)
            self.assertFloatsAlmostEqual(layer.evaluate(self.fiberId[ii], theta + 360, phi), expected,
                                         atol=1e-12, rtol=0)
        self.assertFloatsEqual(layer.evaluate(self.fiberId[2], theta, phi), 0.0)    # NO_LAYER
        self.assertFloatsEqual(layer.evaluate(12345, theta, phi), 0.0)              # not present
        # Mixed fibers in one call
        fibers = self.fiberId[[0, 1, 3, 0]]
        got = layer.evaluate(fibers, theta[:4], phi[:4])
        for kk, fid in enumerate(fibers):
            self.assertFloatsAlmostEqual(got[kk], layer.evaluate(fid, theta[kk], phi[kk]), atol=1e-15, rtol=0)

    def testCorrection(self):
        """The correction is 1 + f, and 1 for cobraHome"""
        layer = self.makeLayer()
        theta = self.rng.uniform(0, 400, self.numFibers)
        phi = self.rng.uniform(0, 180, self.numFibers)
        isHome = np.array([False, True, False, False])
        got = layer.getCorrection(self.fiberId, theta, phi, isHome)
        expected = 1 + layer.evaluate(self.fiberId, theta, phi)
        self.assertFloatsAlmostEqual(got[~isHome], expected[~isHome], atol=1e-15, rtol=0)
        self.assertEqual(got[1], 1.0)
        self.assertEqual(got[2], 1.0)                                              # NO_LAYER

    def testError(self):
        """The error model, from the distance to the nearest training point"""
        layer = self.makeLayer()
        ii = 0
        total = np.sqrt(np.sum(self.amplitude[ii]**2))
        atPoint = layer.getError(self.fiberId[ii], self.theta[0], self.phi[0])
        self.assertFloatsAlmostEqual(atPoint, total*self.errScale[0], rtol=1e-12)
        far = layer.getError(self.fiberId[ii], self.theta[0] + 180, self.phi[0])
        self.assertLessEqual(far, total*self.errScale[-1] + 1e-15)
        self.assertTrue(np.isnan(layer.getError(self.fiberId[2], 10.0, 10.0)))

    def testIO(self):
        """Write and read back (positions and weights are float32)"""
        layer = self.makeLayer()
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            layer.writeFits(filename)
            copy = PfsThetaPhiFine.readFits(filename)
        for name in ("fiberId", "pointStart", "numPoints", "flags", "errDistance", "errScale"):
            np.testing.assert_array_equal(getattr(copy, name), getattr(layer, name), err_msg=name)
        for name in ("amplitude", "lTheta", "lPhi", "theta", "phi", "weight"):
            self.assertFloatsAlmostEqual(getattr(copy, name), getattr(layer, name), rtol=1e-6, atol=0)
        self.assertEqual(copy.arm, "r")
        self.assertEqual(copy.metadata["SOURCE"], "sky")
        theta = self.rng.uniform(0, 400, 20)
        phi = self.rng.uniform(0, 180, 20)
        f64 = layer.evaluate(self.fiberId[1], theta, phi)
        f32 = copy.evaluate(self.fiberId[1], theta, phi)
        self.assertLess(np.max(np.abs(f32 - f64)), 1e-6)                          # cf. layer ~ 1e-3

    def testSmoothModel(self):
        """The layer refuses a smooth model other than its own"""
        basis = makeThetaPhiBasis(1, 1)
        model = PfsThetaPhiModel(
            [3], basis, np.full((1, len(basis)), 0.001), [381.6], [0.0], [0], [100], [0.0], [180.0], [1.0],
            [0.002], [0.0], [], [], np.zeros((1, 0)), dict(r=0.93), {}, dict(ORDER=1, PURE_HARMONICS=1),
        )
        layer = self.makeLayer(metadata=dict(self.metadata, SMOOTH_ID=model.modelId))
        layer.checkSmoothModel(model)
        other = PfsThetaPhiModel(
            [3], basis, np.full((1, len(basis)), 0.002), [381.6], [0.0], [0], [100], [0.0], [180.0], [1.0],
            [0.002], [0.0], [], [], np.zeros((1, 0)), dict(r=0.93), {}, dict(ORDER=1, PURE_HARMONICS=1),
        )
        with self.assertRaises(RuntimeError):
            layer.checkSmoothModel(other)
        with self.assertRaises(RuntimeError):
            self.makeLayer().checkSmoothModel(model)                                 # no SMOOTH_ID

    def testQa(self):
        """The QA table round-trips"""
        num = 7
        qa = ThetaPhiFineQa(
            fiberId=np.full(num, 3, dtype=np.int32), visit=np.arange(num, dtype=np.int32),
            night=np.array(["2026-11-18"]*num), source=np.array(["sky"]*5 + ["quartz"]*2),
            theta=self.rng.uniform(0, 400, num).astype(np.float32),
            phi=self.rng.uniform(0, 180, num).astype(np.float32),
            y=self.rng.normal(0, 0.002, num).astype(np.float32), sigma=np.full(num, 0.002, dtype=np.float32),
            used=np.array([True]*6 + [False]), looPred=self.rng.normal(0, 0.001, num).astype(np.float32),
        )
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            qa.writeFits(filename)
            copy = ThetaPhiFineQa.readFits(filename)
        for name in ("fiberId", "visit", "night", "source", "used", "y"):
            np.testing.assert_array_equal(getattr(copy, name), getattr(qa, name), err_msg=name)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    setup_module(sys.modules["__main__"])
    unittest.main(failfast=True)
