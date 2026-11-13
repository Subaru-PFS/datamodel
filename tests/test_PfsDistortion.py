import io
import os
import tempfile
import unittest

import astropy.io.fits
import numpy as np

from pfs.datamodel.pfsDetectorMap import (
    Box,
    BSpline2D,
    BSplineDistortion,
    DoubleDistortion,
    DoubleRotScaleDistortion,
    MosaicPolynomialDistortion,
    PfsDistortion,
    PolynomialDistortion,
    RotScaleDistortion,
)


class PfsDistortionTestCase(unittest.TestCase):
    """Tests of writing and reading a single PfsDistortion"""

    def setUp(self):
        self.rng = np.random.default_rng(12345)
        self.box = Box(-1.3, -2.4, 3.5, 4.6)  # Non-integer, to check exact persistence of the ranges

    def makeDistortions(self):
        """Make an example of each distortion class"""
        rr = self.rng.uniform
        return [
            PolynomialDistortion(2, 3, self.box, rr(size=6), rr(size=10)),  # Different x,y orders
            DoubleDistortion(2, self.box, rr(size=6), rr(size=6), rr(size=6), rr(size=6)),
            MosaicPolynomialDistortion(2, 3, self.box, rr(size=22)),  # 6 affine + 6 x + 10 y
            RotScaleDistortion(self.box, rr(size=6)),
            DoubleRotScaleDistortion(self.box, rr(size=10)),
            BSplineDistortion(
                [2048.0],
                [self.makeSpline(), BSpline2D.empty()],
                [BSpline2D.empty(), self.makeSpline()],
            ),
        ]

    def makeSpline(self) -> BSpline2D:
        """Make a cubic tensor-product B-spline with random coefficients"""
        xKnots = np.concatenate([[0.0] * 4, [512.0, 1024.0], [2048.0] * 4])
        yKnots = np.concatenate([[0.0] * 4, [2000.0], [4000.0] * 4])
        return BSpline2D(3, xKnots, yKnots, self.rng.uniform(size=(len(xKnots) - 4, len(yKnots) - 4)))

    def assertDistortionsEqual(self, left, right):
        """Assert that two distortions (or their components) are equal"""
        self.assertIs(type(left), type(right))
        self.assertEqual(vars(left).keys(), vars(right).keys())
        for name, value in vars(left).items():
            other = vars(right)[name]
            if isinstance(value, list):
                self.assertEqual(len(value), len(other))
                for vv, oo in zip(value, other):
                    self.assertDistortionsEqual(vv, oo)
            elif isinstance(value, Box):
                self.assertEqual(
                    (value.xMin, value.yMin, value.xMax, value.yMax),
                    (other.xMin, other.yMin, other.xMax, other.yMax),
                )
            elif isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, other)
            else:
                self.assertEqual(value, other)

    def testRoundTrip(self):
        """Test writing and reading through a FITS file"""
        for distortion in self.makeDistortions():
            with self.subTest(distortion=distortion.getName()):
                buffer = io.BytesIO()
                distortion.writeFits(buffer)
                data = buffer.getvalue()
                self.assertEqual(astropy.io.fits.getheader(io.BytesIO(data))["DAMD_VER"], 1)
                copy = PfsDistortion.readFits(io.BytesIO(data))
                self.assertDistortionsEqual(distortion, copy)

                # Reading with the subclass gives the same result
                copy = type(distortion).readFits(io.BytesIO(data))
                self.assertDistortionsEqual(distortion, copy)

    def testFile(self):
        """Test writing and reading a file by name"""
        distortion = self.makeDistortions()[0]
        with tempfile.TemporaryDirectory() as tempDir:
            filename = os.path.join(tempDir, "distortion.fits")
            distortion.writeFits(filename)
            distortion.writeFits(filename)  # Overwriting is allowed
            copy = PfsDistortion.readFits(filename)
        self.assertDistortionsEqual(distortion, copy)

    def testOldOrderKeyword(self):
        """Test reading polynomial distortions written with a single ORDER"""
        for distortion in self.makeDistortions()[:3:2]:  # PolynomialDistortion, MosaicPolynomialDistortion
            with self.subTest(distortion=distortion.getName()):
                buffer = io.BytesIO()
                distortion.writeFits(buffer)
                with astropy.io.fits.open(io.BytesIO(buffer.getvalue())) as fits:
                    header = fits[distortion.getExtName(0)].header
                    self.assertIn("XORDER", header)
                    header["ORDER"] = 4
                    del header["XORDER"]
                    del header["YORDER"]
                    old = io.BytesIO()
                    fits.writeto(old)
                copy = PfsDistortion.readFits(io.BytesIO(old.getvalue()))
                self.assertEqual(copy.xOrder, 4)
                self.assertEqual(copy.yOrder, 4)
                self.assertEqual(copy.order, 4)

    def testOrder(self):
        """Test the backward-compatible ``order`` attribute"""
        rr = self.rng.uniform
        same = PolynomialDistortion(2, 2, self.box, rr(size=6), rr(size=6))
        self.assertEqual(same.order, 2)
        different = PolynomialDistortion(2, 3, self.box, rr(size=6), rr(size=10))
        with self.assertRaises(RuntimeError):
            different.order

    def testWrongClass(self):
        """Test that reading with the wrong subclass fails"""
        buffer = io.BytesIO()
        self.makeDistortions()[0].writeFits(buffer)  # PolynomialDistortion
        with self.assertRaises(RuntimeError):
            RotScaleDistortion.readFits(io.BytesIO(buffer.getvalue()))

    def testGetDistortion(self):
        """Test looking up distortion classes by name, including subclasses of subclasses"""
        for distortion in self.makeDistortions():
            self.assertIs(PfsDistortion.getDistortion(distortion.getName()), type(distortion))

        class Derived(PolynomialDistortion):
            pass

        self.assertIs(PfsDistortion.getDistortion("Derived"), Derived)
        with self.assertRaises(RuntimeError):
            PfsDistortion.getDistortion("NoSuchDistortion")


if __name__ == "__main__":
    unittest.main()
