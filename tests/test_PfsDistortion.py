import io
import os
import tempfile
import unittest

import astropy.io.fits
import numpy as np

from pfs.datamodel.pfsDetectorMap import (
    Box,
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
            PolynomialDistortion(2, self.box, rr(size=6), rr(size=6)),
            DoubleDistortion(2, self.box, rr(size=6), rr(size=6), rr(size=6), rr(size=6)),
            MosaicPolynomialDistortion(2, self.box, rr(size=18)),
            RotScaleDistortion(self.box, rr(size=6)),
            DoubleRotScaleDistortion(self.box, rr(size=10)),
        ]

    def assertDistortionsEqual(self, left, right):
        """Assert that two distortions are equal"""
        self.assertIs(type(left), type(right))
        self.assertEqual(vars(left).keys(), vars(right).keys())
        for name, value in vars(left).items():
            other = vars(right)[name]
            if isinstance(value, Box):
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
