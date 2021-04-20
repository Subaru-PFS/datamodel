import numpy as np

import lsst.utils.tests

from pfs.datamodel import GuideStars


class GuideStarsTestCase(lsst.utils.tests.TestCase):
    def setUp(self):

        self.length = 10

        rng = np.random.RandomState(123)

        self.objId = rng.uniform(size=self.length).astype(np.int64)
        self.epoch = np.array([rng.choice(['J2015.5', 'J2000.0', 'B1950.0']) for i in range(self.length)])
        self.ra = rng.uniform(size=self.length).astype(np.float64)
        self.dec = rng.uniform(size=self.length).astype(np.float64)
        self.pmRa = rng.uniform(size=self.length).astype(np.float32)
        self.pmDec = rng.uniform(size=self.length).astype(np.float32)
        self.parallax = rng.uniform(size=self.length).astype(np.float32)
        self.magnitude = rng.uniform(size=self.length).astype(np.float32)
        self.passband = np.array([rng.choice(['g', 'r', 'i', 'z', 'y']) for i in range(self.length)])
        self.color = rng.uniform(size=self.length).astype(np.float32)
        self.agId = rng.uniform(size=self.length).astype(np.int64)
        self.agX = rng.uniform(size=self.length).astype(np.float32)
        self.agY = rng.uniform(size=self.length).astype(np.float32)
        self.telElev = 60.0
        self.guideStarCatId = 1

        self.guideStars = GuideStars(self.objId, self.epoch, self.ra, self.dec,
                                     self.pmRa, self.pmDec,
                                     self.parallax,
                                     self.magnitude, self.passband,
                                     self.color,
                                     self.agId, self.agX, self.agY,
                                     self.telElev, self.guideStarCatId)

    def assertGuideStars(self, gs):
        self.assertEqual(len(gs), self.length)
        self.assertEqual(gs.objId.tolist(), self.objId.tolist())
        self.assertEqual(gs.epoch.tolist(), self.epoch.tolist())
        self.assertFloatsEqual(gs.ra, self.ra)
        self.assertFloatsEqual(gs.dec, self.dec)
        self.assertFloatsEqual(gs.pmRa, self.pmRa)
        self.assertFloatsEqual(gs.pmDec, self.pmDec)
        self.assertFloatsAlmostEqual(gs.parallax, self.parallax, rtol=3.4e-8, atol=3.4e-8)
        self.assertFloatsAlmostEqual(gs.magnitude, self.magnitude, rtol=3.4e-8, atol=3.4e-8)
        self.assertEqual(gs.passband.tolist(), self.passband.tolist())
        self.assertEqual(gs.agId.tolist(), self.agId.tolist())
        self.assertFloatsAlmostEqual(gs.agX, self.agX, rtol=3.4e-8, atol=3.4e-8)
        self.assertFloatsAlmostEqual(gs.agY, self.agY, rtol=3.4e-8, atol=3.4e-8)
        self.assertFloatsAlmostEqual(gs.telElev, self.telElev, rtol=3.4e-8, atol=3.4e-8)
        self.assertEqual(gs.guideStarCatId, self.guideStarCatId)

    def testBasic(self):
        """Test basic functions"""
        self.assertGuideStars(self.guideStars)

    def testBadLengths(self):
        with self.assertRaises(RuntimeError):
            GuideStars(np.array([123, 234], dtype=np.int64),
                       np.array(['J2015.5'], dtype='a7'),
                       np.array([1.0], dtype=np.float64),
                       np.array([1.0], dtype=np.float64),
                       np.array([1.0], dtype=np.float32),
                       np.array([1.0], dtype=np.float32),
                       np.array([1.0], dtype=np.float32),
                       np.array([1.0], dtype=np.float32),
                       np.array(['g'], dtype='a1'),
                       np.array([1.0], dtype=np.float32),
                       np.array([456], dtype=np.int32),
                       np.array([1.0], dtype=np.float32),
                       np.array([1.0], dtype=np.float32),
                       0.0,
                       0.0)

        with self.assertRaises(RuntimeError):
            GuideStars(np.array([123, 234], dtype=np.int64),
                       np.array(['J2015.5', 'B1950.0'], dtype='a7'),
                       np.array([1.0, 2.0], dtype=np.float64),
                       np.array([1.0], dtype=np.float64),
                       np.array([1.0, 2.0], dtype=np.float32),
                       np.array([1.0], dtype=np.float32),
                       np.array([1.0], dtype=np.float32),
                       np.array([1.0, 3.0], dtype=np.float32),
                       np.array(['g'], dtype='a1'),
                       np.array([1.0, 5.0], dtype=np.float32),
                       np.array([456, 123], dtype=np.int32),
                       np.array([1.0, 10.0], dtype=np.float32),
                       np.array([1.0, 51.1], dtype=np.float32),
                       0.0,
                       0.0)

    def testGoodLengths(self):
        GuideStars(np.array([123, 234], dtype=np.int64),
                   np.array(['J2015.5', 'B1950.0'], dtype='a7'),
                   np.array([1.0, 2.0], dtype=np.float64),
                   np.array([1.0, 3.0], dtype=np.float64),
                   np.array([1.0, 4.0], dtype=np.float32),
                   np.array([1.0, 10], dtype=np.float32),
                   np.array([1.0, 15], dtype=np.float32),
                   np.array([1.0, 10], dtype=np.float32),
                   np.array(['g', 'r'], dtype='a1'),
                   np.array([1.0, 1.0], dtype=np.float32),
                   np.array([456, 123], dtype=np.int32),
                   np.array([1.0, 100.0], dtype=np.float32),
                   np.array([1.0, 5.0], dtype=np.float32),
                   0.0,
                   0.0)

        GuideStars(np.array([123, 234, 345], dtype=np.int64),
                   np.array(['J2015.5', 'B1950.0', 'J2000.0'], dtype='a7'),
                   np.array([1.0, 2.0, 3.0], dtype=np.float64),
                   np.array([1.0, 34.1, 55.2], dtype=np.float64),
                   np.array([1.0, 2.0, 100.0], dtype=np.float32),
                   np.array([1.0, 13.0, 20.0], dtype=np.float32),
                   np.array([1.0, 59.4, 5.0], dtype=np.float32),
                   np.array([1.0, 3.0, 2.0], dtype=np.float32),
                   np.array(['g', 'r', 'i'], dtype='a1'),
                   np.array([1.0, 5.0, 15.0], dtype=np.float32),
                   np.array([456, 123, 23345], dtype=np.int32),
                   np.array([1.0, 10.0, 2], dtype=np.float32),
                   np.array([1.0, 51.1, 39.0], dtype=np.float32),
                   0.0,
                   0.0)

    def testFits(self):
        """Test I/O with FITS"""
        from astropy.io.fits import HDUList
        fits = HDUList()
        self.guideStars.toFits(fits)
        gs = GuideStars.fromFits(fits)
        print(f'self.epoch={self.guideStars.epoch}')
        print(f'gs.epoch={gs.epoch}')
        self.assertGuideStars(gs)


def setup_module(module):
    lsst.utils.tests.init()
