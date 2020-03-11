import os
import sys
import unittest

import numpy as np

import lsst.utils.tests
import lsst.geom

from pfs.datamodel.pfsConfig import PfsConfig, TargetType, FiberStatus

display = None


class PfsConfigTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.numFibers = 2396

        # TargetType
        self.numSky = 240
        self.numFluxStd = 300
        self.numUnassigned = 10
        self.numEngineering = 10
        self.numObject = self.numFibers - (self.numSky + self.numFluxStd +
                                           self.numUnassigned + self.numEngineering)

        # FiberStatus
        self.numBroken = 3
        self.numBlocked = 2
        self.numBlackSpot = 1
        self.numUnilluminated = 3
        self.numGood = self.numFibers - (self.numBroken + self.numBlocked +
                                         self.numBlackSpot + self.numUnilluminated)

        self.raBoresight = 60.0*lsst.geom.degrees
        self.decBoresight = 30.0*lsst.geom.degrees
        self.fov = 1.5*lsst.geom.degrees
        self.pfiScale = 800000.0/self.fov.asDegrees()  # microns/degree
        self.pfiErrors = 10  # microns

        self.pfsDesignId = 12345
        self.visit0 = 67890
        self.fiberId = np.array(list(reversed(range(self.numFibers))))
        rng = np.random.RandomState(12345)
        self.tract = rng.uniform(high=30000, size=self.numFibers).astype(int)
        self.patch = ["%d,%d" % tuple(xy.tolist()) for
                      xy in rng.uniform(high=15, size=(self.numFibers, 2)).astype(int)]

        boresight = lsst.geom.SpherePoint(self.raBoresight, self.decBoresight)
        radius = np.sqrt(rng.uniform(size=self.numFibers))*0.5*self.fov.asDegrees()  # degrees
        theta = rng.uniform(size=self.numFibers)*2*np.pi  # radians
        coords = [boresight.offset(tt*lsst.geom.radians, rr*lsst.geom.degrees) for
                  rr, tt in zip(radius, theta)]
        self.ra = np.array([cc.getRa().asDegrees() for cc in coords])
        self.dec = np.array([cc.getDec().asDegrees() for cc in coords])
        self.pfiNominal = (self.pfiScale*np.array([(rr*np.cos(tt), rr*np.sin(tt)) for
                                                   rr, tt in zip(radius, theta)])).astype(np.float32)
        self.pfiCenter = (self.pfiNominal +
                          rng.normal(scale=self.pfiErrors, size=(self.numFibers, 2))).astype(np.float32)

        self.catId = rng.uniform(high=23, size=self.numFibers).astype(int)
        self.objId = rng.uniform(high=2**63, size=self.numFibers).astype(int)

        self.targetType = np.array([int(TargetType.SKY)]*self.numSky +
                                   [int(TargetType.FLUXSTD)]*self.numFluxStd +
                                   [int(TargetType.SCIENCE)]*self.numObject +
                                   [int(TargetType.UNASSIGNED)]*self.numUnassigned +
                                   [int(TargetType.ENGINEERING)]*self.numEngineering)
        rng.shuffle(self.targetType)
        self.fiberStatus = np.array([int(FiberStatus.BROKENFIBER)]*self.numBroken +
                                    [int(FiberStatus.BLOCKED)]*self.numBlocked +
                                    [int(FiberStatus.BLACKSPOT)]*self.numBlackSpot +
                                    [int(FiberStatus.UNILLUMINATED)]*self.numUnilluminated +
                                    [int(FiberStatus.GOOD)]*self.numGood)
        rng.shuffle(self.fiberStatus)

        self.fiberMag = [np.array([22.0, 23.5, 25.0, 26.0] if
                                  tt in (TargetType.SCIENCE, TargetType.FLUXSTD) else [])
                         for tt in self.targetType]
        self.filterNames = [["g", "i", "y", "H"] if tt in (TargetType.SCIENCE, TargetType.FLUXSTD) else []
                            for tt in self.targetType]

    def assertPfsConfig(self, lhs, rhs):
        for value in ("pfsDesignId", "visit0"):
            self.assertEqual(getattr(lhs, value), getattr(rhs, value), value)
        for value in ("raBoresight", "decBoresight"):
            # Our FITS header writer can introduce some tiny roundoff error
            self.assertAlmostEqual(getattr(lhs, value), getattr(rhs, value), 14, value)
        for value in ("fiberId", "tract", "ra", "dec", "catId", "objId",
                      "pfiCenter", "pfiNominal", "targetType", "fiberStatus"):
            np.testing.assert_array_equal(getattr(lhs, value), getattr(rhs, value), value)
        self.assertEqual(len(lhs.patch), len(rhs.patch))
        self.assertEqual(len(lhs.fiberMag), len(rhs.fiberMag))
        self.assertEqual(len(lhs.filterNames), len(rhs.filterNames))
        for ii in range(len(lhs)):
            self.assertEqual(lhs.patch[ii], rhs.patch[ii], "patch[%d]" % (ii,))
            np.testing.assert_array_equal(lhs.fiberMag[ii], rhs.fiberMag[ii], "fiberMag[%d]" % (ii,))
            self.assertListEqual(lhs.filterNames[ii], rhs.filterNames[ii], "filterNames[%d]" % (ii,))

    def testBasic(self):
        """Test basic operation of PfsConfig"""
        config = PfsConfig(self.pfsDesignId, self.visit0, self.raBoresight.asDegrees(),
                           self.decBoresight.asDegrees(), self.fiberId, self.tract, self.patch,
                           self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                           self.fiberMag, self.filterNames, self.pfiCenter, self.pfiNominal)

        dirName = os.path.splitext(__file__)[0]
        if not os.path.exists(dirName):
            os.makedirs(dirName)

        filename = os.path.join(dirName, config.filename)
        if os.path.exists(filename):
            os.unlink(filename)

        try:
            config.write(dirName=dirName)
            other = PfsConfig.read(self.pfsDesignId, self.visit0, dirName=dirName)
            self.assertPfsConfig(config, other)
        except Exception:
            raise  # Leave file for manual inspection
        else:
            os.unlink(filename)

    def testBadCtor(self):
        """Test bad constructor calls"""
        def extendArray(array):
            """Double the length of the array"""
            return np.concatenate((array, array))

        def extendList(values):
            """Double the length of the list"""
            return values + values

        raBoresight = self.raBoresight.asDegrees()
        decBoresight = self.decBoresight.asDegrees()

        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      extendArray(self.fiberId), self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      self.fiberMag, self.filterNames, self.pfiCenter, self.pfiNominal)
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.fiberId, extendArray(self.tract), self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      self.fiberMag, self.filterNames, self.pfiCenter, self.pfiNominal)
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.fiberId, self.tract, self.patch,
                      extendArray(self.ra), self.dec, self.catId, self.objId, self.targetType,
                      self.fiberStatus, self.fiberMag, self.filterNames, self.pfiCenter, self.pfiNominal)
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.fiberId, self.tract, self.patch,
                      self.ra, extendArray(self.dec), self.catId, self.objId, self.targetType,
                      self.fiberStatus, self.fiberMag, self.filterNames, self.pfiCenter, self.pfiNominal)
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, extendArray(self.catId), self.objId, self.targetType,
                      self.fiberStatus, self.fiberMag, self.filterNames, self.pfiCenter, self.pfiNominal)
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, extendArray(self.objId), self.targetType,
                      self.fiberStatus, self.fiberMag, self.filterNames, self.pfiCenter, self.pfiNominal)
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, extendArray(self.targetType),
                      self.fiberStatus, self.fiberMag, self.filterNames, self.pfiCenter, self.pfiNominal)
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      self.fiberMag, self.filterNames, extendArray(self.pfiCenter), self.pfiNominal)
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      self.fiberMag, self.filterNames, self.pfiCenter, extendArray(self.pfiNominal))
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.fiberId, self.tract, extendList(self.patch),
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      self.fiberMag, self.filterNames, self.pfiCenter, self.pfiNominal)
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      extendList(self.fiberMag), self.filterNames, self.pfiCenter, self.pfiNominal)
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      self.fiberMag, extendList(self.filterNames), self.pfiCenter, self.pfiNominal)

        targetType = self.targetType.copy()
        targetType[self.numFibers//2] = -1
        with self.assertRaises(ValueError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, targetType, self.fiberStatus,
                      self.fiberMag, self.filterNames, self.pfiCenter, self.pfiNominal)

        fiberStatus = self.fiberStatus.copy()
        fiberStatus[self.numFibers//2] = -1
        with self.assertRaises(ValueError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, fiberStatus,
                      self.fiberMag, self.filterNames, self.pfiCenter, self.pfiNominal)

        fiberMag = [extendArray(mag) if ii == self.numFibers//2 else mag
                    for ii, mag in enumerate(self.fiberMag)]
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      fiberMag, self.filterNames, self.pfiCenter, self.pfiNominal)

        filterNames = [extendList(ff) if ii == self.numFibers//5 else ff
                       for ii, ff in enumerate(self.filterNames)]
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      self.fiberMag, filterNames, self.pfiCenter, self.pfiNominal)

        pfiCenter = np.concatenate((self.pfiCenter, self.pfiCenter), axis=1)
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      self.fiberMag, self.filterNames, pfiCenter, self.pfiNominal)

        pfiNominal = np.concatenate((self.pfiNominal, self.pfiNominal), axis=1)
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      self.fiberMag, self.filterNames, self.pfiCenter, pfiNominal)


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
