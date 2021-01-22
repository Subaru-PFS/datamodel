import os
import sys
import unittest

import numpy as np
import astropy.units as u

import lsst.utils.tests
import lsst.geom

from pfs.datamodel.pfsConfig import PfsConfig, TargetType, FiberStatus, PfsDesign, GuideStars

display = None


class PfsConfigTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.numFibers = 2396

        # TargetType
        self.numSky = 240
        self.numFluxStd = 300
        self.numUnassigned = 10
        self.numEngineering = 10
        self.numSuNSS_Imaging = 5
        self.numSuNSS_Diffuse = 3
        self.numObject = self.numFibers - (self.numSky + self.numFluxStd +
                                           self.numUnassigned + self.numEngineering +
                                           self.numSuNSS_Imaging + self.numSuNSS_Diffuse)

        # FiberStatus
        self.numBroken = 3
        self.numBlocked = 2
        self.numBlackSpot = 1
        self.numUnilluminated = 3
        self.numGood = self.numFibers - (self.numBroken + self.numBlocked +
                                         self.numBlackSpot + self.numUnilluminated)

        self.raBoresight = 60.0*lsst.geom.degrees
        self.decBoresight = 30.0*lsst.geom.degrees
        self.posAng = 0.0
        self.arms = 'brn'
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
                                   [int(TargetType.ENGINEERING)]*self.numEngineering +
                                   [int(TargetType.SUNSS_DIFFUSE)]*self.numSuNSS_Diffuse +
                                   [int(TargetType.SUNSS_IMAGING)]*self.numSuNSS_Imaging)
        rng.shuffle(self.targetType)
        self.fiberStatus = np.array([int(FiberStatus.BROKENFIBER)]*self.numBroken +
                                    [int(FiberStatus.BLOCKED)]*self.numBlocked +
                                    [int(FiberStatus.BLACKSPOT)]*self.numBlackSpot +
                                    [int(FiberStatus.UNILLUMINATED)]*self.numUnilluminated +
                                    [int(FiberStatus.GOOD)]*self.numGood)
        rng.shuffle(self.fiberStatus)

        fiberMagnitude = [22.0, 23.5, 25.0, 26.0]
        fiberFluxes = [(f * u.ABmag).to_value(u.nJy) for f in fiberMagnitude]

        self.fiberFlux = [np.array(fiberFluxes if
                                   tt in (TargetType.SCIENCE, TargetType.FLUXSTD) else [])
                          for tt in self.targetType]

        # For these tests, assign psfFlux and totalFlux
        # the same value as the fiber flux
        self.psfFlux = [fFlux for fFlux in self.fiberFlux]
        self.totalFlux = [fFlux for fFlux in self.fiberFlux]

        # Assign corresponding errors as 1% of fiberFlux
        fluxError = [0.01 * f for f in fiberFluxes]
        self.fiberFluxErr = [np.array(fluxError if
                                      tt in (TargetType.SCIENCE, TargetType.FLUXSTD) else [])
                             for tt in self.targetType]
        self.psfFluxErr = [e for e in self.fiberFluxErr]
        self.totalFluxErr = [e for e in self.fiberFluxErr]

        self.filterNames = [["g", "i", "y", "H"] if tt in (TargetType.SCIENCE, TargetType.FLUXSTD) else []
                            for tt in self.targetType]

        self.guideStars = GuideStars.empty()

    def assertPfsConfig(self, lhs, rhs):
        for value in ("pfsDesignId", "visit0"):
            self.assertEqual(getattr(lhs, value), getattr(rhs, value), value)
        for value in ("raBoresight", "decBoresight", "posAng", "arms"):
            # Our FITS header writer can introduce some tiny roundoff error
            self.assertAlmostEqual(getattr(lhs, value), getattr(rhs, value), 14, value)
        for value in ("fiberId", "tract", "ra", "dec", "catId", "objId",
                      "pfiCenter", "pfiNominal", "targetType", "fiberStatus"):
            np.testing.assert_array_equal(getattr(lhs, value), getattr(rhs, value), value)
        self.assertEqual(len(lhs.patch), len(rhs.patch))
        self.assertEqual(len(lhs.fiberFlux), len(rhs.fiberFlux))
        self.assertEqual(len(lhs.filterNames), len(rhs.filterNames))
        for ii in range(len(lhs)):
            self.assertEqual(lhs.patch[ii], rhs.patch[ii], "patch[%d]" % (ii,))
            np.testing.assert_array_almost_equal(lhs.fiberFlux[ii], rhs.fiberFlux[ii],
                                                 decimal=4,
                                                 err_msg="fiberFlux[%d]" % (ii,))
            self.assertListEqual(lhs.filterNames[ii], rhs.filterNames[ii], "filterNames[%d]" % (ii,))

    def testBasic(self):
        """Test basic operation of PfsConfig"""
        config = PfsConfig(self.pfsDesignId, self.visit0, self.raBoresight.asDegrees(),
                           self.decBoresight.asDegrees(),
                           self.posAng,
                           self.arms,
                           self.fiberId, self.tract, self.patch,
                           self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                           self.fiberFlux,
                           self.psfFlux, self.totalFlux,
                           self.fiberFluxErr, self.psfFluxErr, self.totalFluxErr,
                           self.filterNames, self.pfiCenter, self.pfiNominal,
                           self.guideStars)

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
                      self.posAng,
                      self.arms,
                      extendArray(self.fiberId), self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      self.fiberFlux,
                      self.psfFlux, self.totalFlux, self.fiberFluxErr, self.psfFluxErr, self.totalFluxErr,
                      self.filterNames, self.pfiCenter, self.pfiNominal,
                      self.guideStars),
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.posAng,
                      self.arms,
                      self.fiberId, extendArray(self.tract), self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      self.fiberFlux,
                      self.psfFlux, self.totalFlux, self.fiberFluxErr, self.psfFluxErr, self.totalFluxErr,
                      self.filterNames, self.pfiCenter, self.pfiNominal,
                      self.guideStars)
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.posAng,
                      self.arms,
                      self.fiberId, self.tract, self.patch,
                      extendArray(self.ra), self.dec, self.catId, self.objId, self.targetType,
                      self.fiberStatus,
                      self.fiberFlux,
                      self.psfFlux, self.totalFlux, self.fiberFluxErr, self.psfFluxErr, self.totalFluxErr,
                      self.filterNames, self.pfiCenter, self.pfiNominal,
                      self.guideStars)
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.posAng,
                      self.arms,
                      self.fiberId, self.tract, self.patch,
                      self.ra, extendArray(self.dec), self.catId, self.objId, self.targetType,
                      self.fiberStatus,
                      self.fiberFlux,
                      self.psfFlux, self.totalFlux, self.fiberFluxErr, self.psfFluxErr, self.totalFluxErr,
                      self.filterNames, self.pfiCenter, self.pfiNominal,
                      self.guideStars)
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.posAng,
                      self.arms,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, extendArray(self.catId), self.objId, self.targetType,
                      self.fiberStatus,
                      self.fiberFlux,
                      self.psfFlux, self.totalFlux, self.fiberFluxErr, self.psfFluxErr, self.totalFluxErr,
                      self.filterNames, self.pfiCenter, self.pfiNominal,
                      self.guideStars)
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.posAng,
                      self.arms,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, extendArray(self.objId), self.targetType,
                      self.fiberStatus,
                      self.fiberFlux,
                      self.psfFlux, self.totalFlux, self.fiberFluxErr, self.psfFluxErr, self.totalFluxErr,
                      self.filterNames, self.pfiCenter, self.pfiNominal,
                      self.guideStars)
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.posAng,
                      self.arms,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, extendArray(self.targetType),
                      self.fiberStatus,
                      self.fiberFlux,
                      self.psfFlux, self.totalFlux, self.fiberFluxErr, self.psfFluxErr, self.totalFluxErr,
                      self.filterNames, self.pfiCenter, self.pfiNominal,
                      self.guideStars)
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.posAng,
                      self.arms,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      self.fiberFlux,
                      self.psfFlux, self.totalFlux, self.fiberFluxErr, self.psfFluxErr, self.totalFluxErr,
                      self.filterNames, extendArray(self.pfiCenter), self.pfiNominal,
                      self.guideStars)
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.posAng,
                      self.arms,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      self.fiberFlux,
                      self.psfFlux, self.totalFlux, self.fiberFluxErr, self.psfFluxErr, self.totalFluxErr,
                      self.filterNames, self.pfiCenter, extendArray(self.pfiNominal),
                      self.guideStars)
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.posAng,
                      self.arms,
                      self.fiberId, self.tract, extendList(self.patch),
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      self.fiberFlux,
                      self.psfFlux, self.totalFlux, self.fiberFluxErr, self.psfFluxErr, self.totalFluxErr,
                      self.filterNames, self.pfiCenter, self.pfiNominal,
                      self.guideStars)
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.posAng,
                      self.arms,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      extendList(self.fiberFlux),
                      self.psfFlux, self.totalFlux, self.fiberFluxErr, self.psfFluxErr, self.totalFluxErr,
                      self.filterNames, self.pfiCenter, self.pfiNominal,
                      self.guideStars)
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.posAng,
                      self.arms,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      self.fiberFlux,
                      extendList(self.psfFlux), self.totalFlux,
                      self.fiberFluxErr, self.psfFluxErr, self.totalFluxErr,
                      self.filterNames, self.pfiCenter, self.pfiNominal,
                      self.guideStars)
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.posAng,
                      self.arms,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      self.fiberFlux,
                      self.psfFlux, extendList(self.totalFlux),
                      self.fiberFluxErr, self.psfFluxErr, self.totalFluxErr,
                      self.filterNames, self.pfiCenter, self.pfiNominal,
                      self.guideStars)
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.posAng,
                      self.arms,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      self.fiberFlux,
                      self.psfFlux, self.totalFlux,
                      extendList(self.fiberFluxErr), self.psfFluxErr, self.totalFluxErr,
                      self.filterNames, self.pfiCenter, self.pfiNominal,
                      self.guideStars)
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.posAng,
                      self.arms,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      self.fiberFlux,
                      self.psfFlux, self.totalFlux,
                      self.fiberFluxErr, extendList(self.psfFluxErr), self.totalFluxErr,
                      self.filterNames, self.pfiCenter, self.pfiNominal,
                      self.guideStars)
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.posAng,
                      self.arms,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      self.fiberFlux,
                      self.psfFlux, self.totalFlux,
                      self.fiberFluxErr, self.psfFluxErr, extendList(self.totalFluxErr),
                      self.filterNames, self.pfiCenter, self.pfiNominal,
                      self.guideStars)
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.posAng,
                      self.arms,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      self.fiberFlux,
                      self.psfFlux, self.totalFlux,
                      self.fiberFluxErr, self.psfFluxErr, self.totalFluxErr,
                      extendList(self.filterNames), self.pfiCenter, self.pfiNominal,
                      self.guideStars)

        targetType = self.targetType.copy()
        targetType[self.numFibers//2] = -1
        with self.assertRaises(ValueError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.posAng,
                      self.arms,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, targetType, self.fiberStatus,
                      self.fiberFlux,
                      self.psfFlux, self.totalFlux, self.fiberFluxErr, self.psfFluxErr, self.totalFluxErr,
                      self.filterNames, self.pfiCenter, self.pfiNominal,
                      self.guideStars)

        fiberStatus = self.fiberStatus.copy()
        fiberStatus[self.numFibers//2] = -1
        with self.assertRaises(ValueError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.posAng,
                      self.arms,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, fiberStatus,
                      self.fiberFlux,
                      self.psfFlux, self.totalFlux, self.fiberFluxErr, self.psfFluxErr, self.totalFluxErr,
                      self.filterNames, self.pfiCenter, self.pfiNominal,
                      self.guideStars)

        fiberFlux = [extendArray(mag) if ii == self.numFibers//2 else mag
                     for ii, mag in enumerate(self.fiberFlux)]
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.posAng,
                      self.arms,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      fiberFlux,
                      self.psfFlux, self.totalFlux, self.fiberFluxErr, self.psfFluxErr, self.totalFluxErr,
                      self.filterNames, self.pfiCenter, self.pfiNominal,
                      self.guideStars)

        psfFlux = [extendArray(pFlux) if ii == self.numFibers//2 else pFlux
                   for ii, pFlux in enumerate(self.psfFlux)]
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.posAng,
                      self.arms,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      self.fiberFlux,
                      psfFlux, self.totalFlux, self.fiberFluxErr, self.psfFluxErr, self.totalFluxErr,
                      self.filterNames, self.pfiCenter, self.pfiNominal,
                      self.guideStars)

        totalFlux = [extendArray(tFlux) if ii == self.numFibers//2 else tFlux
                     for ii, tFlux in enumerate(self.totalFlux)]
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.posAng,
                      self.arms,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      self.fiberFlux,
                      self.psfFlux, totalFlux, self.fiberFluxErr, self.psfFluxErr, self.totalFluxErr,
                      self.filterNames, self.pfiCenter, self.pfiNominal,
                      self.guideStars)

        fiberFluxErr = [extendArray(ffErr) if ii == self.numFibers//2 else ffErr
                        for ii, ffErr in enumerate(self.fiberFluxErr)]
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.posAng,
                      self.arms,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      self.fiberFlux,
                      self.psfFlux, self.totalFlux,
                      fiberFluxErr, self.psfFluxErr, self.totalFluxErr,
                      self.filterNames, self.pfiCenter, self.pfiNominal,
                      self.guideStars)

        psfFluxErr = [extendArray(pfErr) if ii == self.numFibers//2 else pfErr
                      for ii, pfErr in enumerate(self.psfFluxErr)]
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.posAng,
                      self.arms,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      self.fiberFlux,
                      self.psfFlux, self.totalFlux,
                      self.fiberFluxErr,
                      psfFluxErr, self.totalFluxErr,
                      self.filterNames, self.pfiCenter, self.pfiNominal,
                      self.guideStars)

        totalFluxErr = [extendArray(tfErr) if ii == self.numFibers//2 else tfErr
                        for ii, tfErr in enumerate(self.totalFluxErr)]
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.posAng,
                      self.arms,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      self.fiberFlux,
                      self.psfFlux, self.totalFlux,
                      self.fiberFluxErr,
                      self.psfFluxErr,
                      totalFluxErr,
                      self.filterNames, self.pfiCenter, self.pfiNominal,
                      self.guideStars)

        filterNames = [extendList(ff) if ii == self.numFibers//5 else ff
                       for ii, ff in enumerate(self.filterNames)]
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.posAng,
                      self.arms,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      self.fiberFlux,
                      self.psfFlux, self.totalFlux, self.fiberFluxErr, self.psfFluxErr, self.totalFluxErr,
                      filterNames, self.pfiCenter, self.pfiNominal,
                      self.guideStars)

        pfiCenter = np.concatenate((self.pfiCenter, self.pfiCenter), axis=1)
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.posAng,
                      self.arms,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      self.fiberFlux,
                      self.psfFlux, self.totalFlux, self.fiberFluxErr, self.psfFluxErr, self.totalFluxErr,
                      self.filterNames, pfiCenter, self.pfiNominal,
                      self.guideStars)

        pfiNominal = np.concatenate((self.pfiNominal, self.pfiNominal), axis=1)
        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.posAng,
                      self.arms,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      self.fiberFlux,
                      self.psfFlux, self.totalFlux, self.fiberFluxErr, self.psfFluxErr, self.totalFluxErr,
                      self.filterNames, self.pfiCenter, pfiNominal,
                      self.guideStars)

        with self.assertRaises(RuntimeError):
            PfsConfig(self.pfsDesignId, self.visit0, raBoresight, decBoresight,
                      self.posAng,
                      self.arms,
                      self.fiberId, self.tract, self.patch,
                      self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                      self.fiberFlux,
                      self.psfFlux, self.totalFlux, self.fiberFluxErr, self.psfFluxErr, self.totalFluxErr,
                      self.filterNames, self.pfiCenter, pfiNominal,
                      None)

    def testFromPfsDesign(self):
        """Test PfsConfig.fromPfsDesign"""
        design = PfsDesign(self.pfsDesignId, self.raBoresight.asDegrees(), self.decBoresight.asDegrees(),
                           self.posAng,
                           self.arms,
                           self.fiberId, self.tract, self.patch, self.ra, self.dec,
                           self.catId, self.objId, self.targetType, self.fiberStatus,
                           self.fiberFlux,
                           self.psfFlux, self.totalFlux,
                           self.fiberFluxErr, self.psfFluxErr, self.totalFluxErr,
                           self.filterNames, self.pfiNominal,
                           self.guideStars)
        config = PfsConfig(self.pfsDesignId, self.visit0, self.raBoresight.asDegrees(),
                           self.decBoresight.asDegrees(),
                           self.posAng,
                           self.arms,
                           self.fiberId, self.tract, self.patch,
                           self.ra, self.dec, self.catId, self.objId, self.targetType, self.fiberStatus,
                           self.fiberFlux,
                           self.psfFlux, self.totalFlux,
                           self.fiberFluxErr, self.psfFluxErr, self.totalFluxErr,
                           self.filterNames, self.pfiCenter, self.pfiNominal,
                           self.guideStars)
        self.assertPfsConfig(PfsConfig.fromPfsDesign(design, self.visit0, self.pfiCenter), config)


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
