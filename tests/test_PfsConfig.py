import os
import datetime
import astropy.io.fits as pyfits
import sys
import unittest

import numpy as np
import astropy.units as u

import lsst.utils.tests
import lsst.geom

from pfs.datamodel.pfsConfig import (
    PfsConfig, TargetType, FiberStatus, PfsDesign, GuideStars, InstrumentStatusFlag,
    InstrumentStatusDescription)

from pfs.datamodel.utils import convertToIso8601Utc
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

        self.raBoresight = 60.0  # degrees
        self.decBoresight = 30.0  # degrees
        self.posAng = 0.0  # degrees
        self.arms = 'brn'
        self.fov = 1.5*lsst.geom.degrees
        self.pfiScale = 800.0/self.fov.asDegrees()  # millimeters/degree
        self.pfiErrors = 0.01  # millimeters

        self.pfsDesignId = 12345
        self.visit = 67890
        self.fiberId = np.array(list(reversed(range(self.numFibers))))
        rng = np.random.RandomState(12345)
        self.tract = rng.uniform(high=30000, size=self.numFibers).astype(int)
        self.patch = ["%d,%d" % tuple(xy.tolist()) for
                      xy in rng.uniform(high=15, size=(self.numFibers, 2)).astype(int)]

        boresight = lsst.geom.SpherePoint(self.raBoresight*lsst.geom.degrees,
                                          self.decBoresight*lsst.geom.degrees)
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

        self.epoch = np.full(self.numFibers, "J2000.0")
        self.pmRa = rng.uniform(low=0, high=100, size=self.numFibers)  # mas/yr
        self.pmDec = rng.uniform(low=0, high=100, size=self.numFibers)  # mas/yr
        self.parallax = rng.uniform(low=1e-5, high=10, size=self.numFibers)  # mas

        self.proposalId = np.full(self.numFibers, "S24A-0001QN")
        self.obCode = np.array([f"obcode_{fibnum:04d}" for fibnum in range(self.numFibers)])

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

        self.designName = "Zenith odd-even"
        self.variant = 0
        self.designId0 = 0

        self.header = dict()
        self.camMask = 0
        self.instStatusFlag = 0

        self.obstime = convertToIso8601Utc(datetime.datetime.now(datetime.timezone.utc).isoformat())
        self.obstimeDesign = convertToIso8601Utc(datetime.datetime.now(datetime.timezone.utc).isoformat())
        self.pfsUtilsVer = self.pfsUtilsVerDesign = "w.2025.06"

        self.visit0 = 67889

    def _makeInstance(self, Class, **kwargs):
        """Construct a PfsDesign or PfsConfig using default values

        Parameters
        ----------
        Class : `type`
            Class of instance to construct (``PfsConfig`` or ``PfsDesign).
        **kwargs : `dict`
            Arguments for the constructor. Any missing arguments
            will be provided with defaults from the test.

        Returns
        -------
        instance : ``Class``
            Constructed PfsConfig or PfsDesign.
        """
        # fiberStatus is special, for backwards-compatibility reasons
        needNames = set(Class._keywords + Class._scalars + ["fiberStatus"])
        haveNames = set(kwargs.keys())
        assert len(haveNames - needNames) == 0, "Unrecognised argument"
        for name in needNames - haveNames:
            kwargs[name] = getattr(self, name)
        return Class(**kwargs)

    def makePfsDesign(self, **kwargs):
        """Construct a PfsDesign using default values from the test

        Parameters
        ----------
        **kwargs : `dict`
            Arguments for the `PfsDesign` constructor. Any missing arguments
            will be provided with defaults from the test.

        Returns
        -------
        pfsDesign : `PfsDesign`
            Constructed pfsConfig.
        """
        return self._makeInstance(PfsDesign, **kwargs)

    def makePfsConfig(self, **kwargs):
        """Construct a PfsConfig using default values from the test

        Parameters
        ----------
        **kwargs : `dict`
            Arguments for the `PfsConfig` constructor. Any missing arguments
            will be provided with defaults from the test.

        Returns
        -------
        pfsConfig : `PfsConfig`
            Constructed pfsConfig.
        """
        return self._makeInstance(PfsConfig, **kwargs)

    def assertPfsDesign(self, lhs, rhs):
        self.assertEqual(lhs.pfsDesignId, rhs.pfsDesignId)
        for value in ("raBoresight", "decBoresight", "posAng", "arms"):
            # Our FITS header writer can introduce some tiny roundoff error
            self.assertAlmostEqual(getattr(lhs, value), getattr(rhs, value), 14, value)
        for value in ("fiberId", "tract", "ra", "dec", "catId", "objId",
                      "pfiNominal", "targetType", "fiberStatus",
                      "epoch", "pmRa", "pmDec", "parallax",
                      "proposalId", "obCode"):
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
        self.assertEqual(lhs.designName, rhs.designName)

    def assertPfsConfig(self, lhs, rhs):
        self.assertEqual(lhs.visit, rhs.visit)
        np.testing.assert_array_equal(lhs.pfiCenter, rhs.pfiCenter)

    def testBasic(self):
        """Test basic operation of PfsConfig"""
        config = self.makePfsConfig()

        dirName = os.path.splitext(__file__)[0]
        if not os.path.exists(dirName):
            os.makedirs(dirName)

        filename = os.path.join(dirName, config.filename)
        if os.path.exists(filename):
            os.unlink(filename)

        try:
            config.write(dirName=dirName)
            other = PfsConfig.read(self.pfsDesignId, self.visit, dirName=dirName)
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

        # Longer arrays
        for name in ("fiberId", "tract", "patch", "ra", "dec", "catId", "objId",
                     "epoch", "pmRa", "pmDec", "parallax",
                     "proposalId", "obCode"):
            with self.assertRaises(RuntimeError):
                self.makePfsConfig(**{name: extendArray(getattr(self, name))})

        # Arrays with bad enums
        targetType = self.targetType.copy()
        targetType[self.numFibers//2] = -1
        with self.assertRaises(ValueError):
            self.makePfsConfig(targetType=targetType)
        fiberStatus = self.fiberStatus.copy()
        fiberStatus[self.numFibers//2] = -1
        with self.assertRaises(ValueError):
            self.makePfsConfig(fiberStatus=fiberStatus)

        # Fluxes
        for name in ("fiberFlux", "psfFlux", "totalFlux", "fiberFluxErr", "psfFluxErr", "totalFluxErr"):
            array = [extendArray(mag) if ii == self.numFibers//2 else mag
                     for ii, mag in enumerate(getattr(self, name))]
            with self.assertRaises(RuntimeError):
                self.makePfsConfig(**{name: array})

        # Arrays of points
        for name in ("pfiCenter", "pfiNominal"):
            array = getattr(self, name)
            array = np.concatenate((array, array), axis=1)
            with self.assertRaises(RuntimeError):
                self.makePfsConfig(**{name: array})

        # Duplicate fiberIds
        fiberId = self.fiberId.copy()
        fiberId[123] = fiberId[456]
        with self.assertRaises(ValueError):
            self.makePfsDesign(fiberId=fiberId)

        # Duplicate (catId, objId) combos
        catId = list(self.catId)
        objId = list(self.objId)
        catId[0] = catId[1]
        objId[0] = objId[1]
        catId[2] = catId[3]
        objId[2] = objId[3]
        with self.assertRaises(ValueError):
            self.makePfsDesign(catId=catId, objId=objId)

    def testNonTargetedFiber(self):
        """Tests duplicate (catId, objId) combos
            where catId=-1 and objId=-1
            are accepted, as these correspond to non-targetted fibers
        """
        catId = list(self.catId)
        objId = list(self.objId)
        catId[0] = -1
        objId[0] = -1
        catId[1] = -1
        objId[1] = -1
        self.makePfsDesign(catId=catId, objId=objId)

    def testFromPfsDesign(self):
        """Test PfsConfig.fromPfsDesign"""
        design = self.makePfsDesign()
        config = self.makePfsConfig()
        self.assertPfsConfig(PfsConfig.fromPfsDesign(design, self.visit, self.pfiCenter), config)

    def testDesignName(self):
        """Test creation of design name"""
        name = 'TestName123'
        design = self.makePfsDesign(designName=name)
        self.assertEqual(design.designName, name)

    def testPfsDesignsSame(self):
        """Tests that two designs created with the same parameters are equal.
        """
        design0 = self.makePfsDesign()
        design1 = self.makePfsDesign()
        self.assertPfsDesign(design0, design1)

    def testPfsDesignsSameDesignName(self):
        """Tests that two designs with the same design name and otherwise equal parameters are equal.
        """
        name = 'TestName123'
        design0 = self.makePfsDesign(designName=name)
        design1 = self.makePfsDesign(designName=name)
        self.assertPfsDesign(design0, design1)

    def testPfsDesignsNotSameDesignName(self):
        """Tests that two designs with the same design name and otherwise equal parameters are not equal.
        """
        design0 = self.makePfsDesign(designName='')
        design1 = self.makePfsDesign(designName='TestName123')
        with self.assertRaises(AssertionError):
            self.assertPfsDesign(design0, design1)

    def testPfsDesignVariants(self):
        """Test that pfsDesign variants have correct provenance."""
        design0 = self.makePfsDesign()
        design1 = self.makePfsDesign(variant=1, designId0=design0.pfsDesignId)
        design2 = self.makePfsDesign(variant=2, designId0=design0.pfsDesignId)

        variant0, designForVariant0 = design0.getVariant()
        self.assertEqual(variant0, 0)
        self.assertEqual(designForVariant0, 0)

        baseDesignId = design0.pfsDesignId
        variant1, designForVariant1 = design1.getVariant()
        variant2, designForVariant2 = design2.getVariant()
        self.assertNotEqual(variant1, variant0)
        self.assertNotEqual(variant2, variant1)

        self.assertEqual(baseDesignId, designForVariant1)
        self.assertEqual(baseDesignId, designForVariant2)

    def testPfsConfigVariants(self):
        """Test that pfsConfig can be create from a design with variant """
        design0 = self.makePfsDesign()
        variant1 = 1
        design1 = self.makePfsDesign(variant=variant1, designId0=design0.pfsDesignId)

        config1 = PfsConfig.fromPfsDesign(design1, self.visit, self.pfiCenter)
        self.assertEqual((variant1, design0.pfsDesignId), config1.getVariant())

    def testPfsConfigCopy(self):
        """Test that pfsConfig can be copied with modified entries."""
        config = self.makePfsConfig()

        # making a copy just changing visit
        newVisit = config.visit + 1
        configCopy = config.copy(visit=newVisit)
        self.assertEqual(configCopy.visit, newVisit)

        # checking that those are still equal.
        for scalar in set(configCopy._scalars) - {'visit', 'guideStars'}:
            self.assertEqual(getattr(configCopy, scalar), getattr(config, scalar))

        for field in ['fiberId', 'fiberStatus', 'targetType', 'pfiNominal', 'pfiCenter']:
            np.testing.assert_array_equal(getattr(configCopy, field), getattr(config, field))

    def testAdditionalHeaderCards(self):
        """Test that the provided additional header cards are indeed written to the primary hdu."""

        def getAdditionalCards(visit):
            cards = dict()

            now = datetime.datetime.now(datetime.timezone.utc)
            dayStr = now.strftime('%Y-%m-%d')

            frameId = f'PFSF{visit:06}00'
            expId = f'PFSE{visit:08}'

            cards['FRAMEID'] = (frameId, 'Sequence number in archive')
            cards['EXP-ID'] = (expId, 'Grouping ID for PFS visit')
            cards['TELESCOP'] = ("Subaru", 'Telescope name')
            cards['INSTRUME'] = ("PFS", 'Instrument name')
            cards['OBSERVER'] = ("ALF", 'Name(s) of observer(s)')
            cards['PROP-ID'] = ("o22016", 'Proposal ID')
            cards['DATE-OBS'] = (dayStr, '[YMD] pfsConfig creation date, UTC')

            return cards

        header = getAdditionalCards(67891)
        config = self.makePfsConfig(header=header)

        dirName = os.path.splitext(__file__)[0] + "_PfsConfigTestCase_testAdditionalHeaderCards"
        if not os.path.exists(dirName):
            os.makedirs(dirName)

        filename = os.path.join(dirName, config.filename)
        if os.path.exists(filename):
            os.unlink(filename)

        try:
            config.write(dirName=dirName)
            # re-open the file
            with pyfits.open(filename) as fd:
                phu = fd[0].header
                for key, (value, comment) in header.items():
                    assert key in phu

                    self.assertEqual(value, phu[key])
                    self.assertEqual(comment, phu.comments[key])

        except Exception:
            raise  # Leave file for manual inspection
        else:
            os.unlink(filename)

    def testFromEmptyGuideStars(self):
        """Check that an empty GuideStars instance is correctly instantiated
        if a None value is passed to the corresponding constructor argument
        """
        design = self.makePfsDesign(guideStars=None)
        self.checkGsEmpty(design)

        config = self.makePfsConfig(guideStars=None)
        self.checkGsEmpty(config)

        # Check that a non-empty GuideStar instance can be passed during construction.
        gsNotEmpty = GuideStars.empty()  # Using a tweaked version of an empty GuideStars instance.
        telElev = 123
        gsNotEmpty.telElev = telElev

        design = self.makePfsDesign(guideStars=gsNotEmpty)
        gs = design.guideStars
        self.checkGsArrayAttributesEmpty(gs)
        self.assertEqual(gs.telElev, telElev)

    def checkGsEmpty(self, design):
        """Check that the contents of the
        GuideStars attribute of the PfsDesign is empty.
        """
        gs = design.guideStars
        self.checkGsArrayAttributesEmpty(gs)
        self.assertEqual(gs.telElev, 0.0)
        self.assertEqual(gs.guideStarCatId, 0)

    def checkGsArrayAttributesEmpty(self, gs):
        """Check that the array-like
        attributes of the passed GuideStars
        instance are empty.
        """
        for att in ['objId', 'epoch',
                    'ra', 'dec',
                    'pmRa', 'pmDec',
                    'parallax', 'magnitude',
                    'passband', 'color',
                    'agId', 'agX', 'agY',
                    'epoch']:
            value = getattr(gs, att)
            self.assertTrue(value is not None)
            self.assertTrue(len(value) == 0)

    def testGetitem(self):
        """Test __getitem__"""
        select = np.array([ii % 2 == 0 for ii in range(self.numFibers)], dtype=bool)
        numSelected = select.sum()
        assert numSelected < self.numFibers
        pfsConfig = self.makePfsConfig()
        sub = pfsConfig[select]
        self.assertEqual(len(sub), numSelected)
        self.assertFloatsEqual(sub.fiberId, pfsConfig.fiberId[select])
        self.assertFloatsEqual(sub.objId, pfsConfig.objId[select])

        with self.assertRaises(RuntimeError):
            # Fails because we didn't specify 'allowSubset=True'
            sub.write()

        # But we can write when we use 'allowSubset=True'
        dirName = os.path.splitext(__file__)[0] + "_PfsConfigTestCase_testGetitem"
        if not os.path.exists(dirName):
            os.makedirs(dirName)

        filename = os.path.join(dirName, sub.filename)
        if os.path.exists(filename):
            os.unlink(filename)

        try:
            sub.write(dirName=dirName, allowSubset=True)
            new = PfsConfig.read(self.pfsDesignId, self.visit, dirName=dirName)
            self.assertPfsConfig(new, sub)
        except Exception:
            raise  # Leave file for manual inspection
        else:
            os.unlink(filename)

    def testSelect(self):
        """Test select method"""
        pfsConfig = self.makePfsConfig()

        fiberId = self.fiberId[3]
        sub = pfsConfig.select(fiberId=fiberId)
        self.assertEqual(len(sub), 1)
        self.assertFloatsEqual(sub.fiberId, fiberId)

        targetType = TargetType.FLUXSTD
        sub = pfsConfig.select(targetType=targetType)
        self.assertEqual(len(sub), self.numFluxStd)
        self.assertFloatsEqual(sub.targetType, targetType)

        fiberStatus = FiberStatus.BROKENFIBER
        sub = pfsConfig.select(fiberStatus=fiberStatus)
        self.assertEqual(len(sub), self.numBroken)
        self.assertFloatsEqual(sub.fiberStatus, fiberStatus)

        index = 2*self.numFibers//3
        sub = pfsConfig.select(catId=pfsConfig.catId[index], tract=pfsConfig.tract[index],
                               patch=pfsConfig.patch[index], objId=pfsConfig.objId[index])
        self.assertEqual(len(sub), 1)
        self.assertEqual(sub.catId[0], pfsConfig.catId[index])
        self.assertEqual(sub.tract[0], pfsConfig.tract[index])
        self.assertEqual(sub.patch[0], pfsConfig.patch[index])
        self.assertEqual(sub.objId[0], pfsConfig.objId[index])

        indices = np.array([42, 37, 1234])
        sub = pfsConfig.select(fiberId=self.fiberId[indices])
        self.assertEqual(len(sub), len(indices))
        self.assertFloatsEqual(sub.fiberId, pfsConfig.fiberId[np.sort(indices)])

        fiberStatus = (FiberStatus.BROKENFIBER, FiberStatus.BLOCKED)
        sub = pfsConfig.select(fiberStatus=fiberStatus)
        self.assertEqual(len(sub), self.numBroken + self.numBlocked)
        select = np.zeros(len(pfsConfig), dtype=bool)
        for ff in fiberStatus:
            select |= pfsConfig.fiberStatus == ff
        self.assertFloatsEqual(sub.fiberStatus, pfsConfig[select].fiberStatus)

        with self.assertRaises(RuntimeError):
            # Fails because we didn't specify 'allowSubset=True'
            sub.write()

    def testSelectFiber(self):
        """Test selectFiber"""
        pfsConfig = self.makePfsConfig()

        index = 37
        result = pfsConfig.selectFiber(pfsConfig.fiberId[index])
        self.assertEqual(result, 37)

        index = np.array([42, 37, 1234])
        result = pfsConfig.selectFiber(pfsConfig.fiberId[index])
        self.assertFloatsEqual(result, sorted(index))  # Note the need to sort

        self.assertRaises(RuntimeError, pfsConfig.selectFiber, 123456789)  # Scalar fiberId that's not present

    def testSelectByTargetType(self):
        """Test selectByTargetType"""
        pfsConfig = self.makePfsConfig()
        for name in TargetType.__members__:
            targetType = getattr(TargetType, name)
            indices = pfsConfig.selectByTargetType(targetType)
            select = pfsConfig.targetType == targetType
            self.assertEqual(len(indices), select.sum())
            self.assertFloatsEqual(pfsConfig.fiberId[indices], pfsConfig.fiberId[select])

            fiberId = pfsConfig.fiberId[::-1]
            indices = pfsConfig.selectByTargetType(targetType, fiberId)
            self.assertEqual(len(indices), select.sum())
            self.assertFloatsEqual(pfsConfig.fiberId[indices], pfsConfig.fiberId[select[::-1]])

    def testSelectByFiberStatus(self):
        """Test selectByFiberStatus"""
        pfsConfig = self.makePfsConfig()
        for name in FiberStatus.__members__:
            fiberStatus = getattr(FiberStatus, name)
            indices = pfsConfig.selectByFiberStatus(fiberStatus)
            select = pfsConfig.fiberStatus == fiberStatus
            self.assertEqual(len(indices), select.sum())
            self.assertFloatsEqual(pfsConfig.fiberId[indices], pfsConfig.fiberId[select])

            fiberId = pfsConfig.fiberId[::-1]
            indices = pfsConfig.selectByFiberStatus(fiberStatus, fiberId)
            self.assertEqual(len(indices), select.sum())
            self.assertFloatsEqual(pfsConfig.fiberId[indices], pfsConfig.fiberId[select[::-1]])

    def testGetCameraMask(self):
        """Test getCameraMask method."""
        allCams = PfsConfig._allCams  # Retrieve the full list of cameras

        # Test with an empty list
        self.assertEqual(PfsConfig.getCameraMask([]), 0)

        # Test with a single camera
        self.assertEqual(PfsConfig.getCameraMask([allCams[0]]), 1 << 0)

        # Test with multiple cameras
        self.assertEqual(PfsConfig.getCameraMask([allCams[0], allCams[1]]), (1 << 0) | (1 << 1))

        # Test with all cameras
        expectedMask = sum(1 << i for i in range(len(allCams)))
        self.assertEqual(PfsConfig.getCameraMask(allCams), expectedMask)

        # Test with an invalid camera
        with self.assertRaises(ValueError):
            PfsConfig.getCameraMask(["invalidCam"])

    def testGetCameraList(self):
        """Test getCameraList method."""
        allCams = PfsConfig._allCams  # Retrieve the full list of cameras
        config = self.makePfsConfig()

        # Test with an empty mask
        config.camMask = 0
        self.assertEqual(config.getCameraList(), [])

        # Test with a single camera
        config.camMask = 1 << 0
        self.assertEqual(config.getCameraList(), [allCams[0]])

        # Test with multiple cameras
        config.camMask = (1 << 0) | (1 << 1)
        self.assertEqual(config.getCameraList(), [allCams[0], allCams[1]])

        # Test with all cameras
        config.camMask = sum(1 << i for i in range(len(allCams)))
        self.assertEqual(config.getCameraList(), allCams)

    def testSetInstrumentStatusFlag(self):
        """Test setting instrument status mask with valid and invalid flags."""
        config = self.makePfsConfig()

        # Ensure initial bitmask value is zero
        self.assertEqual(config.instStatusFlag, 0)
        # Ensure absence of flag is decoded correctly.
        self.assertEqual('OK', config.decodeInstrumentStatusFlag())

        # Set a valid flag (INSROT_MISMATCH) and check that the bitmask updates correctly
        config.setInstrumentStatusFlag(InstrumentStatusFlag.INSROT_MISMATCH)
        self.assertEqual(config.instStatusFlag, InstrumentStatusFlag.INSROT_MISMATCH)

        # Ensure setting the same flag again does not change the mask
        config.setInstrumentStatusFlag(InstrumentStatusFlag.INSROT_MISMATCH)
        self.assertEqual(config.instStatusFlag, InstrumentStatusFlag.INSROT_MISMATCH)

        # Ensure decoding function works correctly
        activeFlags = config.decodeInstrumentStatusFlag()
        self.assertEqual('INSROT_MISMATCH', activeFlags)

        # Ensure we get the correct flag description.
        for flag in InstrumentStatusFlag:
            description = config.getInstrumentStatusDescription(flag)
            self.assertEqual(description, InstrumentStatusDescription[flag])

        # Attempt to set an invalid flag (integer instead of a valid InstrumentStatusFlag)
        # Expect a ValueError since only documented flags should be allowed
        with self.assertRaises(ValueError):
            config.setInstrumentStatusFlag(4)  # 4 is not a defined InstrumentStatusFlag

        with self.assertRaises(ValueError):
            config.setInstrumentStatusFlag("INVALID_FLAG")  # Non-integer input


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
