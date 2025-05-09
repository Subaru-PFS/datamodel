import sys
import unittest

import numpy as np

import lsst.utils.tests

from pfs.datamodel.objectGroupMap import ObjectGroupMap


class ObjectGroupMapTestCase(lsst.utils.tests.TestCase):
    """Test for ObjectGroupMap"""
    def setUp(self):
        self.rng = np.random.RandomState(12345)

    def makeObjectGroupMap(
        self,
        numObjects: int,
        numGroups: int,
    ) -> ObjectGroupMap:
        """Create a simple ObjectGroupMap for testing

        Parameters
        ----------
        numObjects : `int`
            Number of objects.
        numGroups : `int`
            Number of different ``objGroup`` values.
        """
        objId = np.arange(1, numObjects + 1, dtype=int)
        objGroup = self.rng.randint(1, numGroups + 1, size=numObjects)

        assert np.all(objId >= 1)
        assert np.all(objId <= numObjects)
        assert np.all(objGroup >= 1)
        assert np.all(objGroup <= numGroups)

        return ObjectGroupMap(objId, objGroup)

    def assertObjectGroupMap(self, ogm: ObjectGroupMap, num: int):
        """Check that the ObjectGroupMap works as expected

        This might be slow, as it iterates over all the objects, one by one.

        Parameters
        ----------
        ogm : `ObjectGroupMap`
            The object group map to check.
        num : `int`
            The expected number of objects in the map.
        """
        self.assertEqual(len(ogm), num)

        # Objects map to their own group
        self.assertFloatsEqual(ogm.objGroup, ogm[ogm.objId])

        # Check that the mapping works in a different order
        objGroup = np.unique(ogm.objGroup)
        for gg in objGroup:
            objId = ogm.objId[ogm.objGroup == gg].copy()
            self.rng.shuffle(objId)  # For good measure

            # Array lookup
            self.assertFloatsEqual(ogm[objId], gg)

            # Scalar lookup
            for ii in range(objId.size):
                result = ogm[objId[ii]]
                self.assertIsInstance(result, int)
                self.assertEqual(result, gg)

    def assertObjectGroupMapEqual(self, ogm1: ObjectGroupMap, ogm2: ObjectGroupMap):
        """Check that two ObjectGroupMap objects are equal"""
        self.assertEqual(len(ogm1), len(ogm2))
        self.assertFloatsEqual(ogm1.objId, ogm2.objId)
        self.assertFloatsEqual(ogm1.objGroup, ogm2.objGroup)

    def testBasic(self):
        """Test basic functionality of ObjectGroupMap"""
        num = 100
        numGroups = 2
        ogm = self.makeObjectGroupMap(num, numGroups)
        self.assertObjectGroupMap(ogm, num)

    def testIO(self):
        """Test I/O of ObjectGroupMap"""
        num = 1000
        numGroups = 3
        ogm = self.makeObjectGroupMap(num, numGroups)
        self.assertEqual(len(ogm), num)

        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            ogm.writeFits(filename)
            new = ObjectGroupMap.readFits(filename)
        self.assertObjectGroupMapEqual(ogm, new)

    def testLarge(self):
        """Test large ObjectGroupMap"""
        num = 100000
        numGroups = 10
        ogm = self.makeObjectGroupMap(num, numGroups)
        self.assertEqual(len(ogm), num)

        index = np.arange(num)
        self.rng.shuffle(index)
        objId = ogm.objId[index]
        objGroup = ogm.objGroup[index]

        check = ogm[objId]
        self.assertFloatsEqual(check, objGroup)

    def testMissing(self):
        """Test missing values in ObjectGroupMap"""
        num = 100
        numGroups = 3
        ogm = self.makeObjectGroupMap(num, numGroups)
        self.assertEqual(len(ogm), num)

        self.assertRaises(KeyError, ogm.__getitem__, num + 1)  # objId is out of range
        self.assertRaises(KeyError, ogm.__getitem__, 0)  # objId is out of range
        ogm[1]  # This should work

    def testCombine(self):
        """Test combining ObjectGroupMap objects"""
        num = 100
        numGroups = 3
        ogm1 = self.makeObjectGroupMap(num, numGroups)

        # Different objIds in the second map
        ogm2 = ObjectGroupMap(ogm1.objId + num, ogm1.objGroup + numGroups)
        combined = ogm1 + ogm2
        self.assertObjectGroupMapEqual(combined, ogm2 + ogm1)  # Different order
        self.assertObjectGroupMapEqual(combined, ObjectGroupMap.combine(ogm1, ogm2))  # Different name
        self.assertObjectGroupMapEqual(combined, ObjectGroupMap.combine(ogm2, ogm1))  # Different order
        self.assertObjectGroupMap(combined, 2*num)

    def testNonunique(self):
        """Test non-unique objId in ObjectGroupMap"""
        num = 100
        numGroups = 3
        ogm = self.makeObjectGroupMap(num, numGroups)

        self.assertRaises(ValueError, ObjectGroupMap, np.ones_like(ogm.objId), ogm.objGroup)

        objId = ogm.objId.copy()
        objGroup = ogm.objGroup.copy()

        index1 = 34
        index2 = 56
        dupObjId = 456
        objId[index1] = dupObjId
        objId[index2] = dupObjId
        self.assertRaises(ValueError, ObjectGroupMap, objId, objGroup)


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    setup_module(sys.modules["__main__"])
    unittest.main()
