import sys
import unittest

import lsst.utils.tests

from pfs.datamodel.pfsConfig import DocEnum

display = None


class TestEnum(DocEnum):
    """Test enumeration"""
    ONE = 1, "one"
    TWO = 2, "two"
    THREE = 3, "three"


class DocEnumTestCase(lsst.utils.tests.TestCase):
    def testBasic(self):
        self.assertEqual(TestEnum.ONE, 1)
        self.assertEqual(TestEnum.TWO, 2)
        self.assertEqual(TestEnum.THREE, 3)

    def testStr(self):
        self.assertEqual(str(TestEnum.ONE), "ONE")
        self.assertEqual(str(TestEnum.TWO), "TWO")
        self.assertEqual(str(TestEnum.THREE), "THREE")

    def testFromString(self):
        self.assertEqual(TestEnum.fromString("ONE"), TestEnum.ONE)
        self.assertEqual(TestEnum.fromString("TWO"), TestEnum.TWO)
        self.assertEqual(TestEnum.fromString("THREE"), TestEnum.THREE)

    def testInvert(self):
        inverted = ~TestEnum.ONE
        self.assertEqual(len(inverted), 2)
        self.assertIn(TestEnum.TWO, inverted)
        self.assertIn(TestEnum.THREE, inverted)

    def testFromList(self):
        result = TestEnum.fromList(["ONE", "THREE"])
        self.assertEqual(len(result), 2)
        self.assertIn(TestEnum.ONE, result)
        self.assertIn(TestEnum.THREE, result)

        result = TestEnum.fromList(["~THREE"])
        self.assertEqual(len(result), 2)
        self.assertIn(TestEnum.ONE, result)
        self.assertIn(TestEnum.TWO, result)

        result = TestEnum.fromList(["TWO", "~ONE"])
        self.assertEqual(len(result), 2)
        self.assertIn(TestEnum.TWO, result)
        self.assertIn(TestEnum.THREE, result)

        result = TestEnum.fromList(["~TWO", "~ONE"])
        self.assertEqual(len(result), 1)
        self.assertIn(TestEnum.THREE, result)

        with self.assertRaises(ValueError):
            TestEnum.fromList(["~TWO", "TWO"])


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
