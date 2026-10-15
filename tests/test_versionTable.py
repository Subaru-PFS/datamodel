import unittest

import astropy.io.fits

from pfs.datamodel.versionTable import VersionTable, MAX_VALUE_LENGTH


class VersionTableTestCase(unittest.TestCase):
    """Test the VersionTable header I/O"""
    def setUp(self):
        self.comment = "XML version used for convergence"

    def write(self, value):
        """Write ``value`` as ``moduleXml`` in the ``config0`` table

        Returns the header and the dict read back from it.
        """
        header = astropy.io.fits.Header()
        VersionTable.write(header, "config0", dict(moduleXml=value))
        return header, VersionTable.read(header, "config0")

    def testLongValue(self):
        """A value over the limit is truncated from the front and flagged"""
        path = "/software/devel/moritani/pfs_instdata/data/pfi/modules/ALL/ALL_20260915_el30_center.xml"
        self.assertGreater(len(path), MAX_VALUE_LENGTH)
        header, versions = self.write(path)

        value = header["W_C0VR07"]
        self.assertEqual(len(value), MAX_VALUE_LENGTH)
        self.assertTrue(value.startswith("..."))
        self.assertTrue(path.endswith(value[3:]))
        self.assertTrue(value.endswith("ALL_20260915_el30_center.xml"))
        self.assertTrue(header.comments["W_C0VR07"].endswith("TRUNCATED"))
        self.assertTrue(header.comments["W_C0VR07"].startswith(self.comment))
        self.assertEqual(versions["moduleXml"], value)

    def testValueAtLimit(self):
        """A value exactly at the limit is written unchanged"""
        value = "x"*MAX_VALUE_LENGTH
        header, versions = self.write(value)
        self.assertEqual(header["W_C0VR07"], value)
        self.assertEqual(header.comments["W_C0VR07"], self.comment)
        self.assertEqual(versions["moduleXml"], value)

    def testShortValue(self):
        """A short value is written unchanged"""
        value = "ALL_20260915_el30_center.xml"
        header, versions = self.write(value)
        self.assertEqual(header["W_C0VR07"], value)
        self.assertEqual(header.comments["W_C0VR07"], self.comment)
        self.assertEqual(versions["moduleXml"], value)


if __name__ == "__main__":
    unittest.main(failfast=True)
