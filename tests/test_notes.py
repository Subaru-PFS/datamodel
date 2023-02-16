import unittest

import numpy as np
import astropy.io.fits
import lsst.utils.tests

from pfs.datamodel.notes import makeNotesClass
from pfs.datamodel.pfsTable import PfsTable, Column


schema = [
    Column("string", str, "This is a string", ""),
    Column("integer", int, "This is an integer", -1),
    Column("number", float, "This is a float", 1.2345),  # Not using NAN, so we can test equality
    Column("boolean", bool, "This is a boolean", False),
]

MyNotes = makeNotesClass("MyNotes", schema, "MY_NOTES")


class MyNotesArray(PfsTable):
    schema = schema
    fitsExtName = "MY_NOTES_ARRAY"


class NotesTestCase(unittest.TestCase):
    """Test the Notes class (single spectrum)"""
    def testBasic(self):
        """Test basic functionality"""
        string = "foobar"
        integer = 12345  # I have the same combination on my luggage
        number = 9.876
        boolean = True
        notes = MyNotes(string, integer, number, boolean)
        self.assertEqual(notes.string, string)
        self.assertEqual(notes.integer, integer)
        self.assertEqual(notes.number, number)
        self.assertEqual(notes.boolean, boolean)
        self.checkIO(notes)

    def checkIO(self, notes):
        """Check that I/O works

        Parameters
        ----------
        notes : `Notes`
            Notes to write and read back.
        """
        fits = astropy.io.fits.HDUList()
        notes.writeFits(fits)
        new = MyNotes.readFits(fits)
        for col in schema:
            self.assertEqual(getattr(notes, col.name), getattr(new, col.name))

    def testEmpty(self):
        """Test that an empty Notes object is created correctly"""
        notes = MyNotes()
        for col in schema:
            self.assertEqual(getattr(notes, col.name), col.default)
        self.checkIO(notes)


class NotesArrayTestCase(unittest.TestCase):
    """Test the PfsTable class for use as notes with multiple spectra"""
    def testBasic(self):
        """Test basic functionality"""
        string = np.array(["foo", "bar"])
        integer = np.array([12345, 67890])
        number = np.array([9.876, 5.432])
        boolean = np.array([True, False])
        notes = MyNotesArray(string=string, integer=integer, number=number, boolean=boolean)
        self.assertEqual(len(notes), 2)
        np.testing.assert_equal(notes.string, string)
        np.testing.assert_equal(notes.integer, integer)
        np.testing.assert_equal(notes.number, number)
        np.testing.assert_equal(notes.boolean, boolean)
        self.checkIO(notes)

    def checkIO(self, notes):
        """Check that I/O works

        Parameters
        ----------
        notes : `Notes`
            Notes to write and read back.
        """
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            notes.writeFits(filename)
            new = MyNotesArray.readFits(filename)
            for col in schema:
                np.testing.assert_equal(getattr(notes, col.name), getattr(new, col.name))

    def testEmpty(self):
        """Test that an empty PfsTable notes object is created correctly"""
        length = 2
        notes = MyNotesArray.empty(length)
        self.assertEqual(len(notes), 2)
        for col in schema:
            np.testing.assert_equal(getattr(notes, col.name), [col.default]*length)
        self.checkIO(notes)


if __name__ == "__main__":
    unittest.main(failfast=True)
