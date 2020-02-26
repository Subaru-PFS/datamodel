import unittest
from pfs.datamodel.utils import createHash


class CreateHashTestCase(unittest.TestCase):
    """ Tests the pfs.model.utils.createHash() function.
    """

    def testList(self):

        self.assertEqual(createHash('1', '2'), 0x39753077792b0554)

    def testListComprehension(self):
        listA = ['a', 'b']
        listB = ['y', 'z']
        listC = [a + b for a in listA for b in listB]

        self.assertEqual(createHash(listC), 0x6f0e879d749c2d60)
