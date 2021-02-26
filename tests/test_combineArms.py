import unittest
from pfs.datamodel.utils import combineArms


class CombineArmsTestCase(unittest.TestCase):
    """ Tests the pfs.model.utils.combineArms() function.
    """

    def testCombinations(self):

        self.assertEqual(combineArms({'b', 'r'}), 'br')
        self.assertEqual(combineArms({'b', 'n', 'r'}), 'brn')
        self.assertEqual(combineArms({'br', 'rb'}), 'br')
        self.assertEqual(combineArms({'br', 'rn'}), 'brn')
        self.assertEqual(combineArms({'bmn', 'b'}), 'bmn')
