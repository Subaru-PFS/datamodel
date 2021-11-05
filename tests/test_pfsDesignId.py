import unittest
from pfs.datamodel.utils import calculate_pfsDesignId

import numpy as np


class PfsDesignIdTestCase(unittest.TestCase):
    """ Tests the pfs.model.utils.calculate_pfsDesignId function.
    """

    def testResolution(self):
        """Tests that tweaking one target by less than 1 arcsec
           does not affect result"""
        fiberId = np.array([1, 456])
        ra = np.array([1.0, 123.7])
        dec = np.array([-81, 17.86])
        expected_designId = calculate_pfsDesignId(fiberId, ra, dec)

        # Adjust one target but within 1 arcsec
        raOk = np.copy(ra)
        decOk = np.copy(dec)
        raOk[1] += 0.4/3600
        decOk[1] += 0.3/3600
        self.assertEquals(expected_designId, calculate_pfsDesignId(fiberId, raOk, decOk))

        # Adjust one target but outside 1 arcsec
        raBad = np.copy(ra)
        decBad = np.copy(dec)
        raBad[1] += 0.7/3600
        decBad[1] += 0.6/3600
        self.assertNotEquals(expected_designId, calculate_pfsDesignId(fiberId, raBad, decBad))

    def testFullScienceFibers(self):
        """Tests with the full expected set of science fibers."""
        nFibers = 2394
        np.random.seed(123)
        fiberId = np.array([f for f in range(nFibers)])
        ra = np.array([np.random.uniform(0., 360.) for _ in range(nFibers)])
        dec = np.array([np.random.uniform(-90., 90.) for _ in range(nFibers)])
        self.assertEquals(0x189bef816fd508c0, calculate_pfsDesignId(fiberId, ra, dec))

    def testNone(self):
        """Tests when at least one, but not all, inputs
        are none."""
        a = np.array([11.0, 123.0, -5.0])
        for p in [(None, a, a),
                  (None, None, a),
                  (None, a, None),
                  (a, None, None)]:
            with self.subTest():
                with self.assertRaises(RuntimeError):
                    calculate_pfsDesignId(*p)
