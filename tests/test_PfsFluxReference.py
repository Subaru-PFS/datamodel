import io
import os
import sys
import unittest

import numpy as np
import numpy.testing

import lsst.utils.tests

from pfs.datamodel.identity import Identity
from pfs.datamodel.masks import MaskHelper
from pfs.datamodel.pfsConfig import PfsConfig, TargetType, FiberStatus
from pfs.datamodel.pfsFluxReference import PfsFluxReference
from pfs.datamodel.pfsSimpleSpectrum import PfsSimpleSpectrum
from pfs.datamodel.wavelengthArray import WavelengthArray

flt_epsilon = np.nextafter(np.float32(1), np.float32(np.inf)) - 1


class PfsFluxReferenceTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        if hasattr(np.random, "default_rng"):
            self.np_random = np.random.default_rng(0x900d5eed)
        else:
            # Old numpy
            np.random.seed(0xd1ff5eed)
            self.np_random = np.random

        self.numSpectra = 50
        self.numSamples = 1000

        self.identity = Identity(visit=654321)
        self.fiberId = np.arange(1, self.numSpectra + 1)
        self.wavelength = WavelengthArray(400, 1200, self.numSamples, dtype=float)
        self.flux = self.np_random.uniform(1, 2, size=(self.numSpectra, self.numSamples))
        self.metadata = {"ABCDE": 12345}
        self.fitFlagNames = MaskHelper()
        self.fitFlag = self.np_random.choice([0, self.fitFlagNames.add("FAIL")], size=self.numSpectra)

        self.fitParams = np.empty(self.numSpectra, dtype=[("alpha", float), ("beta", float)])
        self.fitParams["alpha"] = self.np_random.uniform(1, 2, size=self.numSpectra)
        self.fitParams["beta"] = self.np_random.uniform(1, 2, size=self.numSpectra)

        self.instance = PfsFluxReference(
            self.identity, self.fiberId, self.wavelength, self.flux,
            self.metadata, self.fitFlag, self.fitFlagNames, self.fitParams
        )

    def testLen(self):
        self.assertEqual(len(self.instance), self.numSpectra)

    def testGetItem(self):
        index = (self.instance.fitFlag == 0)
        self.assertEqual(len(self.instance[index]), len(self.instance.fitFlag[index]))

    def testFilename(self):
        path = os.path.join("/path", self.instance.filename)
        identity = PfsFluxReference._parseFilename(path)
        self.assertEqual(identity, self.identity)

    def testWriteFits(self):
        # Write and read
        fileobj = io.BytesIO()
        self.instance.writeFits(fileobj)
        fileobj.seek(0)
        instance = PfsFluxReference.readFits(fileobj)

        # Make sure that the read one is again writable.
        fileobj = io.BytesIO()
        instance.writeFits(fileobj)
        fileobj.seek(0)
        instance = PfsFluxReference.readFits(fileobj)

        self.assertEqual(instance.identity, self.identity)
        self.assertIsNone(numpy.testing.assert_array_equal(instance.fiberId, self.fiberId))
        self.assertFloatsAlmostEqual(instance.wavelength, self.wavelength, rtol=flt_epsilon)
        self.assertFloatsAlmostEqual(instance.flux, self.flux, rtol=flt_epsilon)
        self.assertIsNone(numpy.testing.assert_array_equal(instance.fitFlag, self.fitFlag))
        self.assertEqual(instance.fitFlagNames.flags, self.fitFlagNames.flags)
        for key in self.fitParams.dtype.names:
            self.assertFloatsAlmostEqual(instance.fitParams[key], self.fitParams[key], rtol=flt_epsilon)

        # The read metadata has more keys than the original one.
        # We check that the original (key, value) are preserved.
        for key, value in self.metadata.items():
            self.assertEqual(instance.metadata[key], value)

    def testExtractFiber(self):
        pfsConfig = PfsConfig(
            pfsDesignId=1,
            visit=1,
            raBoresight=123,
            decBoresight=45,
            posAng=0,
            arms="brn",
            fiberId=self.fiberId,
            tract=np.full(shape=self.numSpectra, fill_value=1234, dtype=np.int32),
            patch=np.full(shape=self.numSpectra, fill_value="5,6", dtype="U8"),
            ra=np.full(shape=self.numSpectra, fill_value=123, dtype=float),
            dec=np.full(shape=self.numSpectra, fill_value=45, dtype=float),
            catId=np.full(shape=self.numSpectra, fill_value=1, dtype=np.int32),
            objId=np.arange(1, self.numSpectra + 1, dtype=np.int64),
            targetType=np.full(shape=self.numSpectra, fill_value=TargetType.FLUXSTD, dtype=np.int32),
            fiberStatus=np.full(shape=self.numSpectra, fill_value=FiberStatus.GOOD, dtype=np.int32),
            epoch=np.full(shape=self.numSpectra, fill_value="J2000.0"),
            pmRa=np.full(shape=self.numSpectra, fill_value=0.0, dtype=np.float32),
            pmDec=np.full(shape=self.numSpectra, fill_value=0.0, dtype=np.float32),
            parallax=np.full(shape=self.numSpectra, fill_value=1e-5, dtype=np.float32),
            proposalId=np.full(shape=self.numSpectra, fill_value="J2000.0"),
            obCode=np.array([f"obcode_{fibnum:04d}" for fibnum in range(self.numSpectra)]),
            fiberFlux=[np.zeros(0, dtype=float)] * self.numSpectra,
            psfFlux=[np.zeros(0, dtype=float)] * self.numSpectra,
            totalFlux=[np.zeros(0, dtype=float)] * self.numSpectra,
            fiberFluxErr=[np.zeros(0, dtype=float)] * self.numSpectra,
            psfFluxErr=[np.zeros(0, dtype=float)] * self.numSpectra,
            totalFluxErr=[np.zeros(0, dtype=float)] * self.numSpectra,
            filterNames=[[]] * self.numSpectra,
            pfiCenter=np.zeros(shape=(self.numSpectra, 2), dtype=float),
            pfiNominal=np.zeros(shape=(self.numSpectra, 2), dtype=float),
            guideStars=None,
        )
        fiberId = 1
        spectrum = self.instance.extractFiber(PfsSimpleSpectrum, pfsConfig, fiberId)
        self.assertIsInstance(spectrum, PfsSimpleSpectrum)


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    setup_module(sys.modules["__main__"])
    unittest.main()
