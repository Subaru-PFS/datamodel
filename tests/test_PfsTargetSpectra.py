import sys
from typing import Any, Callable, Dict, Iterable, Optional, TYPE_CHECKING
import unittest
import itertools

import astropy.io.fits

import lsst.afw.image.testUtils
import lsst.utils.tests
import numpy as np
from lsst.pipe.base import Struct
from pfs.datamodel import FluxTable, MaskHelper, Observations, Target, TargetType
from pfs.datamodel import PfsFiberArray, PfsObject
from pfs.datamodel.pfsTargetSpectra import PfsTargetSpectra
from pfs.datamodel.drp import PfsCoadd

display = None


class PfsTargetSpectraTestCase(lsst.utils.tests.TestCase):
    if TYPE_CHECKING:
        assertFloatsEqual: Callable[..., None]
        assertFloatsAlmostEqual: Callable[..., None]

    def setUp(self):
        self.rng = np.random.RandomState(12345)  # I have the same combination on my luggage

    def makeTarget(self, catId: int, objId: int) -> Target:
        """Create a target with random values

        Parameters
        ----------
        catId : `int`
            Catalog identifier.
        objId : `int`
            Object identifier.

        Returns
        -------
        target : `Target`
            Target with random values.
        """
        return Target(
            catId=catId,
            tract=self.rng.randint(1000),
            patch="%d,%d" % tuple(self.rng.randint(100, size=2).tolist()),
            objId=objId,
            ra=self.rng.uniform(),
            dec=self.rng.uniform(),
            targetType=TargetType.SCIENCE,
            fiberFlux=dict(g=self.rng.uniform(), r=self.rng.uniform(), i=self.rng.uniform()),
        )

    def makeObservations(self, size: int = 3) -> Observations:
        """Create observations with random values

        Parameters
        ----------
        size : `int`
            Number of observations made.

        Returns
        -------
        observations : `Observations`
            Observations with random values.
        """
        return Observations(
            visit=self.rng.randint(1000000, size=size),
            arm=[chr(letter) for letter in self.rng.randint(ord("a"), ord("z"), size=size)],
            spectrograph=self.rng.randint(10, size=size),
            pfsDesignId=self.rng.randint(0xFFFFFFFF, size=size),
            fiberId=self.rng.randint(1000, size=size),
            pfiNominal=self.rng.uniform(size=(size, 2)),
            pfiCenter=self.rng.uniform(size=(size, 2)),
        )

    def makeSpectrum(
        self, length: int = 1000, minWavelength: float = 500, maxWavelength: float = 1000
    ) -> Struct:
        """Create a spectrum with random values

        Parameters
        ----------
        length : `int`
            Length of spectrum (pixels).
        minWavelength, maxWavelength : `float`
            Minimum and maximum wavelength (nm).

        Returns
        -------
        spectrum : `Struct`
            Spectrum arrays in a struct with the following attributes:
            - ``wavelength`` (`numpy.ndarray` of `float`)
            - ``flux`` (`numpy.ndarray` of `float`)
            - ``mask`` (`numpy.ndarray` of `int`)
            - ``sky`` (`numpy.ndarray` of `float`)
            - ``covar`` (`numpy.ndarray` of `float`, shape ``3,length``)
            - ``covar2`` (`numpy.ndarray` of `float`, shape ``5,5``)
        """
        return Struct(
            wavelength=np.linspace(minWavelength, maxWavelength, length, dtype=float),
            flux=self.rng.uniform(size=length).astype(np.float32),
            mask=self.rng.randint(0xFFFF, size=length).astype(np.int32),
            sky=self.rng.uniform(size=length).astype(np.float32),
            covar=self.rng.uniform(size=(3, length)).astype(np.float32),
            covar2=self.rng.uniform(size=(5, 5)).astype(np.float32),
        )

    def makeFlags(self) -> MaskHelper:
        """Create a set of flags

        Returns
        -------
        flags : `MaskHelper`
            A set of flags.
        """
        return MaskHelper(FOO=0, BAR=3, BAZ=13, QIX=7)

    def makeMetadata(self) -> Dict[str, Any]:
        """Create metadata

        Returns
        -------
        metadata : `dict` [`str`: POD]
            A set of metadata keyword-value pairs.
        """
        return dict(THIS="is a header", DEMO="nstration", E="mc^2")

    def makeFluxTable(
        self, length: int = 1000, minWavelength: float = 500, maxWavelength: float = 1000
    ) -> FluxTable:
        """Create a fluxTable with random values

        Parameters
        ----------
        length : `int`
            Length of spectrum (pixels).
        minWavelength, maxWavelength : `float`
            Minimum and maximum wavelength (nm).

        Returns
        -------
        fluxTable : `FluxTable`
            FluxTable with random values.
        """
        return FluxTable(
            wavelength=np.linspace(minWavelength, maxWavelength, length, dtype=float),
            flux=self.rng.uniform(size=length).astype(np.float32),
            error=self.rng.uniform(size=length).astype(np.float32),
            mask=self.rng.randint(0xFFFF, size=length).astype(np.int32),
            flags=self.makeFlags(),
        )

    def makePfsFiberArray(self, catId: int, objId: int, length: int = 1000, numObservations: int = 3):
        """Create a spectrum with all required metadata

        We create a `PfsObject`, as a specific subclass of `PfsFiberArray`,
        because we need something where the `NotesClass` class attribute is
        set.

        Parameters
        ----------
        catId : `int`
            Catalog identifier.
        objId : `int`
            Object identifier.
        length : `int`
            Length of spectrum (pixels).
        numObservations : `int`
            Number of observations made.

        Returns
        -------
        spectrum : `PfsFiberArray`
            Spectrum with random values.
        """
        spectrum = self.makeSpectrum(length)
        return PfsObject(
            target=self.makeTarget(catId, objId),
            observations=self.makeObservations(numObservations),
            wavelength=spectrum.wavelength,
            flux=spectrum.flux,
            mask=spectrum.mask,
            sky=spectrum.sky,
            covar=spectrum.covar,
            covar2=spectrum.covar2,
            flags=self.makeFlags(),
            metadata=self.makeMetadata(),
            fluxTable=self.makeFluxTable(length),
        )

    def makePfsTargetSpectra(
        self,
        objId: Iterable[int],
        *,
        catId: Optional[Iterable[int]] = None,
        length: Optional[Iterable[int]] = None,
        numObservations: int = 3,
    ) -> PfsTargetSpectra:
        """Create a set of spectra

        We create a `PfsCoadd`, as a specific subclass of
        `PfsTargetSpectra`, because we need something where the `NotesClass`
        class attribute is set.

        Parameters
        ----------
        objId : iterable of `int`, optional
            Object identifiers to use. If not provided, we use a range.
        catId : iterable of `int`, optional
            Catalog identifiers to use. If not provided, we use a constant
            value.
        length : iterable of `int`, optional
            Lengths of each spectrum (pixels). If not provided, we use a
            constant value.
        numObservations : `int`
            Number of observations made.

        Returns
        -------
        spectra : `PfsTargetSpectra`
            Spectra with random values.
        """
        if catId is None:
            catId = itertools.repeat(12345)
        if length is None:
            length = itertools.repeat(1000)
        metadata = self.makeMetadata()
        return PfsCoadd(
            [self.makePfsFiberArray(ii, jj, ll, numObservations) for ii, jj, ll in zip(catId, objId, length)],
            metadata,
        )

    def assertPfsTargetSpectraEqual(self, left: PfsTargetSpectra, right: PfsTargetSpectra):
        """Assert that two `PfsTargetSpectra` are equal"""
        for key in right.metadata:
            self.assertIn(key, right.metadata)
            self.assertEqual(left.metadata[key], right.metadata[key], key)

        self.assertEqual(len(left), len(right))
        for leftTarget, rightTarget in zip(left, right):
            self.assertTargetEqual(leftTarget, rightTarget)
            self.assertPfsFiberArrayEqual(left[leftTarget], right[leftTarget])

    def assertTargetEqual(self, left: Target, right: Target):
        """Assert that two `Target`s are equal"""
        for attr in ("catId", "tract", "patch", "objId", "ra", "dec", "targetType"):
            self.assertEqual(getattr(left, attr), getattr(right, attr), attr)
        self.assertEqual(left.fiberFlux.keys(), right.fiberFlux.keys())
        self.assertFloatsAlmostEqual(
            np.array(list(left.fiberFlux.values())), np.array(list(right.fiberFlux.values())), atol=1.0e-7
        )

    def assertObservationsEqual(self, left: Observations, right: Observations):
        """Assert that two `Observations` are equal"""
        for attr in ("visit", "spectrograph", "pfsDesignId", "fiberId", "pfiNominal", "pfiCenter"):
            self.assertFloatsEqual(getattr(left, attr), getattr(right, attr))
        self.assertListEqual(left.arm, right.arm)

    def assertFlagsEqual(self, left: MaskHelper, right: MaskHelper):
        """Assert that two `MaskHelper`s are equal"""
        self.assertDictEqual(left.flags, right.flags)

    def assertMetadataEqual(self, left: Dict[str, Any], right: Dict[str, Any]):
        """Assert that two header/metadatas are equal"""
        self.assertDictEqual(left, right)

    def assertFluxTableEqual(self, left: FluxTable, right: FluxTable):
        """Assert that two flux tables are equal"""
        self.assertFloatsEqual(left.wavelength, right.wavelength)
        self.assertFloatsEqual(left.flux, right.flux)
        self.assertFloatsEqual(left.error, right.error)
        self.assertFloatsEqual(left.mask, right.mask)
        self.assertFlagsEqual(left.flags, right.flags)

    def assertPfsFiberArrayEqual(self, left: PfsFiberArray, right: PfsFiberArray):
        """Assert that two spectra are equal"""
        self.assertTargetEqual(left.target, right.target)
        self.assertObservationsEqual(left.observations, right.observations)
        self.assertFloatsEqual(left.wavelength, right.wavelength)
        self.assertFloatsEqual(left.flux, right.flux)
        self.assertFloatsEqual(left.mask, right.mask)
        self.assertFloatsEqual(left.sky, right.sky)
        self.assertFloatsEqual(left.covar, right.covar)
        self.assertFloatsEqual(left.covar2, right.covar2)
        self.assertFlagsEqual(left.flags, right.flags)
        self.assertMetadataEqual(left.metadata, right.metadata)
        self.assertFluxTableEqual(left.fluxTable, right.fluxTable)

    def testBasic(self):
        """Test basic functionality

        PfsTargetSpectra behaves like a ``Dict[Target, PfsFiberArray]``.
        """
        num = 5
        spectra = self.makePfsTargetSpectra(list(range(num)))
        self.assertEqual(len(spectra), num)
        for target, spectrum in zip(spectra, spectra.values()):
            self.assertIn(target, spectra)
            self.assertIn(target.objId, spectra)
            self.assertIn((target.catId, target.objId), spectra)

            # Lookup by Target
            self.assertTargetEqual(target, spectra[target].target)
            self.assertPfsFiberArrayEqual(spectrum, spectra[target])

            # Lookup by objId
            self.assertTargetEqual(target, spectra[target.objId].target)
            self.assertPfsFiberArrayEqual(spectrum, spectra[target.objId])

            # Lookup by catId, objId
            self.assertTargetEqual(target, spectra[target.catId, target.objId].target)
            self.assertPfsFiberArrayEqual(spectrum, spectra[target.catId, target.objId])

            # Lookup by (catId, objId)
            self.assertTargetEqual(target, spectra[(target.catId, target.objId)].target)
            self.assertPfsFiberArrayEqual(spectrum, spectra[(target.catId, target.objId)])

        target = Target(catId=-123, tract=456, patch="123,456", objId=789)
        self.assertNotIn(target, spectra)
        self.assertNotIn(target.objId, spectra)
        self.assertNotIn((target.catId, target.objId), spectra)

    def testLookup(self):
        """Test lookup by various keys"""
        num = 5
        spectra = self.makePfsTargetSpectra(list(range(num)), catId=list(reversed(range(num))))
        spectraList = [spectra[target] for target in spectra]

        # Duplicate catId, objIds
        for ss in spectraList:
            ss.target.catId = 12345
            ss.target.objId = 67890
        with self.assertRaises(RuntimeError):
            PfsCoadd(spectraList)

        # Duplicate objIds
        objId = 67890
        for ss, ii in zip(spectraList, range(num)):
            ss.target.catId = ii
            ss.target.objId = objId
        test = PfsCoadd(spectraList)
        with self.assertRaises(KeyError):
            test[objId]  # objId is ambiguous
        with self.assertRaises(KeyError):
            objId in test
        test[(0, objId)]  # (catId, objId) is not ambiguous
        test[0, objId]  # (catId, objId) is not ambiguous

    def testIO(self):
        """Test I/O functionality"""
        spectra = self.makePfsTargetSpectra(list(range(123)))
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            spectra.writeFits(filename)
            copy = PfsCoadd.readFits(filename)
            self.assertPfsTargetSpectraEqual(copy, spectra)

            # Check that the DAMD_VER is present
            with astropy.io.fits.open(filename) as hdus:
                self.assertIn("DAMD_VER", hdus[0].header)

    def testDifferentLengths(self):
        """Test I/O when the spectra have different lengths"""
        spectra = self.makePfsTargetSpectra(list(range(123)), length=itertools.cycle([999, 1000, 1001]))
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            spectra.writeFits(filename)
            copy = PfsCoadd.readFits(filename)
            self.assertPfsTargetSpectraEqual(copy, spectra)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    setup_module(sys.modules["__main__"])
    unittest.main(failfast=True)
