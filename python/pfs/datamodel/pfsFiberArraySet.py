import os
import re
from typing import Type

import numpy as np

from .utils import astropyHeaderToDict, astropyHeaderFromDict, inheritDocstrings
from .masks import MaskHelper
from .target import Target
from .observations import Observations
from .identity import Identity
from .pfsTable import PfsTable, EmptyTable

__all__ = ["PfsFiberArraySet"]


@inheritDocstrings
class PfsFiberArraySet:
    """A collection of spectra from a common source

    The collection may be from a single arm within a single spectrograph, or
    from the entire instrument. The only requirement is that the elements all
    have the same number of samples.

    Parameters
    ----------
    identity : `pfs.datamodel.Identity`
        Identity of the data.
    fiberId : `numpy.ndarray` of `int`
        Fiber identifiers for each spectrum.
    wavelength : `numpy.ndarray` of `float`
        Array of wavelengths for each spectrum.
    flux : `numpy.ndarray` of `float`
        Array of fluxes for each spectrum.
    mask : `numpy.ndarray` of `int`
        Array of mask pixels for each spectrum.
    sky : `numpy.ndarray` of `float`
        Array of sky values for each spectrum.
    norm : `numpy.ndarray` of `float`
        Normalisation for each spectrum.
    covar : `numpy.ndarray` of `float`
        Array of covariances for each spectrum.
    flags : `dict`
        Mapping of symbolic mask names to mask planes.
    metadata : `dict`
        Keyword-value pairs for the header.
    notes : `PfsTable`
        Reduction notes for each spectrum.
    """
    filenameFormat = None  # Subclasses must override
    """Format for filename (`str`)

    Should include formatting directives for the ``identity`` dict.
    """
    filenameRegex = None  # Subclasses must override
    """Regex for extracting dataId from filename (`str`)

    Should capture the necessary values for the ``identity`` dict.
    """
    filenameKeys = None  # Subclasses must override
    """Key name and type (`list` of `tuple` of `str` and `type`)

    Keys should be in the same order as for the regex.
    """

    NotesClass: Type[PfsTable] = EmptyTable  # Subclasses must override
    """Class for notes (`PfsTable` subclass)"""

    def __init__(
        self, identity, fiberId, wavelength, flux, mask, sky, norm, covar, flags, metadata, notes=None
    ):
        self.identity = identity
        self.fiberId = fiberId
        self.wavelength = wavelength
        self.flux = flux
        self.mask = mask
        self.sky = sky
        self.norm = norm
        self.covar = covar
        self.flags = flags
        self.metadata = metadata

        self.numSpectra = wavelength.shape[0]
        self.length = wavelength.shape[1]

        self.notes = notes if notes is not None else self.NotesClass.empty(self.numSpectra)
        self.validate()

    def validate(self):
        """Validate that all the arrays are of the expected shape"""
        assert self.fiberId.shape == (self.numSpectra,)
        assert self.wavelength.shape == (self.numSpectra, self.length)
        assert self.flux.shape == (self.numSpectra, self.length)
        assert self.mask.shape == (self.numSpectra, self.length)
        assert self.sky.shape == (self.numSpectra, self.length)
        assert self.covar.shape == (self.numSpectra, 3, self.length)
        assert len(self.notes) == self.numSpectra

    @property
    def variance(self):
        """Shortcut for variance"""
        return self.covar[:, 0, :]

    def __len__(self):
        """Return number of spectra"""
        return self.numSpectra

    def __str__(self):
        """Stringify"""
        return "%s{%d spectra of length %d}" % (self.__class__.__name__, self.numSpectra, self.length)

    def __iter__(self):
        """Iteration is unimplemented because it would be inefficient"""
        return NotImplementedError(f"Cannot iterate on {self.__class__.__name__}")

    def __getitem__(self, logical):
        """Sub-selection

        Parameters
        ----------
        logical : `numpy.ndarray` of `bool`
            Boolean array (of same length as ``self``) indicating which fibers
            to select.

        Returns
        -------
        new : ``type(self)``
            A new instance containing only the selected fibers.
        """
        kwargs = {name: getattr(self, name) for name in ("identity", "flags", "metadata")}
        kwargs.update(**{name: getattr(self, name)[logical] for
                         name in ("fiberId", "wavelength", "flux", "mask", "sky", "norm", "covar", "notes")})
        return type(self)(**kwargs)

    def select(self, pfsConfig=None, **kwargs):
        """Return an instance containing only the selected attributes

        Multiple attributes will be combined with ``AND``.

        Parameters
        ----------
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration.  Optional if the only selection is on spectrograph or fiberId
        fiberId : `int` (scalar or array_like), optional
            Fiber identifier to select.  pfsConfig may be omitted
        targetType : `TargetType` (scalar or array_like), optional
            Target type to select.
        fiberStatus : `FiberStatus` (scalar or array_like), optional
            Fiber status to select.
        catId : `int` (scalar or array_like), optional
            Catalog identifier to select.
        tract : `int` (scalar or array_like), optional
            Tract number to select.
        patch : `str` (scalar or array_like), optional
            Patch name to select.
        objId : `int` (scalar or array_like), optional
            Object identifier to select.
        spectrograph : `int` (scalar or array_like), optional
            Spectrograph number to select.  pfsConfig may be omitted

        Returns
        -------
        selected : ``type(self)``
            An instance containing only the selected attributes.
        """
        keys = set(kwargs)
        ll = np.ones(len(self), dtype=bool)
        for kw in ["fiberId", "spectrograph"]:    # no need for a pfsConfig
            if kw in kwargs:
                ll &= np.isin(getattr(self, kw), kwargs[kw])
                keys.discard(kw)

        if len(keys) == 0:
            return self[ll]

        if pfsConfig is None:
            raise RuntimeError(
                "You must provide a pfsConfig file for all selections except"
                " spectrograph=[...], fiberId=[...]" + ("; saw \"%s\"" % '", "'.join(keys)))

        selection = pfsConfig.getSelection(**kwargs)
        return self[np.isin(self.fiberId, pfsConfig.fiberId[selection])]

    @property
    def filename(self):
        """Filename, without directory"""
        return self.getFilename(self.identity)

    @classmethod
    def getFilename(cls, identity):
        """Calculate filename

        Parameters
        ----------
        identity : `pfs.datamodel.Identity`
            Identity of the data.

        Returns
        -------
        filename : `str`
            Filename, without directory.
        """
        return cls.filenameFormat % identity.getDict()

    @classmethod
    def _parseFilename(cls, path):
        """Parse filename to get the file's identity

        Uses the class attributes ``filenameRegex`` and ``filenameKeys`` to
        construct the identity from the filename.

        Parameters
        ----------
        path : `str`
            Path to the file of interest.

        Returns
        -------
        identity : `pfs.datamodel.Identity`
            Identity of the data of interest.
        """
        dirName, fileName = os.path.split(path)
        matches = re.search(cls.filenameRegex, fileName)
        if not matches:
            raise RuntimeError("Unable to parse filename: %s" % (fileName,))
        return Identity.fromDict({kk: tt(vv) for (kk, tt), vv in zip(cls.filenameKeys, matches.groups())})

    @classmethod
    def readFits(cls, filename):
        """Read from FITS file

        This API is intended for use by the LSST data butler, which handles
        translating the desired identity into a filename.

        Parameters
        ----------
        filename : `str`
            Filename of FITS file.

        Returns
        -------
        self : ``cls``
            Constructed instance, from FITS file.
        """
        data = {}
        import astropy.io.fits
        with astropy.io.fits.open(filename, memmap=False) as fd:
            data["metadata"] = astropyHeaderToDict(fd[0].header)
            data["fiberId"] = fd["FIBERID"].data.astype(np.int32)

            # Wavelength: if there's only a single array, it applies to all spectra
            wavelength = fd["WAVELENGTH"].data.astype(np.float64)
            if len(wavelength.shape) == 1:
                wavelength = np.tile(wavelength, (len(data["fiberId"]), 1))
            data["wavelength"] = wavelength

            for attr, dtype in (("flux", np.float32),
                                ("mask", np.uint32),
                                ("sky", np.float32),
                                # "norm" is treated separately, for backwards-compatibility
                                ("covar", np.float32)
                                ):
                hduName = attr.upper()
                data[attr] = fd[hduName].data.astype(dtype)
            version = fd[0].header["DAMD_VER"]
            if version >= 2:
                data["norm"] = fd["NORM"].data.astype(np.float32)
            else:
                data["norm"] = np.ones_like(data["flux"], dtype=np.float32)
            data["identity"] = Identity.fromFits(fd)
            if version >= 3:
                data["notes"] = cls.NotesClass.readHdu(fd)

        data["flags"] = MaskHelper.fromFitsHeader(data["metadata"])
        return cls(**data)

    @classmethod
    def read(cls, identity, dirName="."):
        """Read file given an identity

        This API is intended for use by science users, as it allows selection
        of the correct file by identity (e.g., visit, arm, spectrograph),
        without knowing the file naming convention.

        Parameters
        ----------
        identity : `pfs.datamodel.Identity`
            Identification of the data of interest.
        dirName : `str`, optional
            Directory from which to read.

        Returns
        -------
        self : `PfsFiberArraySet`
            Spectra read from file.
        """
        filename = os.path.join(dirName, cls.getFilename(identity))
        return cls.readFits(filename)

    def writeFits(self, filename):
        """Write to FITS file

        This API is intended for use by the LSST data butler, which handles
        translating the desired identity into a filename.

        Parameters
        ----------
        filename : `str`
            Filename of FITS file.
        """
        # NOTE: When making any changes to this method that modify the output
        # format, increment the DAMD_VER header value and record the change in
        # the versions.txt file.
        self.validate()
        import astropy.io.fits
        fits = astropy.io.fits.HDUList()
        header = self.metadata.copy()
        header.update(self.flags.toFitsHeader())
        header = astropyHeaderFromDict(header)
        header['DAMD_VER'] = (4, "PfsFiberArraySet datamodel version")
        fits.append(astropy.io.fits.PrimaryHDU(header=header))
        fits.append(astropy.io.fits.ImageHDU(self.fiberId.astype(np.int32), name="FIBERID"))

        # Wavelength: we can write a single array if they're all the same
        wavelength = self.wavelength[0]
        if not all(np.all(wl == wavelength) for wl in self.wavelength[1:]):
            wavelength = self.wavelength
        fits.append(astropy.io.fits.ImageHDU(data=wavelength.astype(np.float64), name="WAVELENGTH"))

        for attr, dtype in (
            ("flux", np.float32),
            ("mask", np.uint32),
            ("sky", np.float32),
            ("norm", np.float32),
            ("covar", np.float32)
        ):
            hduName = attr.upper()
            data = getattr(self, attr)
            HduClass = astropy.io.fits.ImageHDU
            if np.issubdtype(dtype, np.integer):
                HduClass = astropy.io.fits.CompImageHDU
            fits.append(HduClass(data.astype(dtype), name=hduName))

        self.identity.toFits(fits)
        self.notes.writeHdu(fits)
        with open(filename, "wb") as fd:
            fits.writeto(fd)

    def write(self, dirName="."):
        """Write to file

        This API is intended for use by science users, as it allows setting the
        correct filename from parameters that make sense, such as which
        exposure, spectrograph, etc.

        Parameters
        ----------
        dirName : `str`, optional
            Directory to which to write.
        """
        filename = os.path.join(dirName, self.filename)
        return self.writeFits(filename)

    @classmethod
    def fromMerge(cls, spectraList, metadata=None):
        """Construct from merging multiple spectra

        The merged spectra are sorted by fiberId.

        Parameters
        ----------
        spectraList : iterable of `PfsFiberArraySet`
            Spectra to combine.
        metadata : `dict` (`str`: POD), optional
            Keyword-value pairs for the header.

        Returns
        -------
        self : `PfsFiberArraySet`
            Merged spectra.
        """
        num = sum(len(ss) for ss in spectraList)
        length = set([ss.length for ss in spectraList])
        if len(length) != 1:
            raise RuntimeError("Multiple lengths when merging spectra: %s" % (length,))
        length = length.pop()
        fiberId = np.empty(num, dtype=int)
        wavelength = np.empty((num, length), dtype=float)
        flux = np.empty((num, length), dtype=np.float32)
        mask = np.empty((num, length), dtype=np.int32)
        sky = np.empty((num, length), dtype=np.float32)
        norm = np.empty((num, length), dtype=np.float32)
        covar = np.empty((num, 3, length), dtype=np.float32)
        notes = cls.NotesClass.empty(num)
        index = 0
        for ss in spectraList:
            select = slice(index, index + len(ss))
            fiberId[select] = ss.fiberId
            wavelength[select] = ss.wavelength
            flux[select] = ss.flux
            mask[select] = ss.mask
            sky[select] = ss.sky
            norm[select] = ss.norm
            covar[select] = ss.covar
            notes[select] = ss.notes
            index += len(ss)

        indices = np.argsort(fiberId)
        identity = Identity.fromMerge([ss.identity for ss in spectraList])
        flags = MaskHelper.fromMerge(list(ss.flags for ss in spectraList))
        return cls(identity, fiberId[indices], wavelength[indices], flux[indices], mask[indices],
                   sky[indices], norm[indices], covar[indices], flags, metadata if metadata else {}, notes)

    def extractFiber(self, FiberArrayClass, pfsConfig, fiberId):
        """Extract a single fiber

        Pulls a single fiber out into a subclass of `pfs.datamodel.PfsFiberArray`.

        Parameters
        ----------
        FiberArrayClass : `type`
            Subclass of `pfs.datamodel.PfsFiberArray` to which to export.
        pfsConfig : `pfs.datamodel.PfsConfig`
            PFS top-end configuration.
        fiberId : `int`
            Fiber ID to export.

        Returns
        -------
        spectrum : ``SpectrumClass``
            Extracted spectrum.
        """
        ii = np.nonzero(self.fiberId == fiberId)[0]
        if len(ii) != 1:
            raise RuntimeError("Number of fibers in PfsFiberArraySet with fiberId = %d is not unity (%d)" %
                               (fiberId, len(ii)))
        ii = ii[0]
        jj = np.nonzero(pfsConfig.fiberId == fiberId)[0]
        if len(jj) != 1:
            raise RuntimeError("Number of fibers in PfsConfig with fiberId = %d is not unity (%d)" %
                               (fiberId, len(jj)))
        jj = jj[0]

        fiberFlux = dict(zip(pfsConfig.filterNames[jj], pfsConfig.fiberFlux[jj]))
        target = Target(pfsConfig.catId[jj], pfsConfig.tract[jj], pfsConfig.patch[jj],
                        pfsConfig.objId[jj], pfsConfig.ra[jj], pfsConfig.dec[jj],
                        pfsConfig.targetType[jj], fiberFlux)
        obs = Observations.makeSingle(self.identity, pfsConfig, fiberId)

        covar = np.zeros((3, self.length), dtype=self.covar.dtype)
        with np.errstate(divide="ignore", invalid="ignore"):
            flux = self.flux[ii]/self.norm[ii]
            sky = self.sky[ii]/self.norm[ii]
            # XXX not dealing with covariance properly.
            covar[:] = self.covar[ii]/self.norm[ii]**2
        covar2 = np.zeros((1, 1), dtype=self.covar.dtype)
        return FiberArrayClass(target, obs, self.wavelength[ii], flux, self.mask[ii], sky,
                               covar, covar2, self.flags)
