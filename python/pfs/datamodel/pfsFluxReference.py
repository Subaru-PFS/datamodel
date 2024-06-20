import os
import re
import numpy as np

from .utils import astropyHeaderToDict, astropyHeaderFromDict
from .masks import MaskHelper
from .target import Target
from .identity import Identity
from .wavelengthArray import WavelengthArray

__all__ = ["PfsFluxReference"]


class PfsFluxReference:
    """Reference spectra for flux calibration.

    This is a set of synthetic spectra fitted to those spectra
    in a single PfsMerged that are marked FLUXSTD.
    Conceptually, LSFs are deconvolved from the observed spectra,
    effects of the atmosphere and the camera system are removed,
    yet galactic extinction is still kept in effect.

    Parameters
    ----------
    identity : `pfs.datamodel.Identity`
        Identity of the data.
    fiberId : `numpy.ndarray` of `int`
        Fiber identifiers for each spectrum.
    wavelength : `pfs.datamodel.wavelengthArray.WavelengthArray`
        Array of wavelengths common to all spectra.
    flux : `numpy.ndarray` of `float`
        Array of fluxes for each spectrum.
    metadata : `dict`
        Keyword-value pairs for the FITS header.
    fitFlag : `numpy.ndarray` of `int`
        Flag for each spectrum, indicating how fitting went.
    fitFlagNames : `pfs.datamodel.MaskHelper`
        Helper for dealing with symbolic names for ``fitFlag`` values.
    fitParams : `numpy.ndarray`
        Structured array. Fit parameters for each spectrum.
        Typical fields include:
          - teff
            Effective temperature in K.
          - logg
            Surface gravity in Log(g/cm/s^2).
          - m
            Metalicity in M/H.
          - alpha
            Alpha-elements abundance in alpha/Fe.
        There may be other fields such as chi^2 of the fit.
    """

    filenameFormat = "pfsFluxReference-%(visit)06d.fits"
    """Format for filename (`str`)

    Should include formatting directives for the ``identity`` dict.
    """

    filenameRegex = r"^pfsFluxReference-(\d{6})\.fits.*$"
    """Regex for extracting dataId from filename (`str`)

    Should capture the necessary values for the ``identity`` dict.
    """

    filenameKeys = [("visit", int)]
    """Key name and type (`list` of `tuple` of `str` and `type`)

    Keys should be in the same order as for the regex.
    """

    def __init__(self, identity, fiberId, wavelength, flux, metadata, fitFlag, fitFlagNames, fitParams):
        self.identity = identity
        self.fiberId = fiberId
        self.wavelength = wavelength
        self.flux = flux
        self.metadata = metadata
        self.fitFlag = fitFlag
        self.fitFlagNames = fitFlagNames
        self.fitParams = fitParams

        self.numSpectra, self.length = flux.shape

        self.validate()

    def validate(self):
        """Validate that all the arrays are of the expected shape"""
        assert self.fiberId.shape == (self.numSpectra,)
        assert self.wavelength.shape == (self.length,)
        assert self.flux.shape == (self.numSpectra, self.length)
        assert self.fitFlag.shape == (self.numSpectra,)
        assert self.fitParams.shape == (self.numSpectra,)

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
        kwargs = {name: getattr(self, name) for name in
                  ("identity", "wavelength", "metadata", "fitFlagNames")}
        kwargs.update(**{name: getattr(self, name)[logical] for
                         name in ("fiberId", "flux", "fitFlag", "fitParams")})
        return type(self)(**kwargs)

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
        filename : `str` or `file`
            Filename of FITS file.

        Returns
        -------
        self : ``cls``
            Constructed instance, from FITS file.
        """
        import astropy.io.fits

        def fits_getdata(hdulist, name, dtype=None, needHeader=False):
            """Get data array from hdulist.

            Use this sub-function to avoid using both hduName and HDUNAME
            in this function definition.
            """
            hdu = hdulist[name.upper()]
            arr = hdu.data
            if dtype is not None:
                arr = np.array(arr, dtype=dtype)
            if needHeader:
                return arr, hdu.header
            else:
                return arr

        data = {}

        with astropy.io.fits.open(filename) as fd:
            data["fiberId"] = fits_getdata(fd, "fiberId", dtype=np.int32)
            data["flux"], wcsHeader = fits_getdata(fd, "flux", dtype=np.float32, needHeader=True)
            data["fitFlag"], flagHeader = fits_getdata(fd, "fitFlag", dtype=np.int32, needHeader=True)
            data["fitParams"] = fits_getdata(fd, "fitParams")
            data["identity"] = Identity.fromFits(fd)
            data["metadata"] = astropyHeaderToDict(fd[0].header)
            data["wavelength"] = WavelengthArray.fromFitsHeader(
                wcsHeader, data["flux"].shape[1], dtype=float
            )
            data["fitFlagNames"] = MaskHelper.fromFitsHeader(flagHeader)

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
        self : `PfsFluxReference`
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
        filename : `str` or `file`
            Filename of FITS file.
        """
        # NOTE: When making any changes to this method that modify the output
        # format, increment the DAMD_VER header value and record the change in
        # the versions.txt file.
        import astropy.io.fits

        self.validate()
        fits = astropy.io.fits.HDUList()
        header = astropyHeaderFromDict(self.metadata)
        header['DAMD_VER'] = (1, "PfsFluxReference datamodel version")
        fits.append(astropy.io.fits.PrimaryHDU(header=header))

        def makeHDU(attr, dtype, header, hduType=astropy.io.fits.ImageHDU):
            """Get a member of ``self`` as a FITS HDU.

            Use this sub-function to avoid using both hduName and HDUNAME
            in this function definition.
            """
            hduName = attr.upper()
            data = getattr(self, attr)
            if dtype is not None:
                data = np.asarray(data, dtype=dtype)
            if isinstance(header, dict):
                header = astropy.io.fits.Header(header)
            return hduType(data, name=hduName, header=header)

        fits.append(makeHDU("fiberId", np.int32, None))
        fits.append(makeHDU("flux", np.float32, self.wavelength.toFitsHeader()))
        fits.append(makeHDU("fitFlag", np.int32, self.fitFlagNames.toFitsHeader()))
        fits.append(makeHDU("fitParams", None, None, hduType=astropy.io.fits.BinTableHDU))

        self.identity.toFits(fits)
        if hasattr(filename, "write"):
            # `filename` is already a file object.
            fits.writeto(filename)
        else:
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

    def extractFiber(self, SpectrumClass, pfsConfig, fiberId):
        """Extract a single fiber

        Pulls a single fiber out into a subclass of
        `pfs.datamodel.PfsSimpleSpectrum`.

        Parameters
        ----------
        SpectrumClass : `type`
            Subclass of `pfs.datamodel.PfsSimpleSpectrum` to which to export.
        pfsConfig : `pfs.datamodel.PfsConfig`
            PFS top-end configuration.
        fiberId : `int`
            Fiber ID to export.

        Returns
        -------
        spectrum : `SpectrumClass`
            Extracted spectrum.
        """
        ii = np.nonzero(self.fiberId == fiberId)[0]
        if len(ii) != 1:
            raise RuntimeError("Number of fibers in PfsFluxReference with fiberId = %d is not unity (%d)" %
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

        flux = self.flux[ii]

        flags = MaskHelper()
        mask = np.where(np.isfinite(flux), 0, flags.add("BAD")).astype(np.int32)

        return SpectrumClass(target, self.wavelength, flux, mask, flags)
