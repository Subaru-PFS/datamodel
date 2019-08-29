import os
import re
import numpy as np
from scipy.interpolate import interp1d

from .utils import astropyHeaderToDict, astropyHeaderFromDict
from .masks import MaskHelper
from .target import TargetData, TargetObservations

__all__ = ["PfsSpectra"]


class PfsSpectra:
    """A collection of spectra from a common source

    The collection may be from a single arm within a single spectrograph, or
    from the entire instrument. The only requirement is that the elements all
    have the same number of samples.

    Parameters
    ----------
    identity : `dict`
        Keyword-value pairs identifying the data of interest. Common keywords
        include ``visit``, ``pfsDesignId``, ``spectrograph``, ``arm``.
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
    covar : `numpy.ndarray` of `float`
        Array of covariances for each spectrum.
    flags : `dict`
        Mapping of symbolic mask names to mask planes.
    metadata : `dict`
        Keyword-value pairs for the header.
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

    def __init__(self, identity, fiberId, wavelength, flux, mask, sky, covar, flags, metadata):
        self.identity = identity
        self.fiberId = fiberId
        self.wavelength = wavelength
        self.flux = flux
        self.mask = mask
        self.sky = sky
        self.covar = covar
        self.flags = flags
        self.metadata = metadata

        self.numSpectra = wavelength.shape[0]
        self.length = wavelength.shape[1]

    def validate(self):
        """Validate that all the arrays are of the expected shape"""
        assert self.fiberId.shape == (self.numSpectra,)
        assert self.wavelength.shape == (self.numSpectra, self.length)
        assert self.flux.shape == (self.numSpectra, self.length)
        assert self.mask.shape == (self.numSpectra, self.length)
        assert self.sky.shape == (self.numSpectra, self.length)
        assert self.covar.shape == (self.numSpectra, 3, self.length)

    def __len__(self):
        """Return number of spectra"""
        return self.numSpectra

    def __str__(self):
        """Stringify"""
        return "%s{%d spectra of length %d}" % (self.__class__.__name__, self.numSpectra, self.length)

    @property
    def filename(self):
        """Filename, without directory"""
        return self.getFilename(self.identity)

    @classmethod
    def getFilename(cls, identity):
        """Calculate filename

        Parameters
        ----------
        identity : `dict`
            Keyword-value pairs identifying the data of interest. Common keywords
            include ``visit``, ``pfsDesignId``, ``spectrograph``, ``arm``.

        Returns
        -------
        filename : `str`
            Filename, without directory.
        """
        return cls.filenameFormat % identity

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
        identity : `dict`
            Keyword-value pairs identifying the data of interest.
        """
        dirName, fileName = os.path.split(path)
        matches = re.search(cls.filenameRegex, fileName)
        if not matches:
            raise RuntimeError("Unable to parse filename: %s" % (fileName,))
        return {kk: tt(vv) for (kk, tt), vv in zip(cls.filenameKeys, matches.groups())}

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
        with astropy.io.fits.open(filename) as fd:
            data["metadata"] = astropyHeaderToDict(fd[0].header)
            for attr in ("fiberId", "wavelength", "flux", "mask", "sky", "covar"):
                hduName = attr.upper()
                data[attr] = fd[hduName].data
            data["identity"] = {nn: fd["CONFIG"].data.field(nn)[0] for nn in fd["CONFIG"].data.names}

        data["flags"] = MaskHelper.fromFitsHeader(data["metadata"])
        return cls(**data)

    @classmethod
    def read(cls, identity, dirName="."):
        """Read file given an identity

        This API is intended for use by science users, as it allows selection
        of the correct file from parameters that make sense, such as which
        exposure, spectrograph, etc.

        Parameters
        ----------
        identity : `dict`
            Keyword-value pairs identifying the data of interest. Common keywords
            include ``visit``, ``pfsDesignId``, ``spectrograph``, ``arm``.
        dirName : `str`, optional
            Directory from which to read.

        Returns
        -------
        self : `PfsSpectra`
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
        self.validate()
        import astropy.io.fits
        fits = astropy.io.fits.HDUList()
        header = self.metadata.copy()
        header.update(self.flags.toFitsHeader())
        fits.append(astropy.io.fits.PrimaryHDU(header=astropyHeaderFromDict(header)))
        for attr in ("fiberId", "wavelength", "flux", "mask", "sky", "covar"):
            hduName = attr.upper()
            data = getattr(self, attr)
            fits.append(astropy.io.fits.ImageHDU(data, name=hduName))

        # CONFIG table
        def columnFormat(data):
            """Return appropriate column format for some data"""
            if isinstance(data, str):
                return f"{len(data)}A"
            if isinstance(data, (float, np.float32, np.float64)):
                return "E"  # Don't expect to need double precision
            if isinstance(data, (int, np.int32, np.int64)):
                return "K"  # Use 64 bits because space is not a concern, and we need to support pfsDesignId
            raise TypeError(f"Unable to determine suitable column format for {data}")

        columns = [astropy.io.fits.Column(name=kk, format=columnFormat(vv), array=[vv]) for
                   kk, vv in self.identity.items()]
        fits.append(astropy.io.fits.BinTableHDU.from_columns(columns, name="CONFIG"))

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
    def fromMerge(cls, identityKeys, spectraList):
        """Construct from merging multiple spectra

        Parameters
        ----------
        identityKeys : iterable of `str`
            Keys to select from the input spectra's ``identity`` for the
            combined spectra's ``identity``.
        spectraList : iterable of `PfsSpectra`
            Spectra to combine.

        Returns
        -------
        self : `PfsSpectra`
            Merged spectra.
        """
        num = sum(len(ss) for ss in spectraList)
        length = set([ss.length for ss in spectraList])
        if len(length) != 1:
            raise RuntimeError("Multiple lengths when merging spectra: %s" % (length,))
        length = length.pop()
        identity = {kk: spectraList[0].identity[kk]for kk in identityKeys}
        fiberId = np.empty(num, dtype=int)
        wavelength = np.empty((num, length), dtype=float)
        flux = np.empty((num, length), dtype=float)
        mask = np.empty((num, length), dtype=int)
        sky = np.empty((num, length), dtype=float)
        covar = np.empty((num, 3, length), dtype=float)
        index = 0
        for ss in spectraList:
            select = slice(index, index + len(ss))
            fiberId[select] = ss.fiberId
            wavelength[select] = ss.wavelength
            flux[select] = ss.flux
            mask[select] = ss.mask
            sky[select] = ss.sky
            covar[select] = ss.covar
            index += len(ss)
        metadata = {}
        flags = MaskHelper.fromMerge(list(ss.flags for ss in spectraList))
        return cls(identity, fiberId, wavelength, flux, mask, sky, covar, flags, metadata)

    def plot(self, fiberId=None, usePixels=False, ignorePixelMask=0x0, show=True):
        """Plot the spectra

        Parameters
        ----------
        fiberId : iterable of `int`, optional
            Fibers to plot, or ``None`` to plot all.
        usePixels : `bool`, optional
            Plot as a function of pixel index, rather than wavelength?
        ignorePixelMask : `int`, optional
            Mask to apply to flux pixels.
        show : `bool`, optional
            Show the plot?

        Returns
        -------
        figure : `matplotlib.Figure`
            Figure containing the plot.
        axes : `matplotlib.Axes`
            Axes containing the plot.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm

        if fiberId is None:
            fiberId = self.fiberId
        if usePixels:
            wavelength = np.arange(self.length)
            xLabel = "Pixel"
        else:
            wavelength = self.wavelength
            xLabel = "Wavelength (nm)"

        figure, axes = plt.subplots()

        colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(fiberId)))
        for ff, cc in zip(fiberId, colors):
            index = np.where(self.fiberId == ff)[0]
            good = (self.mask[index] & ignorePixelMask) == 0
            axes.plot(wavelength[index][good], self.flux[index][good], ls="solid", color=cc)

        axes.set_xlabel(xLabel)
        axes.set_ylabel("Flux")

        if show:
            figure.show()
        return figure, axes

    def resample(self, wavelength, fiberId=None):
        """Construct a new PfsSpectra resampled to a common wavelength vector

        Parameters
        ----------
        wavelength : `numpy.ndarray` of `float`
            New wavelength values (nm).
        fiberId : `numpy.ndarray` of int, optional
            Fibers to resample. If ``None``, resample all fibers.

        Returns
        -------
        result : `PfsSpectra`
            Resampled spectra.
        """
        if fiberId is None:
            fiberId = self.fiberId

        # how to interpolate
        kwargs = dict(kind='linear',
                      bounds_error=False,
                      fill_value=0,
                      copy=True,
                      assume_sorted=True,
                      )

        numSpectra = len(fiberId)
        numSamples = len(wavelength)
        flux = np.empty((numSpectra, numSamples), dtype=self.flux.dtype)
        mask = np.empty((numSpectra, numSamples), dtype=self.mask.dtype)
        sky = np.empty((numSpectra, numSamples), dtype=self.sky.dtype)
        covar = np.zeros((numSpectra, 3, numSamples), dtype=self.covar.dtype)

        for ii, ff in enumerate(fiberId):
            jj = np.argwhere(self.fiberId == ff)[0][0]
            flux[ii] = interp1d(self.wavelength[jj], self.flux[jj], **kwargs)(wavelength)
            sky[ii] = interp1d(self.wavelength[jj], self.sky[jj], **kwargs)(wavelength)
            kwargs.update(fill_value=np.inf)
            # XXX dropping covariance on the floor: just doing the variance for now
            covar[ii][0] = interp1d(self.wavelength[jj], self.covar[jj][0], **kwargs)(wavelength)
            kwargs.update(fill_value=self.flags["NO_DATA"], kind='nearest')
            mask[ii] = interp1d(self.wavelength[jj], self.mask[jj],
                                **kwargs)(wavelength).astype(self.mask.dtype)

        return type(self)(self.identity, fiberId, np.concatenate([[wavelength]]*numSpectra),
                          flux, mask, sky, covar, self.flags, self.metadata)

    def extractFiber(self, SpectrumClass, pfsConfig, fiberId):
        """Extract a single fiber

        Pulls a single fiber out into a ``PfsSpectrum``.

        Parameters
        ----------
        SpectrumClass : `type`
            Subclass of `pfs.datamodel.PfsSpectrum` to which to export.
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
            raise RuntimeError("Number of fibers in PfsSpectra with fiberId = %d is not unity (%d)" %
                               (fiberId, len(ii)))
        ii = ii[0]
        jj = np.nonzero(pfsConfig.fiberId == fiberId)[0]
        if len(jj) != 1:
            raise RuntimeError("Number of fibers in PfsConfig with fiberId = %d is not unity (%d)" %
                               (fiberId, len(jj)))
        jj = jj[0]

        fiberMag = dict(zip(pfsConfig.filterNames[jj], pfsConfig.fiberMag[jj]))
        target = TargetData(pfsConfig.catId[jj], pfsConfig.tract[jj], pfsConfig.patch[jj],
                            pfsConfig.objId[jj], pfsConfig.ra[jj], pfsConfig.dec[jj],
                            pfsConfig.targetType[jj], fiberMag)
        obs = TargetObservations([self.identity], np.array([fiberId]), np.array([pfsConfig.pfiNominal[jj]]),
                                 np.array([pfsConfig.pfiCenter[jj]]))
        # XXX not dealing with covariance properly.
        covar = np.zeros((3, self.length), dtype=self.covar.dtype)
        covar[:] = self.covar[ii]
        covar2 = np.zeros((1, 1), dtype=self.covar.dtype)
        return SpectrumClass(target, obs, self.wavelength[ii], self.flux[ii], self.mask[ii], self.sky[ii],
                             covar, covar2, self.flags)
