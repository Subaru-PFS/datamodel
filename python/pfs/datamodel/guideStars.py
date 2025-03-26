import numpy as np
import astropy.io.fits
from enum import IntFlag

__all__ = ["GuideStars"]


class GuideStars:
    """Table of guide star targets

    This will be the principle output of the ETS Shuffle tool.

    Parameters
    ----------
    objId : `numpy.ndarray` of `int64`
        The identifier of the guide star within the
        catalog.
    epoch : `numpy.chararray`
        reference epoch for guide star data.
    ra : `numpy.ndarray` of `float64`
        Right Ascension for each guide star, degrees.
    dec : `numpy.ndarray` of `float64`
        Declination for each guide star, degrees.
    pmRa : `numpy.ndarray` of `float32`
        Proper motion in direction of Right Ascension
        for each guide star, mas/year.
    pmDec : `numpy.ndarray` of `float32`
        Proper motion in direction of Declination
        for each guide star, mas/year.
    parallax : `numpy.ndarray` of `float32`
        parallax for each guide star, mas.
    magnitude : `numpy.ndarray` of `float32`
        parallax for each guide star, mas.
    passband : `numpy.chararray`
        passband for corresponding magnitude.
    color : `numpy.ndarray` of `float32`
        Gaia broadband BP-RP color for each guide star.
    agId : `numpy.ndarray` of `int32`
        Identifier for the AG camera that is expected to detect
        the corresponding guide star. This can have a value from 0 to 5 inclusive.
    agX : `numpy.ndarray' of `float32`
        The expected x-position of the guide star on the
        appropriate AG camera, pixels.
    agY : `numpy.ndarray' of `float32`
        The expected y-position of the guide star on the
        appropriate AG camera, pixels.
    telElev : `float`
        The telescope elevation, degrees.
    guideStarCatId : `int`
        The identifier for the catalogue from which the guide stars
        were taken.
    flag : `numpy.ndarray' of `int`
        The flag information for the guidestar catalog.
    """
    _hduName = "GUIDESTARS"  # HDU name to use

    def __init__(self, objId, epoch, ra, dec,
                 pmRa, pmDec,
                 parallax,
                 magnitude, passband,
                 color,
                 agId, agX, agY,
                 telElev, guideStarCatId, flag=None):

        flag = np.zeros_like(objId, dtype='int32') if flag is None else flag

        attributes = np.array(['objId', 'epoch', 'ra', 'dec', 'pmRa', 'pmDec', 'parallax',
                               'magnitude', 'passband', 'color', 'agId', 'agX', 'agY', 'flag'])

        dims = np.array([len(objId.shape),
                         len(epoch.shape),
                         len(ra.shape),
                         len(dec.shape),
                         len(pmRa.shape),
                         len(pmDec.shape),
                         len(parallax.shape),
                         len(magnitude.shape),
                         len(passband.shape),
                         len(color.shape),
                         len(agId.shape),
                         len(agX.shape),
                         len(agY.shape),
                         len(flag.shape),])

        if np.any(dims != 1):
            attributes[dims != 1]
            raise RuntimeError(f"The following attributes do not have"
                               f" the correct dimensions: {attributes[dims != 1].tolist()}")

        attributeErrorDict = {}
        for attribute, shape in zip(attributes, [objId.shape, epoch.shape, ra.shape, dec.shape,
                                    epoch.shape,
                                    pmRa.shape, pmDec.shape,
                                    parallax.shape,
                                    magnitude.shape, passband.shape,
                                    color.shape,
                                    agId.shape, agX.shape, agY.shape, flag.shape]):
            if (shape[0] != objId.shape[0]):
                attributeErrorDict[attribute] = shape

        if attributeErrorDict:
            raise RuntimeError(f"The following attributes do not have the shapes consistent with objId"
                               f" (whose shape is {objId.shape}) : "
                               f"{attributeErrorDict}")

        self.objId = objId
        self.epoch = epoch
        self.ra = ra
        self.dec = dec
        self.pmRa = pmRa
        self.pmDec = pmDec
        self.parallax = parallax
        self.magnitude = magnitude
        self.passband = passband
        self.color = color
        self.agId = agId
        self.agX = agX
        self.agY = agY
        self.epoch = epoch
        self.telElev = telElev
        self.guideStarCatId = guideStarCatId
        self.flag = flag

    def __len__(self):
        """Return number of elements"""
        return len(self.objId)

    def toFits(self, fits):
        """Write to a FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            Opened FITS file.
        """
        # NOTE: When making any changes to this method that modify the output
        # format, increment the DAMD_VER header value and record the change in
        # the versions.txt file.
        if len(self.passband) != 0:
            maxLength = max(len(ff) for ff in self.passband)
        else:
            maxLength = 1

        from astropy.io.fits import BinTableHDU, Column
        header = astropy.io.fits.Header()
        header['DAMD_VER'] = (1, "GuideStars datamodel version")
        header['TEL_ELEV'] = (self.telElev, "telescope elevation [degrees]")
        header['GS_CATID'] = (self.guideStarCatId,
                              "identifier of catalogue containing the guide stars")

        hdu = BinTableHDU.from_columns([
            Column("objId", "K", array=self.objId),
            Column("epoch", format='A7', array=self.epoch),
            Column("ra", "D", array=self.ra, unit='degrees'),
            Column("dec", "D", array=self.dec, unit='degrees'),
            Column("pmRa", "E", array=self.pmRa, unit='mas/year'),
            Column("pmDec", "E", array=self.pmDec, unit='mas/year'),
            Column("parallax", "E", array=self.parallax, unit='mas'),
            Column("magnitude", "E", array=self.magnitude),
            Column("passband", format='A%d' % maxLength, array=self.passband),
            Column("color", "E", array=self.color),
            Column("agId", "J", array=self.agId),
            Column("agX", "E", array=self.agX, unit='pixels'),
            Column("agY", "E", array=self.agY, unit='pixels'),
            Column("flag", "J", array=self.flag),
        ], header=header, name=self._hduName)
        fits.append(hdu)

    @classmethod
    def fromFits(cls, fits):
        """Construct from a FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            Opened FITS file.

        Returns
        -------
        self : `GuideStars`
            Constructed `GuideStars`.
        """
        hdu = fits[cls._hduName]
        flag = hdu.data["flag"] if "flag" in hdu.data.dtype.names else np.zeros_like(hdu.data["objId"])

        return cls(hdu.data["objId"].astype(np.int64),
                   hdu.data["epoch"],
                   hdu.data["ra"].astype(np.float64),
                   hdu.data["dec"].astype(np.float64),
                   hdu.data["pmRa"].astype(np.float32),
                   hdu.data["pmDec"].astype(np.float32),
                   hdu.data["parallax"].astype(np.float32),
                   hdu.data["magnitude"].astype(np.float32),
                   hdu.data["passband"],
                   hdu.data["color"].astype(np.float32),
                   hdu.data["agId"].astype(np.int32),
                   hdu.data["agX"].astype(np.float32),
                   hdu.data["agY"].astype(np.float32),
                   hdu.header['TEL_ELEV'],
                   hdu.header['GS_CATID'],
                   flag.astype(np.int32),
                   )

    @classmethod
    def empty(cls):
        """Construct an instance that has no entries.

        This is useful for situations where a GUIDESTARS
        HDU needs to be written out according to the
        datamodel.txt specification, but no meaningful
        guidestar information can be provided.
        """
        return cls(np.array([], dtype=np.int64),
                   np.array([], dtype='a7'),
                   np.array([], dtype=np.float64),
                   np.array([], dtype=np.float64),
                   np.array([], dtype=np.float32),
                   np.array([], dtype=np.float32),
                   np.array([], dtype=np.float32),
                   np.array([], dtype=np.float32),
                   np.array([], dtype='a1'),
                   np.array([], dtype=np.float32),
                   np.array([], dtype=np.int32),
                   np.array([], dtype=np.float32),
                   np.array([], dtype=np.float32),
                   0.0,
                   0,
                   np.array([], dtype=np.int32),
                   )


class AutoGuiderStarMask(IntFlag):
    """
    Represents a bitmask for guide star properties.

    Attributes:
        GAIA: Gaia DR3 catalog.
        HSC: HSC PDR3 catalog.
        PMRA: Proper motion RA is measured.
        PMRA_SIG: Proper motion RA measurement is significant (SNR>5).
        PMDEC: Proper motion Dec is measured.
        PMDEC_SIG: Proper motion Dec measurement is significant (SNR>5).
        PARA: Parallax is measured.
        PARA_SIG: Parallax measurement is significant (SNR>5).
        ASTROMETRIC: Astrometric excess noise is small (astrometric_excess_noise<1.0).
        ASTROMETRIC_SIG: Astrometric excess noise is significant (astrometric_excess_noise_sig>2.0).
        NON_BINARY: Not a binary system (RUWE<1.4).
        PHOTO_SIG: Photometric measurement is significant (SNR>5).
        GALAXY: Is a galaxy candidate.
    """
    GAIA = 0x00001
    HSC = 0x00002
    PMRA = 0x00004
    PMRA_SIG = 0x00008
    PMDEC = 0x00010
    PMDEC_SIG = 0x00020
    PARA = 0x00040
    PARA_SIG = 0x00080
    ASTROMETRIC = 0x00100
    ASTROMETRIC_SIG = 0x00200
    NON_BINARY = 0x00400
    PHOTO_SIG = 0x00800
    GALAXY = 0x01000


def decode_gs_bitmask(flags):
    """
    Decode the bitmask for guide stars.

    Parameters
    ----------
    flags : `numpy.ndarray` of `int16`
        The flag value to be decoded.

    Returns
    -------
    decoded_info: `dict`
        A dictionary containing the decoded info for each flag key.
    """
    decoded_info = {k.name: np.array([bool(flag & getattr(AutoGuiderStarMask, k.name)) for flag in flags])
                    for k in AutoGuiderStarMask}
    return decoded_info
