import types

import numpy as np
from astropy.io.fits import BinTableHDU, Column

from .utils import calculatePfsVisitHash, wraparoundNVisit

__all__ = ("Observations",)


class Observations(types.SimpleNamespace):
    """A group of observations of a spectroscopic target

    Parameters
    ----------
    visit : `numpy.ndarray` of `int`
        Visit identifiers for each observation.
    arm : iterable of `str`
        Arm identifiers for each observation.
    spectrograph : `numpy.ndarray` of `int`
        Spectrograph identifier for each observation.
    pfsDesignId : `numpy.ndarray` of `int`
        Top-end design identifier for each observation.
    fiberId : `numpy.ndarray` of `int`
        Array of fiber identifiers for this object in each observation.
    pfiNominal : `numpy.ndarray` of `float`
        Array of nominal fiber positions (x,y) for this object in each
        observation.
    pfiCenter : `numpy.ndarray` of `float`
        Array of actual fiber positions (x,y) for this object in each
        observation.
    """
    def __init__(self, visit, arm, spectrograph, pfsDesignId, fiberId, pfiNominal, pfiCenter):
        self.visit = visit
        self.arm = arm
        self.spectrograph = spectrograph
        self.pfsDesignId = pfsDesignId
        self.fiberId = fiberId
        self.pfiNominal = pfiNominal
        self.pfiCenter = pfiCenter

        self.num = len(self.fiberId)
        self.validate()

    def __len__(self):
        """Number of observations"""
        return self.num

    def validate(self):
        """Validate that all arrays are of the expected shape"""
        assert len(self.visit) == self.num
        assert self.visit.shape == (self.num,)
        assert len(self.arm) == self.num
        assert self.spectrograph.shape == (self.num,)
        assert self.pfsDesignId.shape == (self.num,)
        assert self.fiberId.shape == (self.num,)
        assert self.pfiNominal.shape == (self.num, 2)
        assert self.pfiCenter.shape == (self.num, 2)

    def calculateVisitHash(self):
        """Calculate hash of the exposure inputs

        Returns
        -------
        hash : `int`
            Hash, truncated to 63 bits.
        """
        return calculatePfsVisitHash(set(self.visit))

    def getIdentity(self):
        """Return the identity of these observations

        Returns
        -------
        identity : `dict`
            Keyword-value pairs identifying these observations.
        """
        return dict(nVisit=wraparoundNVisit(len(set(self.visit))),
                    pfsVisitHash=self.calculateVisitHash(),
                    )

    @classmethod
    def fromFits(cls, fits):
        """Construct from a FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            Opened FITS file.

        Returns
        -------
        self : `Observations`
            Constructed observations.
        """
        hdu = fits["OBSERVATIONS"]
        kwargs = {col: hdu.data[col] for col in
                  ("visit", "arm", "spectrograph", "pfsDesignId", "fiberId", "pfiNominal", "pfiCenter")}
        return cls(**kwargs)

    def toFits(self, fits):
        """Write to a FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            Opened FITS file.
        """
        armLength = max(len(arm) for arm in self.arm)
        columns = [Column("visit", "J", array=self.visit),
                   Column("arm", f"{armLength}A", array=self.arm),
                   Column("spectrograph", "J", array=self.spectrograph),
                   Column("fiberId", "J", array=self.fiberId),
                   Column("pfsDesignId", "K", array=self.pfsDesignId),
                   Column("pfiNominal", "2E", array=self.pfiNominal),
                   Column("pfiCenter", "2E", array=self.pfiCenter),
                   ]
        hdu = BinTableHDU.from_columns(columns, name="OBSERVATIONS")
        fits.append(hdu)

    @classmethod
    def makeSingle(cls, identity, pfsConfig, fiberId):
        """Construct for a single observation

        Parameters
        ----------
        identity : `pfs.datamodel.Identity`
            Identity of the exposure.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration.
        fiberId : `int`
            Fiber identifier.

        Returns
        -------
        self : `Observations`
            Observations, consisting of a single exposure.
        """
        index = np.nonzero(pfsConfig.fiberId == fiberId)[0]
        if len(index) != 1:
            raise RuntimeError("Number of fibers in PfsConfig with fiberId = %d is not unity (%d)" %
                               (fiberId, len(index)))
        index = index[0]

        return cls(
            visit=np.array([identity.visit]),
            arm=[identity.arm],
            spectrograph=np.array([identity.spectrograph]),
            fiberId=np.array([fiberId]),
            pfsDesignId=np.array([pfsConfig.pfsDesignId]),
            pfiNominal=np.array([pfsConfig.pfiNominal[index]]),
            pfiCenter=np.array([pfsConfig.pfiCenter[index]]),
        )
