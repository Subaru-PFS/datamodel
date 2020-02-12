#
# A couple of useful functions to calculate PFS's SHA-1s
#
import hashlib
import numpy as np


def calculatePfsVisitHash(visits):
    """Calculate and return a hash from a list of visits

    Parameters
    ----------
    visits : `list` of `int`
        List of visit numbers.

    Returns
    -------
    hash : `int`
        Hash of the visits.
    """
    check = set(visits)
    if len(check) != len(visits):
        from collections import Counter
        counts = Counter(visits)
        raise ValueError(f"List of visits is not unique: {[vv for vv in counts if counts[vv] > 1]}")
    return createHash([str(vv).encode() for vv in sorted(visits)])


def calculate_pfsDesignId(fiberIds, ras, decs):
    """Calculate and return the hash from a set of lists of
    fiberId, ra, and dec"""

    if fiberIds is None:
        if ras is None and decs is None:
            return 0x0

        raise RuntimeError(
            "Either all or none of fiberId, ra, and dec may be None")

    if (ras == 0.0).all() and (decs == 0.0).all():  # don't check fiberIds as this may be lab data
        return 0x0

    return createHash(["%d %.0f %.0f" % (fiberId, ra, dec) for fiberId, ra, dec in zip(fiberIds, ras, decs)])


def createHash(*args):
    """Create a hash from the input strings truncated to 64 bits.

    Parameters
    ----------
    *args : `str`
        input string values used to generate the hash.

    Returns
    -------
    truncatedHash : `int`
        truncated hash value
    """
    m = hashlib.sha1()
    for l in list(args):
        m.update(str(l).encode())

    return int(m.hexdigest(), 16) & 0xffffffffffffffff


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def makeFullCovariance(covar):
    """Given a matrix of the diagonal part of the covariance matrix return a full matrix

    This is mostly useful for visualising the COVAR arrays in pfsArm/pfsObject files

    Specifically,
       covar[0, 0:]    Diagonal
       covar[1, 0:-1]  +-1 off diagonal
       covar[2, 0:-2]  +-2 off diagonal
    """
    nband = covar.shape[0]
    C = np.zeros(covar.shape[1]**2).reshape((covar.shape[1], covar.shape[1]))

    i = np.arange(C.shape[0])

    C[i, i] = covar[0]                     # diagonal
    for j in range(1, covar.shape[0]):     # bands near the diagonal
        C[i[j:], i[:-j]] = covar[j][0:-j]         # above the diagonal
        C[i[:-j], i[j:]] = covar[j][0:-j]         # below the diagonal

    return C


def astropyHeaderToDict(header):
    """Convert an astropy FITS header to a dict

    Comments are not preserved, nor are ``COMMENT`` or ``HISTORY`` cards.

    Parameters
    ----------
    header : `astropy.io.fits.Header`
        FITS header.

    Returns
    -------
    metadata : `dict`
        FITS header keywords and values.
    """
    return {key: value for key, value in header.items() if key not in set(("HISTORY", "COMMENT"))}


def astropyHeaderFromDict(metadata):
    """Convert a dict to an astropy FITS header

    Parameters
    ----------
    metadata : `dict`
        FITS header keywords and values.

    Returns
    -------
    header : `astropy.io.fits.Header`
        FITS header.
    """
    import astropy.io.fits
    header = astropy.io.fits.Header()
    for key, value in metadata.items():
        if len(key) > 8 and not key.startswith("HIERARCH"):
            key = "HIERARCH " + key
        header.append((key, value))
    return header


def wraparoundNVisit(nVisit):
    """Wraparound number of visits to acceptable range (0-999)

    Parameters
    ----------
    nVisit : `int`
        number of visits

    Returns
    -------
    nVisit_wrapped : `int`
        wraparound number of visits  
    """
    return nVisit % 1000
