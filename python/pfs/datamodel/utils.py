#
# A couple of useful functions to calculate PFS's SHA-1s
#
import hashlib
import inspect
import functools
from typing import Optional, Set, Type, TypeVar, Union
from logging import Logger
import datetime

import numpy as np
import astropy.io.fits

__all__ = ("calculatePfsVisitHash", "createHash", "astropyHeaderToDict", "astropyHeaderFromDict",
           "wraparoundNVisit", "inheritDocstrings", "checkHeaderKeyword", "convertToIso8601Utc")


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


def calculate_pfsDesignId(fiberIds, ras, decs, variant=0):
    """Calculate and return the hash for a design.

    We hash all the (fiberId, ra, dec) tuples, with the coordinates rounded to 1 arcsec.

    Parameters
    ----------
    fiberIds: `list` of `int`
       The fiber ids for the objects
    ras, decs : `list` of `float`
       The sky coordinates of the objects. Rounded to 1 arcsec.
    variant: `int`, optional
       If non-0, added to the hash input

    Returns
    -------
    sha : `int`
       hash of all inputs, truncated to 63-bits.
    """
    if fiberIds is None:
        if ras is None and decs is None:
            return 0x0

        raise RuntimeError(
            "Either all or none of fiberId, ra, and dec may be None")

    if (ras == 0.0).all() and (decs == 0.0).all():  # don't check fiberIds as this may be lab data
        return 0x0

    def _roundToArcsec(d):
        if np.isnan(d):
            return d  # Just return nan in this case
        return int(d*3600.0 + 0.5)/3600.0

    # Regardless of the arcsec rounding, we need to choose a precision for the string respresentation.
    # If datamodel.txt phrasing were a little sloppier, we could just use integer arcseconds.
    hashParts = ["%d %0.6f %0.6f" % (fiberId, _roundToArcsec(ra), _roundToArcsec(dec))
                 for fiberId, ra, dec in zip(fiberIds, ras, decs)]

    # Also hash variant id. For backward compatibility, do *not* change the hash if variant=0, which
    # indicates that we are not a variant.
    if variant != 0:
        hashParts.append(str(variant))

    return createHash(hashParts)


def createHash(*args):
    """Create a hash from the input strings truncated to 63 bits.

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
    for ss in list(args):
        m.update(str(ss).encode())

    return int(m.hexdigest(), 16) & 0x7fffffffffffffff


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def makeFullCovariance(covar):
    """Given a matrix of the diagonal part of the covariance matrix return a full matrix

    This is mostly useful for visualising the COVAR arrays in pfsArm/pfsObject files

    Specifically,
       covar[0, 0:]    Diagonal
       covar[1, 0:-1]  +-1 off diagonal
       covar[2, 0:-2]  +-2 off diagonal
    """
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

    ``COMMENT`` and ``HISTORY`` cards are not preserved.

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
        if key in ("HISTORY", "COMMENT"):
            continue
        if len(key) > 8 and not key.startswith("HIERARCH"):
            key = "HIERARCH " + key
        if isinstance(value, str) and key.startswith("HIERARCH") and len(key) + len(value) >= 77:
            # astropy.io.fits.Header doesn't handle long HIERARCH strings
            continue
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


def inheritDocstrings(cls):
    """Class decorator to inherit docstrings from base classes

    Docstrings are copied, changing any instances of the base class name to
    the subclass name. The docstring is inserted into a new method that calls
    the parent class implementation.
    """
    baseClasses = cls.__mro__[1:-1]  # Excluding the subclass and 'object'
    for name, attr in inspect.getmembers(cls):
        if not inspect.isfunction(attr) and not inspect.ismethod(attr):
            continue
        if name in cls.__dict__:
            # Has its own implementation, and therefore own docstring.
            continue
        if name.startswith("_"):
            # Private method: user shouldn't be looking here anyway
            continue
        for base in baseClasses:
            if hasattr(base, name):
                impl = getattr(base, name)
                break
        else:
            raise RuntimeError(f"Unable to find implementation for {cls.__name__}.{name}")
        if not hasattr(impl, "__doc__"):
            # No docstring to copy
            continue
        docstring = impl.__doc__
        if not any(base.__name__ in docstring for base in baseClasses):
            # No need to fix anything
            continue
        for base in baseClasses:
            docstring = docstring.replace(base.__name__, cls.__name__)

        # Modify the docstring in an override
        method = base.__dict__[impl.__name__]  # No binding to the base class
        if isinstance(method, classmethod):
            override = functools.partial(method.__func__)
            override.__doc__ = docstring
            override = classmethod(override)  # Now bind to the subclass
        else:
            override = functools.partial(method)
            override.__doc__ = docstring

        setattr(cls, name, override)

    return cls


def combineArms(arms):
    """Combine and order input arm identifications

    Re-orders the input arms such that the 'b' arm
    is rendered first, the 'r' arm next, followed
    by 'm' and 'n' arms. For example, the input {'rnb'}
    will be outputted as 'brn'.

    As this method may be used multiple times during the
    merging process, this needs to handle not just
    sets of single arm characters, eg., {'b', 'n'} but
    also pre-combined sets eg. {'brn', 'nb'}

    Parameters
    ----------
    arms : set[`str`]
        set of arm specifications, eg {'b', 'r'} or {'brn', 'b'}

    Returns
    -------
    ordered_arms : `str`
        ordered arm specification compressed into a string, eg 'br'
    """
    combinedSet = set()
    for entry in arms:
        for arm in entry:
            combinedSet.add(arm)

    armOrder = dict(b=1, r=2, m=3, n=4)
    return "".join(sorted(combinedSet, key=lambda aa: armOrder[aa]))


def checkHeaderKeyword(
    header: astropy.io.fits.Header,
    keyword: str,
    expected: Union[bool, str, int, float],
    comment: str = None,
    allowFix: bool = False,
    log: Optional[Logger] = None,
) -> bool:
    """Check that the keyword exists in the header

    Parameters
    ----------
    header : `astropy.io.fits.Header`
        FITS header; may be updated if the ``keyword`` is not present.
    keyword : `str`
        Keyword of interest
    expected : `bool`, `str`, `int` or `float`
        Expected value of ``keyword``.
    comment : `str`
        Comment to give keyword in FITS header if updating.
    allowFix : `bool`, optional
        Allow the header to be updated with the expected value?
    log : `Logger`, optional
        Logger to use, or ``None`` for no logging.

    Returns
    -------
    modified : `bool`
        Did we modify the header?

    Raises
    ------
    ValueError
        If the keyword is missing or has the incorrect value, and
        ``allowFix=False``.
    """
    if keyword in header:
        value = header[keyword]
        if value == expected:
            return False
        if allowFix:
            header[keyword] = (expected, comment)
            if log:
                log.info(f"Fixed value of {keyword} = {repr(value)} --> {repr(expected)}")
            return True
        raise ValueError(f"Value mismatch for {keyword}: got {repr(value)} but expected {repr(expected)}")
    if allowFix:
        header[keyword] = (expected, comment)
        if log:
            log.info(f"Added value of {keyword} = {repr(expected)}")
        return True
    raise ValueError(f"No header keyword {keyword}")


T = TypeVar("T")


def subclasses(cls: Type[T]) -> Set[Type[T]]:
    """Return a set of all subclasses of the provided class"""
    subs = cls.__subclasses__()
    return set(subs).union(*[subclasses(ss) for ss in subs])


def convertToIso8601Utc(datestr):
    """Convert an ISO 8601 date string to a standardized UTC format with a 'Z' suffix.

    This function ensures that the input datetime string is correctly interpreted
    as UTC and formatted in ISO 8601 format. It properly handles 'Z' notation
    and timezone offsets, converting everything to UTC.

    Parameters
    ----------
    datestr : `str`
        Input date string in ISO 8601 format.

    Returns
    -------
    formatted_date : `str`
        The date string formatted as 'YYYY-MM-DDTHH:MM:SS.sssZ' in UTC.

    Raises
    ------
    ValueError
        If the input string is not a valid ISO 8601 date.
    """
    # Handle 'Z' case manually (Python 3.8-3.10 doesn't support it)
    if datestr.endswith("Z"):
        datestr = datestr[:-1] + "+00:00"

    # Parse the datetime string
    date = datetime.datetime.fromisoformat(datestr)

    # Convert to UTC if it has a timezone
    if date.tzinfo is not None:
        date = date.astimezone(datetime.timezone.utc)
    else:
        date = date.replace(tzinfo=datetime.timezone.utc)

    # Format with milliseconds and 'Z' suffix
    return date.isoformat(timespec='milliseconds').replace("+00:00", "Z")
