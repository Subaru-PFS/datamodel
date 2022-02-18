#
# A couple of useful functions to calculate PFS's SHA-1s
#
import hashlib
import inspect
import functools
import numpy as np

__all__ = ("calculatePfsVisitHash", "createHash", "astropyHeaderToDict", "astropyHeaderFromDict",
           "wraparoundNVisit", "inheritDocstrings",)


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

    def _roundToArcsec(d):
        if np.isnan(d):
            return d  # Just return nan is this case
        return int(d*3600.0 + 0.5)/3600.0

    # Regardless of the arcsec rounding, we need to choose a precision for the string respresentation.
    # If datamodel.txt phrasing were a little sloppier, we could just use integer arcseconds.
    return createHash(["%d %0.6f %0.6f" % (fiberId, _roundToArcsec(ra), _roundToArcsec(dec))
                       for fiberId, ra, dec in zip(fiberIds, ras, decs)])


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
