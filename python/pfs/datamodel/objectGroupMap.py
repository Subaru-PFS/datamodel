import numpy as np
from numpy.typing import ArrayLike
import astropy.io.fits


__all__ = ("ObjectGroupMap",)


def arrayMap(
    fromValues: np.ndarray,
    toValues: np.ndarray,
    find: ArrayLike,
) -> ArrayLike:
    """Mapping from an int array to another array

    Parameters
    ----------
    fromValues : `np.ndarray`
        The values to map from. This is assumed to be an integer array sorted
        in ascending order.
    toValues : `np.ndarray`
        The values to map to.
    find : `array-like`
        The values to find in the ``fromValues`` array.

    Returns
    -------
    mapped : `array-like`
        The mapped values.
    """
    find = np.asarray(find, dtype=fromValues.dtype)
    index = np.searchsorted(fromValues, find)
    if np.any(index < 0) or np.any(index >= fromValues.size):
        raise KeyError("Cannot find all provided values in mapping")
    if np.any(fromValues[index] != find):
        raise KeyError("Cannot find all provided values in mapping")
    mapped = toValues[index]
    return mapped.item() if np.isscalar(find) else mapped


class ObjectGroupMap:
    """Mapping of object groups

    Contains the mapping between objId and objGroup.

    Parameters
    ----------
    lookup : `ObjectGroupMapLookup`
        A mapping from objId to objGroup. The objId array must be unique. The
        objGroup array must be the same length as the objId array.

    Raises
    ------
    ValueError
        If objId and objGroup are not the same length or if there are duplicate
        objId values.
    """

    TYPE_OBJID = np.int64
    TYPE_OBJGROUP = np.int32
    FITS_EXTNAME = "OBJECT_GROUP_MAP"

    def __init__(self, objId: np.ndarray, objGroup: np.ndarray) -> None:
        if objId.size != objGroup.size:
            raise ValueError("objId and objGroup must have the same length")
        if objId.size != np.unique(objId).size:
            raise ValueError("Duplicate objId values found")
        index = np.argsort(objId)
        self.objId = objId[index]
        self.objGroup = objGroup[index]

    @classmethod
    def combine(cls, *maps: "ObjectGroupMap") -> "ObjectGroupMap":
        """Combine multiple ObjectGroupMaps into one.

        Parameters
        ----------
        *maps : `ObjectGroupMap`
            The ObjectGroupMaps to combine.

        Returns
        -------
        combined : `ObjectGroupMap`
            The combined ObjectGroupMap.
        """
        objId = np.concatenate([mm.objId for mm in maps])
        objGroup = np.concatenate([mm.objGroup for mm in maps])
        return cls(objId, objGroup)

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"({len(self)} objects)"

    def __len__(self) -> int:
        """Return the total number of objects in the mapping."""
        return self.objId.size

    def __add__(self, other: "ObjectGroupMap") -> "ObjectGroupMap":
        """Add another ObjectGroupMap to this one.

        Parameters
        ----------
        other : `ObjectGroupMap`
            The other `ObjectGroupMap` to add.

        Returns
        -------
        combined : `ObjectGroupMap`
            The combined `ObjectGroupMap`.
        """
        return self.combine(self, other)

    def __getitem__(self, objId: ArrayLike) -> ArrayLike:
        """Get the object group given objId.

        Parameters
        ----------
        objId : `array-like`
            The ``objId`` value(s) to look up.

        Returns
        -------
        objGroup: array-like
            The object group for the given ``objId`` value(s).
        """
        isScalar = np.isscalar(objId)
        objId = np.asarray(objId, dtype=self.TYPE_OBJID)
        result = arrayMap(self.objId, self.objGroup, objId)
        return int(result.item()) if isScalar else result

    def writeFits(self, filename: str) -> None:
        """Write to a FITS file

        Parameters
        ----------
        filename : `str`
            The name of the FITS file to write.
        """
        # NOTE: When making any changes to this method that modify the output
        # format, increment the DAMD_VER header value, and record the change in
        # the versions.txt file.
        header = astropy.io.fits.Header()
        header["DAMD_VER"] = (1, "ObjectGroupMap datamodel version")
        table = astropy.io.fits.BinTableHDU.from_columns([
            astropy.io.fits.Column(name="objId", format="K", array=self.objId),
            astropy.io.fits.Column(name="objGroup", format="J", array=self.objGroup)
        ], name=self.FITS_EXTNAME, header=header)
        fits = astropy.io.fits.HDUList([astropy.io.fits.PrimaryHDU(), table])
        fits.writeto(filename, overwrite=True)

    @classmethod
    def readFits(cls, filename: str) -> "ObjectGroupMap":
        """Read from a FITS file

        Parameters
        ----------
        filename : `str`
            The name of the FITS file to read.

        Returns
        -------
        map : `ObjectGroupMap`
            The object group map.
        """
        with astropy.io.fits.open(filename) as fits:
            table = fits[cls.FITS_EXTNAME].data
            objId = table["objId"].astype(cls.TYPE_OBJID)
            objGroup = table["objGroup"].astype(cls.TYPE_OBJGROUP)
            return cls(objId, objGroup)
