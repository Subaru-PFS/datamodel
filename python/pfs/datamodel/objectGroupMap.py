from typing import Any, Dict, Optional, Tuple, overload

import numpy as np
from numpy.typing import ArrayLike
import astropy.io.fits


__all__ = ("ObjectGroupMap",)


ObjectGroupMapLookup = Dict[int, Tuple[np.ndarray, np.ndarray]]  # Mapping of catId to (objId, objGroup)


class ObjectGroupMap:
    """Mapping of object groups

    Contains the mapping between catId, objId and objGroup.

    Parameters
    ----------
    lookup : `ObjectGroupMapLookup`
        A mapping from catId to a tuple of objId and objGroup arrays.

    Raises
    ------
    ValueError
        If catId, objId and objGroup are not the same length or if there are
        duplicate objId values for the same catId.
    """

    TYPE_CATID = np.int32
    TYPE_OBJID = np.int64
    TYPE_OBJGROUP = np.int32
    FITS_EXTNAME = "OBJECT_GROUP_MAP"

    def __init__(self, lookup: ObjectGroupMapLookup) -> None:
        self._lookup = lookup
        for catId in lookup:
            objId, objGroup = lookup[catId]
            if objId.size != objGroup:
                raise ValueError("objId and objGroup must have the same length")
            if objId.size != np.unique(objId).size:
                raise ValueError(f"Duplicate objId values found for catId={catId}")

    @classmethod
    def fromArrays(cls, catId: np.ndarray, objId: np.ndarray, objGroup: np.ndarray) -> "ObjectGroupMap":
        """Create an ObjectGroupMap from arrays.

        Parameters
        ----------
        catId : `np.ndarray`
            Array of catId values.
        objId : `np.ndarray`
            Array of objId values.
        objGroup : `np.ndarray`
            Array of object group values.

        Returns
        -------
        mapping : `ObjectGroupMap`
            The object group map.
        """
        if len(catId) != len(objId) or len(catId) != len(objGroup):
            raise ValueError("catId, objId and objGroup must have the same length")

        lookup: Dict[int: tuple[np.ndarray, np.ndarray]] = {}
        for cc in np.unique(catId):
            select = cc == catId
            lookup[cc] = (objId[select], objGroup[select])
        return cls(lookup)

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
        arrays = [mm.getArrays() for mm in maps]
        catId = np.concatenate([arr[0] for arr in arrays])
        objId = np.concatenate([arr[1] for arr in arrays])
        objGroup = np.concatenate([arr[2] for arr in arrays])
        return cls.fromArrays(catId, objId, objGroup)

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"({len(self)} objects for catId={self.catId().tolist()})"

    def __len__(self) -> int:
        """Return the total number of objects in the mapping."""
        return sum(lookup[0].size for lookup in self._lookup.values())

    def getArrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the arrays of catId, objId and objGroup.

        Returns
        -------
        catId : `np.ndarray`
            The catId values.
        objId : `np.ndarray`
            The objId values.
        objGroup : `np.ndarray`
            The object group values.
        """
        length = len(self)
        catId = np.zeros(length, dtype=self.TYPE_CATID)
        objId = np.zeros(length, dtype=self.TYPE_OBJID)
        objGroup = np.zeros(length, dtype=self.TYPE_OBJGROUP)
        index = 0
        for cc in sorted(self._lookup.keys()):
            oo, gg = self._lookup[cc]
            num = oo.size
            catId[index:index + num] = cc
            objId[index:index + num] = oo
            objGroup[index:index + num] = gg
            index += num
        return catId, objId, objGroup

    def catId(self) -> np.ndarray:
        """Return unique catId values"""
        return np.array(sorted(self._lookup.keys()), dtype=self.TYPE_CATID)

    def objId(self, catId: int) -> np.ndarray:
        """Get the objId values for a given catId.

        Parameters
        ----------
        catId : `int`
            The catId to look up.

        Returns
        -------
        objId : `np.ndarray`
            The objId values for the given catId.
        """
        # All objId values should be unique for a given catId
        assert self._lookup[catId][0].size == np.unique(self._lookup[catId][0]).size
        return self._lookup[catId][0]

    def objGroup(self, catId: int) -> np.ndarray:
        """Get the unique objGroup values for a given catId.

        Parameters
        ----------
        catId : `int`
            The catId to look up.

        Returns
        -------
        objGroup : `np.ndarray`
            The unique objGroup values for the given catId.
        """
        return np.unique(self._lookup[catId][1])

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
        return self.combine([self, other])

    @overload
    def __getitem__(self, objId: ArrayLike) -> ArrayLike: ...

    @overload
    def __getitem__(self, catId: ArrayLike, objId: ArrayLike) -> ArrayLike: ...

    def __getitem__(self, catId: ArrayLike, objId: Optional[ArrayLike] = None) -> ArrayLike:
        """Get the object group for a given catId and objId.

        The ``objId`` can be provided without a ``catId``, so long as it is
        unique.

        Parameters
        ----------
        catId : array-like
            The catId to look up.
        objId : array-like
            The objId to look up.

        Returns
        -------
        objGroup: array-like
            The object group for the given catId and objId.

        Raises
        ------
        ValueError
            If the catId and objId arrays are not the same length.
        KeyError
            If the objId only is specfied and is not unique.
        """
        catId = np.asarray(catId, dtype=self.TYPE_CATID)
        objGroup = np.full_like(catId, -1, dtype=self.TYPE_OBJGROUP)

        if objId is None:
            # We have a bunch of objId values which should be unique
            # across all catId values, so we iterate over all catId values.
            objId = catId
            for lookup in self._lookup.values():
                select = np.isin(objId, lookup[0])
                if not np.any(select):
                    continue
                if np.any(objGroup[select] != -1):
                    raise KeyError(f"objId={objId[objGroup[select] != -1]} are not unique within map")
                objGroup[select] = lookup[1][np.searchsorted(lookup[0], objId[select])]
        else:
            objId = np.asarray(objId, dtype=self.TYPE_OBJID)
            if objId.size != catId.size:
                raise ValueError("catId and objId must have the same length")
            for cc in np.unique(catId):
                select = cc == catId
                lookup = self._lookup[cc]
                objGroup[select] = lookup[1][np.searchsorted(lookup[0], objId[select])]

        if np.any(objGroup == -1):
            index = np.where(objGroup == -1)[0]
            raise KeyError(f"Lookup failed for index={index}")

        return objGroup.item() if np.isscalar(catId) else objGroup

    @overload
    def dataId(self, objId: int) -> Dict[str, Any]: ...

    @overload
    def dataId(self, catId: int, objId: int) -> Dict[str, Any]: ...

    def dataId(self, catId: int, objId: Optional[int] = None) -> Dict[str, Any]:
        """Get the dataId for a given object.

        The ``objId`` can be provided without a ``catId``, so long as it is
        unique.

        This adds only the ``catId`` and ``objGroup`` to the dataId; you will
        need more entries (e.g., ``instrument`` and ``combination``) to
        retrieve a ``pfsCoadd``.

        Parameters
        ----------
        catId : `int`
            The catId to look up.
        objId : `int`
            The objId to look up.

        Returns
        -------
        dataId : `dict`
            The data identifier to retrieve the nominated object.
        """
        objGroup: Optional[int] = None
        if objId is None:
            objId = catId
            catId = None
            for cc, lookup in self._lookup.items():
                select = lookup[0] == objId
                if not np.any(select):
                    continue
                if catId is not None:
                    raise KeyError(f"objId={objId} is not unique within map")
                assert select.sum() == 1
                catId = cc
                objGroup = lookup[1][select]
            if catId is None:
                raise KeyError(f"objId={objId} not found in any catId")
        else:
            lookup = self._lookup[catId]
            objGroup = lookup[1][np.searchsorted(lookup[0], objId)]

        return dict(cat_id=catId, obj_group=objGroup)

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
        catId, objId, objGroup = self.getArrays()
        table = astropy.io.fits.BinTableHDU.from_columns([
            astropy.io.fits.Column(name="catId", format="J", array=catId),
            astropy.io.fits.Column(name="objId", format="K", array=objId),
            astropy.io.fits.Column(name="objGroup", format="J", array=objGroup)
        ], name=self.FITS_EXTNAME, header=header)
        fits = astropy.io.fits.HDUList([table])
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
            catId = table["catId"].astype(cls.TYPE_CATID)
            objId = table["objId"].astype(cls.TYPE_OBJID)
            objGroup = table["objGroup"].astype(cls.TYPE_OBJGROUP)
            return cls.fromArrays(catId, objId, objGroup)
