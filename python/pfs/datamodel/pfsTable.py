from typing import Dict, Iterable, Type, TypeVar

import astropy.io.fits
import numpy as np

__all__ = ("PfsTable",)


SubTable = TypeVar("SubTable", bound="PfsTable")
"""A generic sub-class of PfsTable"""


class PfsTable:
    """A table of values

    This class is focussed on I/O of table classes. For an implementation that
    provides more features, such as lookups and iteration see the drp_stella
    package.

    Subclasses must set the ``schema`` and ``fitsExtName`` class
    variables.

    Parameters
    ----------
    **columns : `np.ndarray`
        The column data, each indexed by column name (which should be in the
        ``schema``).
    """

    schema: Dict[str, type]
    """Schema to use for table (`dict` mapping `str` to `type`)

    Each keyword is a column name (`str`), with the value being the `type` for
    that column.
    """

    fitsExtName: str
    """FITS extension name (`str`)"""

    damdVer: int = 1
    """Datamodel version number"""

    aliases: Dict[str, Iterable[str]] = {}
    """Aliases for columns (`dict` mapping `str` to an iterable of `str`)

    This provides support for renaming columns. The current column name (must be
    present in the ``schema``) is associated with a list of alternate names that
    could be used in FITS files.
    """

    def __init__(self, **columns: np.ndarray):
        schema = self.schema
        missing = set(schema.keys()) - set(columns.keys())
        if missing:
            raise RuntimeError(f"Missing columns: {missing}")

        allShapes = set(array.shape for array in columns.values())
        if len(allShapes) != 1:
            raise RuntimeError(f"Columns have differing shapes: {allShapes}")
        shape = allShapes.pop()
        if len(shape) != 1:
            raise RuntimeError(f"Columns are not one-dimensional arrays: shape={shape}")
        self.length = shape[0]

        for col, array in columns.items():
            setattr(self, col, array.astype(schema[col]))

    def __len__(self) -> int:
        """Return the number of rows in the table"""
        return self.length

    def __getattr__(self, column: str) -> np.ndarray:  # helpful for types
        """Get column

        This method mostly exists for the benefit of type checkers, as it sets
        the type for the columns.
        """
        return getattr(self, column)

    @property
    def columns(self) -> Dict[str, np.ndarray]:
        """Get a `dict` of column data, indexed by column name"""
        return {name: getattr(self, name) for name in self.schema}

    @classmethod
    def readFits(cls: Type[SubTable], filename: str) -> SubTable:
        """Read from file

        Parameters
        ----------
        filename : `str`
            Name of file from which to read.

        Returns
        -------
        self : cls
            Constructed object from reading file.
        """
        with astropy.io.fits.open(filename) as fits:
            hdu = fits[cls.fitsExtName]
            columns: Dict[str, np.ndarray] = {}
            available = set(hdu.data.columns.names)
            for name, dtype in cls.schema.items():
                if name in available:
                    array = hdu.data[name]
                else:
                    aliases = cls.aliases.get(name, {})
                    for nn in aliases:
                        if nn in available:
                            array = hdu.data[nn]
                            break
                    else:
                        raise RuntimeError(
                            f"Neither column {name} nor its aliases {aliases} are present in the table"
                        )
                columns[name] = array.astype(dtype)
        return cls(**columns)

    def writeFits(self, filename: str):
        """Write to file

        Parameters
        ----------
        filename : `str`
            Name of file to which to write.
        """
        # NOTE: When making any changes to this method that modify the output
        # format, increment the damdVer class attribute.
        format = {
            int: "K",
            float: "D",
            np.int32: "J",
            np.float32: "E",
            bool: "L",
            np.uint8: "B",
            np.int16: "I",
            np.int64: "K",
            np.float64: "D",
        }

        def getFormat(name: str, dtype: type) -> str:
            """Determine suitable FITS column format

            This is a simple mapping except for string types.

            Parameters
            ----------
            name : `str`
                Column name, so we can get the data if we need to inspect it.
            dtype : `type`
                Data type.

            Returns
            -------
            format : `str`
                FITS column format string.
            """
            if issubclass(dtype, str):
                length = (
                    max(len(ss) for ss in getattr(self, name)) if len(self) > 0 else 0
                )
                length = max(1, length)  # Minimum length of 1 makes astropy happy
                return f"{length}A"
            return format[dtype]

        columns = [
            astropy.io.fits.Column(
                name=name, format=getFormat(name, dtype), array=getattr(self, name)
            )
            for name, dtype in self.schema.items()
        ]
        hdu = astropy.io.fits.BinTableHDU.from_columns(columns, name=self.fitsExtName)

        hdu.header["INHERIT"] = True
        hdu.header["DAMD_VER"] = (
            self.damdVer,
            f"{self.__class__.__name__} datamodel version",
        )

        fits = astropy.io.fits.HDUList([astropy.io.fits.PrimaryHDU(), hdu])
        with open(filename, "wb") as fd:
            fits.writeto(fd)
