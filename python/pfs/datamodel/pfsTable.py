from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, Type, TypeVar, Union

import astropy.io.fits
import numpy as np

__all__ = ("POD", "Column", "PfsTable", "EmptyTable")

POD = Union[
    int,
    float,
    np.int32,
    np.float32,
    bool,
    np.uint8,
    np.int16,
    np.int64,
    np.float64,
    str,
]
"""Plain old data

These are the column types that are supported by PfsTable.
"""


@dataclass
class Column:
    """Column definition

    Parameters
    ----------
    name : `str`
        Name of column.
    dtype : `type`
        Data type of column.
    doc : `str`
        Documentation for column.
    default : `POD`
        Default value for column.
    """

    name: str
    dtype: Type[POD]
    doc: str
    default: POD


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

    schema: Iterable[Column]
    """Schema to use for table (iterable of `Column`)"""

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
        missing = set(col.name for col in self.schema) - set(columns.keys())
        if missing:
            raise RuntimeError(f"Missing columns: {missing}")

        allShapes = set(array.shape for array in columns.values())
        if len(allShapes) > 0:
            if len(allShapes) != 1:
                raise RuntimeError(f"Columns have differing shapes: {allShapes}")
            shape = allShapes.pop()
            if len(shape) != 1:
                raise RuntimeError(
                    f"Columns are not one-dimensional arrays: shape={shape}"
                )
            self.length = shape[0]
        else:
            self.length = 0

        for col in schema:
            setattr(self, col.name, columns[col.name])

    if TYPE_CHECKING:

        def __getattr__(self, name: str) -> POD:
            ...

        def __setattr__(self, name: str, value: POD) -> None:
            ...

    @classmethod
    def getSchemaDict(cls) -> Dict[str, Column]:
        """Get the schema as a `dict`

        The `dict` is indexed by column name.

        Returns
        -------
        schema : `dict` mapping `str` to `Column`
            The schema, indexed by column name.
        """
        return {col.name: col for col in cls.schema}

    def __len__(self) -> int:
        """Return the number of rows in the table"""
        return self.length

    def __getitem__(self: SubTable, selection: Union[np.ndarray, slice]) -> SubTable:
        """Sub-selection

        Parameters
        ----------
        selection : `numpy.ndarray` of `bool` or `slice`
            Boolean array (of same length as ``self``) indicating which rows
            to select; or a slice.

        Returns
        -------
        new : ``type(self)``
            A new instance containing only the selected rows.
        """
        columns = {col.name: getattr(self, col.name)[selection] for col in self.schema}
        return type(self)(**columns)

    def __setitem__(self, selection: Union[np.ndarray, slice], other: SubTable):
        """Set sub-selection

        Parameters
        ----------
        selection : `numpy.ndarray` of `bool` or `slice`
            Boolean array (of same length as ``self``) indicating which rows
            to select; or a slice.
        other : ``type(self)``
            The values to set.
        """
        for col in self.schema:
            getattr(self, col.name)[selection] = getattr(other, col.name)

    def setRow(self, index: int, **columns: POD):
        """Set values for a given row

        Parameters
        ----------
        index : `int`
            Row index.
        **columns : `POD`
            Column values, indexed by column name.
        """
        for name, value in columns.items():
            getattr(self, name)[index] = value

    @classmethod
    def empty(cls: Type[SubTable], length: int) -> SubTable:
        """Create an empty table

        Parameters
        ----------
        length : `int`
            Number of rows in the table.

        Returns
        -------
        self : cls
            Constructed object.
        """
        columns = {
            col.name: np.full(length, col.default, dtype=col.dtype)
            for col in cls.schema
        }
        return cls(**columns)

    @property
    def columns(self) -> Dict[str, np.ndarray]:
        """Get a `dict` of column data, indexed by column name"""
        return {col.name: getattr(self, col.name) for col in self.schema}

    @classmethod
    def readHdu(cls: Type[SubTable], fits: astropy.io.fits.HDUList) -> SubTable:
        """Read from FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file from which to read.

        Returns
        -------
        self : cls
            Constructed object from reading file.
        """
        hdu = fits[cls.fitsExtName]
        columns: Dict[str, np.ndarray] = {}
        available = set(hdu.data.columns.names)
        for col in cls.schema:
            if col.name in available:
                array = hdu.data[col.name]
            else:
                aliases = cls.aliases.get(col.name, {})
                for nn in aliases:
                    if nn in available:
                        array = hdu.data[nn]
                        break
                else:
                    array = np.full(len(hdu.data), col.default, dtype=col.dtype)
            columns[col.name] = array.astype(col.dtype)
        return cls(**columns)

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
            return cls.readHdu(fits)

    def writeHdu(self, fits: astropy.io.fits.HDUList):
        """Write to FITS HDU

        Parameters
        ----------
        hdu : `astropy.io.fits.HDUList`
            FITS file to which to write.
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
                name=col.name,
                format=getFormat(col.name, col.dtype),
                array=getattr(self, col.name),
            )
            for col in self.schema
        ]

        header = astropy.io.fits.Header()
        header["DAMD_VER"] = (
            self.damdVer,
            f"{self.__class__.__name__} datamodel version",
        )
        header["INHERIT"] = True
        for ii, col in enumerate(self.schema):
            # TDOCn is not a FITS standard keyword, but we want to write the column doc for our users and
            # there is no standard for that. The name comes from lsst.afw.table.
            header[f"TDOC{ii + 1}"] = col.doc

        fits.append(
            astropy.io.fits.BinTableHDU.from_columns(
                columns, name=self.fitsExtName, header=header
            )
        )

    def writeFits(self, filename: str):
        """Write to FITS file

        Parameters
        ----------
        filename : `str`
            Filename to which to write.
        """
        fits = astropy.io.fits.HDUList([astropy.io.fits.PrimaryHDU()])
        self.writeHdu(fits)
        with open(filename, "wb") as fd:
            fits.writeto(fd)


class EmptyTable(PfsTable):
    """A table with no columns

    Parameters
    ----------
    length : `int`
        Number of rows.
    """

    schema: Iterable[Column] = []

    def __init__(self, length: int):
        self.length = length

    def __getitem__(self: SubTable, selection: Union[np.ndarray, slice]) -> SubTable:
        """Sub-selection

        Parameters
        ----------
        selection : `numpy.ndarray` of `bool` or `slice`
            Boolean array (of same length as ``self``) indicating which rows
            to select; or a slice.

        Returns
        -------
        new : ``type(self)``
            A new instance containing only the selected rows.
        """
        if isinstance(selection, np.ndarray):
            if selection.dtype != bool:
                raise TypeError(f"Expected bool array; got {selection.dtype}")
            return type(self)(selection.sum())
        elif isinstance(selection, slice):
            return type(self)(len(range(*selection.indices(len(self)))))
        else:
            raise TypeError(f"Expected slice or bool array; got {type(selection)}")

    @classmethod
    def empty(cls: Type["EmptyTable"], length: int) -> "EmptyTable":
        """Create an empty table

        Parameters
        ----------
        length : `int`
            Number of rows in the table.

        Returns
        -------
        self : cls
            Constructed object.
        """
        return EmptyTable(length)
