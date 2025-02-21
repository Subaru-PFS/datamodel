from dataclasses import make_dataclass, field
from typing import Dict, Iterable, Type, TYPE_CHECKING

import numpy as np
from astropy.io.fits import HDUList

from .pfsTable import PfsTable, Column, POD


class Notes:
    """Base class for notes about spectral reduction of a single spectrum.

    This provides I/O for the notes by converting to a PfsTable with a single
    row.
    """

    schema: Iterable[Column]
    _TableClass: Type[PfsTable]

    if TYPE_CHECKING:

        def __getattr__(self, name: str) -> POD:
            ...

        def __setattr__(self, name: str, value: POD) -> None:
            ...

    def update(self, **kwargs: POD):
        """Update values

        Parameters
        ----------
        kwargs
            Values to update.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def getDict(self) -> Dict[str, POD]:
        """Return a dictionary of the values

        Returns
        -------
        values : `dict` [`str`, `POD`]
            Values.
        """
        return {col.name: getattr(self, col.name) for col in self.schema}

    @classmethod
    def readFits(cls, fits: HDUList) -> "Notes":
        """Read from FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file from which to read.

        Returns
        -------
        self : `Notes`
            Constructed object from reading file.
        """
        table = cls._TableClass.readHdu(fits)
        columns = {col.name: getattr(table, col.name)[0] for col in cls.schema}
        return cls(**columns)

    def writeFits(self, fits: HDUList):
        """Write to FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file to which to write.
        """
        columns = {
            col.name: np.array([getattr(self, col.name)], dtype=col.dtype)
            for col in self.schema
        }
        table = self._TableClass(**columns)
        table.writeHdu(fits)


def makeNotesClass(
    name: str, schema: Iterable[Column], fitsExtName: str = "NOTES"
) -> Type[Notes]:
    """Build a Notes class

    The class will be a `dataclass` with members according to the provided
    ``schema``, and inherit from `Notes` (to provide I/O).

    Parameters
    ----------
    name : `str`
        Name of the class to build.
    schema : `Iterable` [ `Column` ]
        Schema for the class.
    fitsExtName : `str`, optional
        Name of the FITS extension.

    Returns
    -------
    cls : `type`
        Built class.
    """
    tableNamespace = dict(
        schema=schema, fitsExtName=fitsExtName
    )  # class variables for PfsTable
    TableClass = type(name, (PfsTable,), tableNamespace)
    fieldList = [(col.name, col.dtype, field(default=col.default)) for col in schema]
    namespace = dict(schema=schema, _TableClass=TableClass)
    return make_dataclass(name, fieldList, bases=(Notes,), namespace=namespace)
