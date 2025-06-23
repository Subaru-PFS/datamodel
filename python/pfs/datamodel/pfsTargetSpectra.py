from collections.abc import Mapping
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union, cast, overload

import astropy.io.fits
import numpy as np
import yaml
from astropy.io.fits import BinTableHDU, Column, CompImageHDU, HDUList, Header, ImageHDU, PrimaryHDU

from .masks import MaskHelper
from .observations import Observations
from .pfsConfig import TargetType
from .pfsTable import PfsTable
from .target import Target
from .fluxTable import FluxTable
from .pfsFiberArray import PfsFiberArray

__all__ = ["PfsTargetSpectra",]


class PfsTargetSpectra(Mapping[Target, PfsFiberArray]):
    """A collection of `PfsFiberArray` indexed by target

    Parameters
    ----------
    spectra : list of `PfsFiberArray`
        Spectra to be indexed by target.
    """

    PfsFiberArrayClass: Type[PfsFiberArray]  # Subclasses must override
    NotesClass: Type[PfsTable]  # Subclasses must override

    def __init__(self, spectra: Sequence[PfsFiberArray], metadata: Optional[Mapping[str, Any]] = None):
        super().__init__()
        if metadata is None:
            metadata = astropy.io.fits.Header()
        self.spectra: Dict[Target, PfsFiberArray] = {spectrum.target: spectrum for spectrum in spectra}
        if len(self.spectra) != len(spectra):
            raise RuntimeError("Spectra targets not unique")
        self.metadata = metadata

        # Lookup by catId,objId
        self._byCatIdObjId: Dict[Tuple[int, int], PfsFiberArray]
        self._byCatIdObjId = {(ss.target.catId, ss.target.objId): ss for ss in spectra}
        if len(self._byCatIdObjId) != len(spectra):
            raise RuntimeError("Spectra catId,objId not unique")

        # Lookup by objId (only when all objIds are unique)
        self._byObjId: Optional[Dict[int, PfsFiberArray]]
        self._byObjId = {tt.objId: ss for tt, ss in self.spectra.items()}
        if len(self._byObjId) != len(self.spectra):
            # Disable lookup by objId
            self._byObjId = None

    @overload
    def __getitem__(self, objId: int) -> PfsFiberArray:
        ...

    @overload
    def __getitem__(self, catId: int, objId: int) -> PfsFiberArray:
        ...

    @overload
    def __getitem__(self, catIdObjId: Tuple[int, int]) -> PfsFiberArray:
        ...

    @overload
    def __getitem__(self, target: Target) -> PfsFiberArray:
        ...

    def __getitem__(
        self, first: Union[int, Target, Tuple[int, int]], second: Optional[int] = None
    ) -> PfsFiberArray:
        """Retrieve spectrum for target

        The following overloads are supported:
        - Retrieve by ``objId`` (only available when all ``objId`` are unique):
            ``spectra[objId]``
        - Retrieve by ``catId`` and ``objId``: ``spectra[catId, objId]`` or
            ``spectra[(catId, objId)]``
        - Retrieve by `Target`: ``spectra[target]``
        """
        if second is not None:
            return self._byCatIdObjId[(cast(int, first), second)]
        if isinstance(first, tuple):
            return self._byCatIdObjId[first]

        if isinstance(first, int):
            if self._byObjId is None:
                raise KeyError("objId lookup not available when all objIds are not unique")
            return self._byObjId[cast(int, first)]
        return self.spectra[first]

    def __iter__(self) -> Iterator[Target]:
        """Return iterator over targets"""
        return iter(self.spectra)

    def __len__(self) -> int:
        """Return number of spectra"""
        return len(self.spectra)

    def __contains__(self, target) -> bool:
        """Return whether target has spectrum

        The following overloads are supported:
        - Check by ``objId`` (only available when all ``objId`` are unique):
            ``objId in spectra``
        - Check by ``catId`` and ``objId``: ``(catId, objId) in spectra``
        - Check by `Target`: ``target in spectra``
        """
        if isinstance(target, tuple):
            return target in self._byCatIdObjId
        if isinstance(target, int):
            if self._byObjId is None:
                raise KeyError("objId lookup not available when objIds are not unique")
            return cast(int, target) in self._byObjId
        return target in self.spectra

    @classmethod
    def readFits(cls, filename: str, **kwargs) -> "PfsTargetSpectra":
        """Read a collection of spectra from a FITS file, optionally
        filtered by keyword arguments.

        Parameters
        ----------
        filename : `str`
            Filename of FITS file.
        **kwargs : `dict`
            Keyword arguments to filter the spectra. The keys supported depend on the
            subclass of `PfsTargetSpectra`. For example, the `PfsFiberArray` class supports
            filtering by `targetType`, `catId`, and `objId`.

        Returns
        -------
        self : ``cls``
            Constructed instance, from FITS file.
        """

        def readData(hduName: str, dtype: type, mask=None) -> Union[np.ndarray, List[np.ndarray]]:
            """Read data from FITS file

            Parameters
            ----------
            hduName : `str`
                Name of the HDU.
            dtype : `type`
                Data type.
            mask : `np.ndarray`, optional
                Mask to apply to the data. If provided, the data will be read as a masked array.

            Returns
            -------
            data : `np.ndarray` or list of `np.ndarray`
                Data read from FITS file.
            """
            hdu = fits[hduName]
            if isinstance(hdu, (ImageHDU, CompImageHDU)):
                if mask is None:
                    return hdu.data.astype(dtype)
                else:
                    if hdu.data.ndim == 1:
                        # Same for all spectra, such as wavelength, do not filter
                        return hdu.data.astype(dtype)
                    else:
                        # Different for each spectrum, such as flux, do filter
                        return hdu.data[mask].astype(dtype)
            elif isinstance(hdu, BinTableHDU):
                # This is a special case of storing variable length arrays in a table.
                # Rows of 2D arrays are stored as columns of a table (e.g. covariance)
                numRows = len(hdu.data.dtype)
                if numRows == 1:
                    # 1D array stored as a single column
                    if mask is None:
                        return [row["value"].astype(dtype) for row in hdu.data]
                    else:
                        return [row.astype(dtype) for row in hdu.data["value"][mask]]
                else:
                    # 2D array stored as multiple columns
                    data: List[np.ndarray] = []
                    if mask is not None:
                        index = np.where(mask)[0]
                    else:
                        index = range(len(hdu.data))

                    for ii in index:
                        data.append(np.array(
                            [hdu.data[f"row_{jj}"][ii] for jj in range(numRows)],
                            dtype=dtype))

                    return data
            else:
                raise TypeError(f"HDU '{hduName}' is not an ImageHDU, CompImageHDU or BinTableHDU.")

        spectra = []
        with astropy.io.fits.open(filename) as fits:
            # Read the main headers and the list of targets.
            header = fits[0].header
            targetHdu = fits["TARGET"].data

            # If any keyword arguments are provided, filter the spectra
            mask = None
            for key, value in kwargs.items():
                if key not in targetHdu.columns.names:
                    raise KeyError(f"Keyword argument '{key}' not found in TARGET HDU.")
                m = targetHdu[key] == value
                mask = m if mask is None else mask & m

            targetFluxHdu = fits["TARGETFLUX"].data
            observationsHdu = fits["OBSERVATIONS"].data
            wavelengthData = readData("WAVELENGTH", np.float64, mask=mask)
            fluxData = readData("FLUX", np.float32, mask=mask)
            maskData = readData("MASK", np.int32, mask=mask)
            skyData = readData("SKY", np.float32, mask=mask)
            covarData = readData("COVAR", np.float32, mask=mask)
            covar2Data = readData("COVAR2", np.float32, mask=mask) if "COVAR2" in fits else None
            metadataHdu = fits["METADATA"].data
            fluxTableHdu = fits["FLUXTABLE"].data
            notesTable = cls.NotesClass.readHdu(fits)

            if mask is None:
                targets = enumerate(targetHdu.targetId)
            else:
                targets = zip(np.where(mask)[0], targetHdu.targetId[mask])

            for ii, (index, targetId) in enumerate(targets):
                row = targetHdu[index]
                select = targetFluxHdu.targetId == targetId
                fiberFlux = dict(
                    zip(
                        ("".join(np.char.decode(ss.astype("S"))) for ss in targetFluxHdu.filterName[select]),
                        targetFluxHdu.fiberFlux[select],
                    )
                )
                target = Target(
                    row["catId"],
                    row["tract"],
                    "".join(row["patch"]),
                    row["objId"],
                    row["ra"],
                    row["dec"],
                    TargetType(row["targetType"]),
                    fiberFlux=fiberFlux,
                )

                select = observationsHdu.targetId == targetId
                obsTime = None
                if "obsTime" in observationsHdu.columns.names:
                    obsTime = ["".join(ss) for ss in observationsHdu.obsTime[select]]
                expTime = None
                if "expTime" in observationsHdu.columns.names:
                    expTime = observationsHdu.expTime[select]
                observations = Observations(
                    observationsHdu.visit[select],
                    ["".join(np.char.decode(ss.astype("S"))) for ss in observationsHdu.arm[select]],
                    observationsHdu.spectrograph[select],
                    observationsHdu.pfsDesignId[select],
                    observationsHdu.fiberId[select],
                    observationsHdu.pfiNominal[select],
                    observationsHdu.pfiCenter[select],
                    obsTime,
                    expTime,
                )

                metadataRow = metadataHdu[index]
                assert metadataRow["targetId"] == targetId

                metadata = yaml.load(
                    # This complicated conversion is required in order to preserve the newlines
                    "".join(np.char.decode(metadataRow["metadata"].astype("S"))),
                    Loader=yaml.SafeLoader,
                )
                flags = MaskHelper.fromFitsHeader(metadata, strip=True)

                fluxTableRow = fluxTableHdu[index]
                assert fluxTableRow["targetId"] == targetId
                fluxTable = FluxTable(
                    fluxTableRow["wavelength"],
                    fluxTableRow["flux"],
                    fluxTableRow["error"],
                    fluxTableRow["mask"],
                    flags,
                )

                notes = cls.PfsFiberArrayClass.NotesClass(
                    **{col.name: getattr(notesTable, col.name)[index] for col in notesTable.schema}
                )

                # Wavelength: we might write a single array if everything has the same length and value
                wavelength: np.ndarray
                if isinstance(wavelengthData, np.ndarray) and len(wavelengthData.shape) == 1:
                    wavelength = wavelengthData
                else:
                    wavelength = wavelengthData[ii]

                spectrum = cls.PfsFiberArrayClass(
                    target,
                    observations,
                    wavelength,
                    fluxData[ii],
                    maskData[ii],
                    skyData[ii],
                    covarData[ii],
                    covar2Data[ii] if covar2Data is not None else [],
                    flags,
                    metadata,
                    fluxTable,
                    notes,
                )
                spectra.append(spectrum)

        return cls(spectra, header)

    def writeFits(self, filename: str):
        """Write to FITS file

        This API is intended for use by the LSST data butler, which handles
        translating the desired identity into a filename.

        Parameters
        ----------
        filename : `str`
            Filename of FITS file.
        """
        # NOTE: When making any changes to this method that modify the output
        # format, increment the DAMD_VER header value and record the change in
        # the versions.txt file.
        fits = HDUList()
        header = Header(self.metadata)
        header['DAMD_VER'] = (1, "PfsTargetSpectra datamodel version")
        fits.append(PrimaryHDU(header=header))

        targetId = np.arange(len(self), dtype=np.int16)
        fits.append(
            BinTableHDU.from_columns(
                [
                    Column("targetId", "I", array=targetId),
                    Column("catId", "J", array=[target.catId for target in self]),
                    Column("tract", "J", array=[target.tract for target in self]),
                    Column("patch", "PA()", array=[target.patch for target in self]),
                    Column("objId", "K", array=[target.objId for target in self]),
                    Column("ra", "D", array=[target.ra for target in self]),
                    Column("dec", "D", array=[target.dec for target in self]),
                    Column("targetType", "I", array=[int(target.targetType) for target in self]),
                ],
                name="TARGET",
            )
        )

        numFluxes = sum(len(target.fiberFlux) for target in self)
        targetFluxIndex = np.empty(numFluxes, dtype=np.int16)
        filterName: List[str] = []
        fiberFlux = np.empty(numFluxes, dtype=np.float32)
        start = 0
        for tt, target in zip(targetId, self):
            num = len(target.fiberFlux)
            stop = start + num
            targetFluxIndex[start:stop] = tt
            filterName += list(target.fiberFlux.keys())
            fiberFlux[start:stop] = np.array(list(target.fiberFlux.values()))
            start = stop

        fits.append(
            BinTableHDU.from_columns(
                [
                    Column("targetId", "I", array=targetFluxIndex),
                    Column("filterName", "PA()", array=filterName),
                    Column("fiberFlux", "E", array=fiberFlux),
                ],
                name="TARGETFLUX",
            )
        )

        numObservations = sum(len(ss.observations) for ss in self.values())
        observationsIndex = np.empty(numObservations, dtype=np.int16)
        visit = np.empty(numObservations, dtype=np.int32)
        arm: List[str] = []
        spectrograph = np.empty(numObservations, dtype=np.int16)
        pfsDesignId = np.empty(numObservations, dtype=np.int64)
        fiberId = np.empty(numObservations, dtype=np.int32)
        pfiNominal = np.empty((numObservations, 2), dtype=float)
        pfiCenter = np.empty((numObservations, 2), dtype=float)
        obsTime: List[str] = []
        expTime = np.empty(numObservations, dtype=float)
        start = 0
        for tt, spectrum in zip(targetId, self.values()):
            observations = spectrum.observations
            num = len(observations)
            stop = start + num
            observationsIndex[start:stop] = tt
            visit[start:stop] = observations.visit
            arm += list(observations.arm)
            spectrograph[start:stop] = observations.spectrograph
            pfsDesignId[start:stop] = observations.pfsDesignId
            fiberId[start:stop] = observations.fiberId
            pfiNominal[start:stop] = observations.pfiNominal
            pfiCenter[start:stop] = observations.pfiCenter
            obsTime += list(observations.obsTime)
            expTime[start:stop] = observations.expTime
            start = stop

        fits.append(
            BinTableHDU.from_columns(
                [
                    Column("targetId", "I", array=observationsIndex),
                    Column("visit", "J", array=visit),
                    Column("arm", "PA()", array=arm),
                    Column("spectrograph", "I", array=spectrograph),
                    Column("pfsDesignId", "K", array=pfsDesignId),
                    Column("fiberId", "J", array=fiberId),
                    Column("pfiNominal", "2D", array=pfiNominal),
                    Column("pfiCenter", "2D", array=pfiCenter),
                    Column("obsTime", "PA()", array=obsTime),
                    Column("expTime", "E", array=expTime),
                ],
                name="OBSERVATIONS",
            )
        )

        lengths = [len(spectrum) for spectrum in self.values()]
        sameLengths = len(set(lengths)) == 1

        def writeComponent(component: str, hduName: str, dtype: type[object]):
            """Write a component of the spectra

            If all spectra have the same length, then we write as an image.
            Otherwise, we write as a table.

            Parameters
            ----------
            component : `str`
                Name of the component.
            hduName : `str`
                Name of the HDU.
            dtype : `type`
                Data type.
            """
            data = [getattr(spectrum, component).astype(dtype) for spectrum in self.values()]
            if len(data) == 0:
                fits.append(ImageHDU(data=np.array(data, dtype=dtype), name=hduName))
                return
            allDims = set([len(dd.shape) for dd in data])
            if len(allDims) != 1:
                raise RuntimeError(f"Data for {component} have different dimensions: {allDims}")
            dims = allDims.pop()
            if dims >= 3:
                raise RuntimeError(f"Data for {component} have too many dimensions: {dims}")
            if sameLengths and dims == 1:
                HduClass = CompImageHDU if dtype in (np.int16, np.int32, np.int64) else ImageHDU
                fits.append(HduClass(data=np.array(data), name=hduName))
                return
            code = {np.float32: "E", np.float64: "D", np.int16: "I", np.int32: "J", np.int64: "K"}[dtype]
            if dims == 1:
                table = BinTableHDU.from_columns([Column("value", f"P{code}()", array=data)], name=hduName)
            else:
                allNumRows = set([dd.shape[0] for dd in data])
                if len(allNumRows) != 1:
                    raise RuntimeError(f"Data for {component} have different numbers of rows: {allNumRows}")
                numRows = allNumRows.pop()
                columns = [
                    Column(f"row_{ii}", f"P{code}()", array=[dd[ii] for dd in data]) for ii in range(numRows)
                ]
                table = BinTableHDU.from_columns(columns, name=hduName)
            fits.append(table)

        # Wavelength: we can write a single array if everything has the length and value
        wavelength = [spectrum.wavelength for spectrum in self.values()]
        if sameLengths and all(np.all(wl == wavelength[0]) for wl in wavelength):
            fits.append(ImageHDU(data=wavelength[0], name="WAVELENGTH"))
        else:
            writeComponent("wavelength", "WAVELENGTH", np.float64)

        writeComponent("flux", "FLUX", np.float32)
        writeComponent("mask", "MASK", np.int32)
        writeComponent("sky", "SKY", np.float32)
        writeComponent("covar", "COVAR", np.float32)

        haveCovar2 = [spectrum.covar2 is not None for spectrum in self.values()]
        if len(set(haveCovar2)) == 2:
            raise RuntimeError("covar2 must be uniformly populated")
        if any(haveCovar2):
            fits.append(ImageHDU(data=[spectrum.covar2 for spectrum in self.values()], name="COVAR2"))

        # Metadata table
        metadata: List[str] = []
        for spectrum in self.values():
            md = spectrum.metadata.copy()
            md.update(spectrum.flags.toFitsHeader())
            metadata.append(yaml.dump(md))
        fits.append(
            BinTableHDU.from_columns(
                [
                    Column("targetId", "I", array=targetId),
                    Column("metadata", "PA()", array=metadata),
                ],
                name="METADATA",
            )
        )

        fits.append(
            BinTableHDU.from_columns(
                [
                    Column("targetId", "I", array=targetId),
                    Column(
                        "wavelength",
                        "QD()",
                        array=[
                            spectrum.fluxTable.wavelength if spectrum.fluxTable else []
                            for spectrum in self.values()
                        ],
                    ),
                    Column(
                        "flux",
                        "QE()",
                        array=[
                            spectrum.fluxTable.flux if spectrum.fluxTable else []
                            for spectrum in self.values()
                        ],
                    ),
                    Column(
                        "error",
                        "QE()",
                        array=[
                            spectrum.fluxTable.error if spectrum.fluxTable else []
                            for spectrum in self.values()
                        ],
                    ),
                    Column(
                        "mask",
                        "QJ()",
                        array=[
                            spectrum.fluxTable.mask if spectrum.fluxTable else []
                            for spectrum in self.values()
                        ],
                    ),
                ],
                name="FLUXTABLE",
            )
        )

        notes = self.NotesClass.empty(len(self))
        for ii, spectrum in enumerate(self.values()):
            notes.setRow(ii, **spectrum.notes.getDict())
        notes.writeHdu(fits)

        with open(filename, "wb") as fd:
            fits.writeto(fd)
