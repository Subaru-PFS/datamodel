from collections.abc import Mapping
from typing import Dict, Iterator, Iterable, List, Optional, Tuple, Type, Union, cast, overload

import astropy.io.fits
import numpy as np
import yaml
from astropy.io.fits import BinTableHDU, Column, HDUList, ImageHDU

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
    spectra : iterable of `PfsFiberArray`
        Spectra to be indexed by target.
    """

    PfsFiberArrayClass: Type[PfsFiberArray]  # Subclasses must override
    NotesClass: Type[PfsTable]  # Subclasses must override

    def __init__(self, spectra: Iterable[PfsFiberArray]):
        super().__init__()
        self.spectra: Dict[Target, PfsFiberArray] = {spectrum.target: spectrum for spectrum in spectra}
        if len(self.spectra) != len(spectra):
            raise RuntimeError("Spectra targets not unique")

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
    def readFits(cls, filename: str) -> "PfsTargetSpectra":
        """Read from FITS file

        Parameters
        ----------
        filename : `str`
            Filename of FITS file.

        Returns
        -------
        self : ``cls``
            Constructed instance, from FITS file.
        """
        spectra = []
        with astropy.io.fits.open(filename) as fits:
            targetHdu = fits["TARGET"].data
            targetFluxHdu = fits["TARGETFLUX"].data
            observationsHdu = fits["OBSERVATIONS"].data
            wavelengthHdu = fits["WAVELENGTH"].data
            fluxHdu = fits["FLUX"].data
            maskHdu = fits["MASK"].data
            skyHdu = fits["SKY"].data
            covarHdu = fits["COVAR"].data
            covar2Hdu = fits["COVAR2"].data if "COVAR2" in fits else None
            metadataHdu = fits["METADATA"].data
            fluxTableHdu = fits["FLUXTABLE"].data
            notesTable = cls.NotesClass.readHdu(fits)

            for ii, row in enumerate(targetHdu):
                targetId = row["targetId"]
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
                observations = Observations(
                    observationsHdu.visit[select],
                    ["".join(np.char.decode(ss.astype("S"))) for ss in observationsHdu.arm[select]],
                    observationsHdu.spectrograph[select],
                    observationsHdu.pfsDesignId[select],
                    observationsHdu.fiberId[select],
                    observationsHdu.pfiNominal[select],
                    observationsHdu.pfiCenter[select],
                )

                metadataRow = metadataHdu[ii]
                assert metadataRow["targetId"] == targetId

                metadata = yaml.load(
                    # This complicated conversion is required in order to preserve the newlines
                    "".join(np.char.decode(metadataRow["metadata"].astype("S"))),
                    Loader=yaml.SafeLoader,
                )
                flags = MaskHelper.fromFitsHeader(metadata, strip=True)

                fluxTableRow = fluxTableHdu[ii]
                assert fluxTableRow["targetId"] == targetId
                fluxTable = FluxTable(
                    fluxTableRow["wavelength"],
                    fluxTableRow["flux"],
                    fluxTableRow["error"],
                    fluxTableRow["mask"],
                    flags,
                )

                notes = cls.PfsFiberArrayClass.NotesClass(
                    **{col.name: getattr(notesTable, col.name)[ii] for col in notesTable.schema}
                )

                spectrum = cls.PfsFiberArrayClass(
                    target,
                    observations,
                    wavelengthHdu[ii],
                    fluxHdu[ii],
                    maskHdu[ii],
                    skyHdu[ii],
                    covarHdu[ii],
                    covar2Hdu[ii] if covar2Hdu is not None else [],
                    flags,
                    metadata,
                    fluxTable,
                    notes,
                )
                spectra.append(spectrum)

        return cls(spectra)

    def writeFits(self, filename: str):
        """Write to FITS file

        This API is intended for use by the LSST data butler, which handles
        translating the desired identity into a filename.

        Parameters
        ----------
        filename : `str`
            Filename of FITS file.
        """
        fits = HDUList()

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
                ],
                name="OBSERVATIONS",
            )
        )

        fits.append(ImageHDU(data=[spectrum.wavelength for spectrum in self.values()], name="WAVELENGTH"))
        fits.append(ImageHDU(data=[spectrum.flux for spectrum in self.values()], name="FLUX"))
        fits.append(ImageHDU(data=[spectrum.mask for spectrum in self.values()], name="MASK"))
        fits.append(ImageHDU(data=[spectrum.sky for spectrum in self.values()], name="SKY"))
        fits.append(ImageHDU(data=[spectrum.covar for spectrum in self.values()], name="COVAR"))
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
                        "PD()",
                        array=[
                            spectrum.fluxTable.wavelength if spectrum.fluxTable else []
                            for spectrum in self.values()
                        ],
                    ),
                    Column(
                        "flux",
                        "PD()",
                        array=[
                            spectrum.fluxTable.flux if spectrum.fluxTable else []
                            for spectrum in self.values()
                        ],
                    ),
                    Column(
                        "error",
                        "PD()",
                        array=[
                            spectrum.fluxTable.error if spectrum.fluxTable else []
                            for spectrum in self.values()
                        ],
                    ),
                    Column(
                        "mask",
                        "PJ()",
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
