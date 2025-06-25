from collections.abc import Mapping
from typing import Dict, Iterator, Optional, Sequence, Tuple, Type, Union, cast, overload, ClassVar

import numpy as np

import astropy
from astropy.table import Table
from .pfsZCandidates import PfsZCandidates
from .target import Target
from .pfsConfig import TargetType


__all__ = ["PfsCoZCandidates",]


class PfsCoZCandidates(Mapping[Target, PfsZCandidates]):
    """A collection of `PfsZCandidates` indexed by target

    Parameters
    ----------
    spectra : `list` [ `PfsZCandidates` ]
        Redshift Candidates to be indexed by target.
    zgrids: `dict` [`str` , `np.ndarray`]
    """

    PfsZCandidatesClass: ClassVar[Type[PfsZCandidates]]

    def __init__(self, spectra: Sequence[PfsZCandidates], zgrids: dict):
        super().__init__()
        self.zCandidates: Dict[Target, PfsZCandidates] = {spectrum.target: spectrum for spectrum in spectra}

        if len(self.zCandidates) != len(spectra):
            raise RuntimeError("Spectra targets not unique")

        self.zgrids = zgrids
        # Lookup by catId,objId
        self._byCatIdObjId: Dict[Tuple[int, int], PfsZCandidates]
        self._byCatIdObjId = {(ss.target.catId, ss.target.objId): ss for ss in spectra}
        if len(self._byCatIdObjId) != len(spectra):
            raise RuntimeError("Spectra catId,objId not unique")

        # Lookup by objId (only when all objIds are unique)
        self._byObjId: Optional[Dict[int, PfsZCandidates]]
        self._byObjId = {tt.objId: ss for tt, ss in self.zCandidates.items()}
        if len(self._byObjId) != len(self.zCandidates):
            # Disable lookup by objId
            self._byObjId = None

    @overload
    def __getitem__(self, objId: int) -> PfsZCandidates:
        ...

    @overload
    def __getitem__(self, catId: int, objId: int) -> PfsZCandidates:
        ...

    @overload
    def __getitem__(self, catIdObjId: Tuple[int, int]) -> PfsZCandidates:
        ...

    @overload
    def __getitem__(self, target: Target) -> PfsZCandidates:
        ...

    def __getitem__(
        self, first: Union[int, Target, Tuple[int, int]], second: Optional[int] = None
    ) -> PfsZCandidates:
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
        return self.zCandidates[first]

    def __iter__(self) -> Iterator[Target]:
        """Return iterator over targets"""
        return iter(self.zCandidates)

    def __len__(self) -> int:
        """Return number of spectra"""
        return len(self.zCandidates)

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
        return target in self.zCandidates

    @classmethod
    def readFits(cls, filename: str) -> "PfsCoZCandidates":
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

        def get_at_targetId(data: Table, targetId: np.int16):
            """Extract data from FITS table at a certain targetId

            Parameters
            ----------
            data : `Table`
                fits table
            targetId : `np.int16`
                TargetId

            Returns
            -------
            data : `np.ndarray` or list of `np.ndarray`
                Data read from FITS file.
            """
            mask = data['targetId'] == targetId
            ret = data[mask]
            if len(ret) == 1:
                return ret[0]
            else:
                return ret

        def get_models_at_targetId(modelsCo: Table, zCands: Table):
            """Extract models from FITS table at a certain targetId

            Parameters
            ----------
            modelsCo : `Table`
                fits table
            zCands : `Table`
                Table containing the z candidates for a certain targetId

            Returns
            -------
            data : `list` [`np.ndarray`]
                Data read from FITS file.
            """
            if len(zCands) > 0:
                imin = zCands["modelId"].min()
                imax = zCands["modelId"].max()
                return modelsCo[imin:imax+1]
            else:
                return []

        zCandidates = []
        with astropy.io.fits.open(filename) as hdul:
            targetHdu = hdul["TARGET"].data
            warnings_co = Table(hdul["WARNINGS"].data)
            errors_co = Table(hdul["ERRORS"].data)
            classification_co = Table(hdul["CLASSIFICATION"].data)
            zcands_co = dict()
            models_co = dict()
            ln_pdf_co = dict()
            z_grid = dict()
            lines_co = dict()
            grid_name = {"GALAXY": "REDSHIFT",
                         "QSO": "REDSHIFT",
                         "STAR": "VELOCITY"}

            for o in ["GALAXY", "QSO", "STAR"]:
                zcands_co[o] = Table(hdul[f"{o}_CANDIDATES"].data)
                models_co[o] = hdul[f"{o}_MODELS"].data
                hdu_grid = f"{o}_{grid_name[o]}_GRID"
                z_grid[o] = hdul[hdu_grid].data[grid_name[o].lower()]
                ln_pdf_co[o] = hdul[f"{o}_LN_PDF"].data
                if o != "STAR":
                    lines_co[o] = Table(hdul[f"{o}_LINES"].data)
            for ii, row in enumerate(targetHdu):
                targetId = row["targetId"]
                target = Target(
                    row["catId"],
                    row["tract"],
                    "".join(row["patch"]),
                    row["objId"],
                    row["ra"],
                    row["dec"],
                    TargetType(row["targetType"]),
                    fiberFlux=0,
                )
                warnings = get_at_targetId(warnings_co, targetId)
                errors = get_at_targetId(errors_co, targetId)
                classification = get_at_targetId(classification_co, targetId)
                zcands = dict()
                models = dict()
                ln_pdf = dict()
                lines = dict()
                for o in ["GALAXY", "QSO", "STAR"]:
                    zcands[o] = get_at_targetId(zcands_co[o], targetId)
                    models[o] = get_models_at_targetId(models_co[o],
                                                       zcands[o])
                    ln_pdf[o] = ln_pdf_co[o][targetId]
                    if o != "STAR":
                        lines[o] = get_at_targetId(lines_co[o], targetId)
                zCandidates.append(PfsZCandidates(target,
                                                  errors,
                                                  warnings,
                                                  classification,
                                                  zcands,
                                                  models,
                                                  ln_pdf,
                                                  lines))
            return cls(zCandidates, z_grid)
