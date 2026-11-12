"""Line-spread functions: plain containers, and their FITS representation

These hold the numbers that describe the line-spread function (LSF) of a set
of spectra, so that they can be written to and read from FITS without the
code that evaluates them (which is in drp_stella).

- `PfsArmLsfData`: the LSF over one detector (arm, spectrograph). Each
  parameter of the profile is a smooth function (`PfsLsfSurface`) of position
  on the detector, and each fibre's trace position is a spline in row.
- `PfsLsfInput`: how one input spectrum enters a combined (merged or coadded)
  spectrum: the map from output to input pixels, and its smooth weight.
- `PfsLsf`: the LSF of a set of spectra. For a pfsArm it is one detector's
  LSF; for a pfsMerged or pfsCoadd it is the LSFs of the detectors of the
  inputs, and how each input enters each spectrum.

The HDUs are described in ``datamodel.txt`` ("Line-spread functions").
"""
import json
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.io.fits import BinTableHDU, Column, HDUList, PrimaryHDU

__all__ = ("PfsLsfSurface", "PfsArmLsfData", "PfsLsfInput", "PfsLsf", "lsfToDatamodel")

LSF_VERSION = 1  # Version of the LSF HDUs' format


def lsfToDatamodel(lsf, keys: Optional[List] = None) -> Optional["PfsLsf"]:
    """The datamodel representation of an LSF attached to spectra

    Parameters
    ----------
    lsf : `PfsLsf`, or an object with a ``toDatamodel`` method, or `None`
        LSF (e.g. drp_stella's, which knows how to convert itself).
    keys : `list`, optional
        The spectra in the order in which they are written, if they are
        identified by their index (e.g. the targets of a pfsCoadd); passed to
        ``toDatamodel``.

    Returns
    -------
    lsf : `PfsLsf` or `None`
        Datamodel representation, or `None` if there's no LSF.
    """
    if lsf is None or isinstance(lsf, PfsLsf):
        return lsf
    return lsf.toDatamodel() if keys is None else lsf.toDatamodel(keys)


def _jsonDefault(value):
    """Convert numpy types for JSON"""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


@dataclass
class PfsLsfSurface:
    """A parameter of the LSF as a smooth function over the detector

    A two-dimensional Chebyshev series in detector coordinates ``(u, v)``,
    plus constant offsets for groups (e.g. a fibre's MTP sub-bundle).

    Parameters
    ----------
    name : `str`
        Parameter name (e.g. ``rms``).
    degree : (`int`, `int`)
        Chebyshev degree in ``(u, v)``.
    domain : (`float`, `float`, `float`, `float`)
        Ranges ``(uMin, uMax, vMin, vMax)`` mapped onto ``[-1, 1]``.
    coeffs : `numpy.ndarray` of `float`
        Coefficients, in the order of `numpy.polynomial.chebyshev.chebvander2d`.
    offsets : `dict` [`str`, `dict` [`str`, (`float`, `float`)]]
        For each grouping factor, the offset (and its error) of each group.
    reference : `dict` [`str`, `str` or `None`]
        Each factor's reference group, whose offset was zero in the fit
        (the offsets may have been recentred since, e.g. to average zero).
    rms, chi2 : `float`
        RMS residual and chi^2 of the fit.
    ndof, nUsed : `int`
        Degrees of freedom of, and number of points used in, the fit.
    """

    name: str
    degree: Tuple[int, int]
    domain: Tuple[float, float, float, float]
    coeffs: np.ndarray
    offsets: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=dict)
    reference: Dict[str, Optional[str]] = field(default_factory=dict)
    rms: float = np.nan
    chi2: float = np.nan
    ndof: int = 0
    nUsed: int = 0


@dataclass
class PfsArmLsfData:
    """The LSF over one detector

    Parameters
    ----------
    arm : `str`
        Arm (``b``, ``r``, ``m``, ``n``).
    spectrograph : `int`
        Spectrograph module.
    length : `int`
        Number of rows (pixels) in each spectrum.
    kind : `str`
        ``disk`` (a disk convolved with a blur; see the surfaces), or
        ``gaussian`` (a Gaussian of width ``gaussianWidth`` for each fibre).
    fiberId : `numpy.ndarray` of `int`
        Fibres covered.
    surfaces : `dict` [`str`, `PfsLsfSurface`]
        The parameters, as smooth functions over the detector (``disk``).
    extraction : `str`
        How the spectra were extracted: ``optimal`` or ``boxcar``.
    blurOrder : `int`
        Edgeworth order of the blur (2 is a Gaussian).
    centroidZeroPoint : `float`
        Offset between the data's and the free fits' first moments (pixels).
    xSplit : `float`
        Pixels with ``x >= xSplit`` are on the right-hand CCD.
    affine : `numpy.ndarray` of `float`, shape ``(3, 2)``, or `None`
        Transform from the right-hand CCD's pixels to the detector frame
        (``[x, y, 1] @ affine``); `None` for a monolithic detector.
    affineMaxError : `float`
        Largest error of the affine transform (pixels).
    isEngineering : `numpy.ndarray` of `bool`
        Which fibres are engineering (IIS) fibres.
    fiberGroups : `dict` [`str`, `dict` [`int`, `str`]]
        Per-fibre group labels (e.g. ``{"mtp": {fiberId: "U3-1"}}``) whose
        offsets are part of the model.
    iisExtraBlur : (`float`, `float`) or `None`
        Extra blur of the engineering fibres: variance (pixels^2) and fourth
        cumulant (pixels^4).
    gaussianWidth : `numpy.ndarray` of `float` or `None`
        Width (pixels) of each fibre's LSF (``gaussian``).
    traceKnots : `numpy.ndarray` of `float`
        Rows of the knots of the fibres' trace splines.
    traceValues : `numpy.ndarray` of `float`, shape ``(numFibers, numKnots)``
        Each fibre's trace position x (pixels) at the knots.
    wavelengthKnots : `numpy.ndarray` of `float` or `None`
        Rows of the knots of the fibres' wavelength splines (omitted when the
        LSF is stored with spectra, which have their own wavelengths).
    wavelengthValues : `numpy.ndarray` of `float` or `None`
        Each fibre's vacuum wavelength (nm) at the knots.
    metadata : `dict`
        Provenance (anything JSON can represent).
    """

    arm: str
    spectrograph: int
    length: int
    kind: str
    fiberId: np.ndarray
    surfaces: Dict[str, PfsLsfSurface] = field(default_factory=dict)
    extraction: str = "optimal"
    blurOrder: int = 2
    centroidZeroPoint: float = 0.0
    xSplit: float = np.inf
    affine: Optional[np.ndarray] = None
    affineMaxError: float = 0.0
    isEngineering: Optional[np.ndarray] = None
    fiberGroups: Dict[str, Dict[int, str]] = field(default_factory=dict)
    iisExtraBlur: Optional[Tuple[float, float]] = None
    gaussianWidth: Optional[np.ndarray] = None
    traceKnots: Optional[np.ndarray] = None
    traceValues: Optional[np.ndarray] = None
    wavelengthKnots: Optional[np.ndarray] = None
    wavelengthValues: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.fiberId = np.asarray(self.fiberId, dtype=int)
        if self.isEngineering is None:
            self.isEngineering = np.zeros(len(self.fiberId), dtype=bool)
        self.isEngineering = np.asarray(self.isEngineering, dtype=bool)


@dataclass
class PfsLsfInput:
    """How one input spectrum enters one spectrum of a combination

    Parameters
    ----------
    spectrum : `int`
        The combined spectrum: its ``fiberId`` (pfsMerged), or the index of
        its target (pfsCoadd).
    armIndex : `int`
        Index (in `PfsLsf.arms`) of the input's detector LSF.
    fiberId : `int`
        Fibre of the input.
    order : `int`
        Order of the resampling of the input (as for the flux).
    indexPositions : `numpy.ndarray` of `int`
        Output pixels at which the map to input pixels is sampled.
    indexSamples : `numpy.ndarray` of `float`
        Input pixel at each of ``indexPositions`` (interpolated with a cubic
        spline).
    indexMaxError : `float`
        Maximum error of that interpolation (input pixels).
    weightMinIndex, weightMaxIndex : `int`
        Range of output pixels covered by the input.
    weightBinSize : `int`
        Size of the bins (output pixels) of the smooth weight.
    weightValues : `numpy.ndarray` of `float`
        The input's weight in each bin (interpolated linearly between the
        bins' centres).
    """

    spectrum: int
    armIndex: int
    fiberId: int
    order: int
    indexPositions: np.ndarray
    indexSamples: np.ndarray
    indexMaxError: float
    weightMinIndex: int
    weightMaxIndex: int
    weightBinSize: int
    weightValues: np.ndarray


@dataclass
class PfsLsf:
    """The LSF of a set of spectra

    Parameters
    ----------
    arms : `list` of `PfsArmLsfData`
        Detector LSFs.
    inputs : `list` of `PfsLsfInput`
        For a combination (pfsMerged, pfsCoadd), how each input enters each
        spectrum; empty for the spectra of a single detector (pfsArm), whose
        LSF is ``arms[0]`` at their own wavelengths.
    """

    arms: List[PfsArmLsfData]
    inputs: List[PfsLsfInput] = field(default_factory=list)

    @classmethod
    def merge(cls, lsfList: List["PfsLsf"]) -> "PfsLsf":
        """Merge the LSFs of several sets of spectra (e.g. spectrographs)

        The sets must have different spectra (e.g. ``fiberId``).
        """
        arms: List[PfsArmLsfData] = []
        inputs: List[PfsLsfInput] = []
        for lsf in lsfList:
            offset = len(arms)
            arms += lsf.arms
            inputs += [replace(ii, armIndex=ii.armIndex + offset) for ii in lsf.inputs]
        return cls(arms, inputs)

    @property
    def isCombined(self) -> bool:
        """Is this the LSF of a combination of spectra?"""
        return len(self.inputs) > 0

    def toFits(self, fits: HDUList) -> None:
        """Append the LSF's HDUs to a FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file to which to append.
        """
        arms = self.arms
        groupFactors = sorted({ff for arm in arms for ff in arm.fiberGroups})

        armHdu = BinTableHDU.from_columns([
            Column("armIndex", "J", array=np.arange(len(arms))),
            _stringColumn("arm", [arm.arm for arm in arms]),
            Column("spectrograph", "J", array=[arm.spectrograph for arm in arms]),
            Column("length", "J", array=[arm.length for arm in arms]),
            _stringColumn("kind", [arm.kind for arm in arms]),
            _stringColumn("extraction", [arm.extraction for arm in arms]),
            Column("blurOrder", "J", array=[arm.blurOrder for arm in arms]),
            Column("centroidZeroPoint", "D", array=[arm.centroidZeroPoint for arm in arms]),
            Column("xSplit", "D", array=[arm.xSplit for arm in arms]),
            Column("affine", "6D", array=[np.full(6, np.nan) if arm.affine is None else
                                          np.asarray(arm.affine, dtype=float).ravel() for arm in arms]),
            Column("affineMaxError", "D", array=[arm.affineMaxError for arm in arms]),
            Column("iisExtraBlur", "2D", array=[np.full(2, np.nan) if arm.iisExtraBlur is None else
                                                np.asarray(arm.iisExtraBlur, dtype=float) for arm in arms]),
            Column("traceKnots", "PD()", array=[_orEmpty(arm.traceKnots) for arm in arms]),
            Column("wavelengthKnots", "PD()", array=[_orEmpty(arm.wavelengthKnots) for arm in arms]),
            _stringColumn("metadata", [json.dumps(arm.metadata, default=_jsonDefault) for arm in arms]),
        ], name="LSF_ARM")
        armHdu.header["LSF_VER"] = (LSF_VERSION, "Version of the LSF HDUs")
        fits.append(armHdu)

        surfaces = [(ii, ss) for ii, arm in enumerate(arms) for ss in arm.surfaces.values()]
        fits.append(BinTableHDU.from_columns([
            Column("armIndex", "J", array=[ii for ii, _ in surfaces]),
            _stringColumn("name", [ss.name for _, ss in surfaces]),
            Column("degree", "2J", array=_fixed([ss.degree for _, ss in surfaces], 2, np.int32)),
            Column("domain", "4D", array=_fixed([np.ravel(ss.domain) for _, ss in surfaces], 4, float)),
            Column("coeffs", "PD()", array=[np.asarray(ss.coeffs, dtype=float) for _, ss in surfaces]),
            Column("rms", "D", array=[ss.rms for _, ss in surfaces]),
            Column("chi2", "D", array=[ss.chi2 for _, ss in surfaces]),
            Column("ndof", "J", array=[ss.ndof for _, ss in surfaces]),
            Column("nUsed", "J", array=[ss.nUsed for _, ss in surfaces]),
        ], name="LSF_SURFACE"))

        offsets = []
        for ii, ss in surfaces:
            for factor in sorted(set(ss.offsets) | set(ss.reference)):
                table = ss.offsets.get(factor, {})
                reference = ss.reference.get(factor)
                labels = sorted(set(table) | ({reference} if reference is not None else set()))
                if not labels:  # A factor with no groups: record its existence (and lack of reference)
                    offsets.append((ii, ss.name, factor, "", 0.0, 0.0, False, False))
                for label in labels:
                    value, error = table.get(label, (0.0, 0.0))
                    isReference = label == reference
                    offsets.append((ii, ss.name, factor, label, value, error, isReference, label in table))
        fits.append(BinTableHDU.from_columns([
            Column("armIndex", "J", array=[oo[0] for oo in offsets]),
            _stringColumn("surface", [oo[1] for oo in offsets]),
            _stringColumn("factor", [oo[2] for oo in offsets]),
            _stringColumn("label", [oo[3] for oo in offsets]),
            Column("value", "D", array=[oo[4] for oo in offsets]),
            Column("error", "D", array=[oo[5] for oo in offsets]),
            Column("isReference", "L", array=[oo[6] for oo in offsets]),
            Column("hasOffset", "L", array=[oo[7] for oo in offsets]),
        ], name="LSF_OFFSET"))

        rows = [(ii, jj) for ii, arm in enumerate(arms) for jj in range(len(arm.fiberId))]
        columns = [
            Column("armIndex", "J", array=[ii for ii, _ in rows]),
            Column("fiberId", "J", array=[arms[ii].fiberId[jj] for ii, jj in rows]),
            Column("isEngineering", "L", array=[arms[ii].isEngineering[jj] for ii, jj in rows]),
            Column("gaussianWidth", "D", array=[np.nan if arms[ii].gaussianWidth is None else
                                                arms[ii].gaussianWidth[jj] for ii, jj in rows]),
            Column("traceValues", "PD()", array=[_rowOrEmpty(arms[ii].traceValues, jj) for ii, jj in rows]),
            Column("wavelengthValues", "PD()",
                   array=[_rowOrEmpty(arms[ii].wavelengthValues, jj) for ii, jj in rows]),
        ]
        for factor in groupFactors:
            labels = [arms[ii].fiberGroups.get(factor, {}).get(int(arms[ii].fiberId[jj]), "")
                      for ii, jj in rows]
            columns.append(_stringColumn(f"group_{factor}", labels))
        fits.append(BinTableHDU.from_columns(columns, name="LSF_FIBER"))

        if self.inputs:
            inputs = self.inputs
            fits.append(BinTableHDU.from_columns([
                Column("spectrum", "J", array=[ii.spectrum for ii in inputs]),
                Column("armIndex", "J", array=[ii.armIndex for ii in inputs]),
                Column("fiberId", "J", array=[ii.fiberId for ii in inputs]),
                Column("order", "J", array=[ii.order for ii in inputs]),
                Column("indexPositions", "PJ()", array=[np.asarray(ii.indexPositions, dtype=np.int32)
                                                        for ii in inputs]),
                Column("indexSamples", "PD()", array=[np.asarray(ii.indexSamples, dtype=float)
                                                      for ii in inputs]),
                Column("indexMaxError", "D", array=[ii.indexMaxError for ii in inputs]),
                Column("weightMinIndex", "J", array=[ii.weightMinIndex for ii in inputs]),
                Column("weightMaxIndex", "J", array=[ii.weightMaxIndex for ii in inputs]),
                Column("weightBinSize", "J", array=[ii.weightBinSize for ii in inputs]),
                Column("weightValues", "PE()", array=[np.asarray(ii.weightValues, dtype=np.float32)
                                                      for ii in inputs]),
            ], name="LSF_INPUT"))

    @classmethod
    def fromFits(cls, fits: HDUList) -> Optional["PfsLsf"]:
        """Read the LSF from a FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file from which to read.

        Returns
        -------
        lsf : `PfsLsf` or `None`
            The LSF, or `None` if the file has none.
        """
        if "LSF_ARM" not in fits:
            return None
        # Read whole columns: selecting rows of a FITS table loses its variable-length arrays
        armTable = _columns(fits["LSF_ARM"])
        fiberTable = _columns(fits["LSF_FIBER"])
        surfaceTable = _columns(fits["LSF_SURFACE"])
        offsetTable = _columns(fits["LSF_OFFSET"])
        groupFactors = [name[len("group_"):] for name in fiberTable if name.startswith("group_")]

        arms = []
        for ii in range(len(armTable["armIndex"])):
            rows = np.flatnonzero(np.asarray(fiberTable["armIndex"]) == ii)
            fiberId = np.array([fiberTable["fiberId"][jj] for jj in rows], dtype=int)
            affine = np.asarray(armTable["affine"][ii], dtype=float)
            iisExtraBlur = np.asarray(armTable["iisExtraBlur"][ii], dtype=float)
            traceValues = [np.asarray(fiberTable["traceValues"][jj], dtype=float) for jj in rows]
            wavelengthValues = [np.asarray(fiberTable["wavelengthValues"][jj], dtype=float) for jj in rows]
            gaussianWidth = np.array([fiberTable["gaussianWidth"][jj] for jj in rows], dtype=float)

            surfaces = {}
            for kk in np.flatnonzero(np.asarray(surfaceTable["armIndex"]) == ii):
                name = str(surfaceTable["name"][kk])
                offsets: Dict[str, Dict[str, Tuple[float, float]]] = {}
                reference: Dict[str, Optional[str]] = {}
                for oo in range(len(offsetTable["armIndex"])):
                    if offsetTable["armIndex"][oo] != ii or str(offsetTable["surface"][oo]) != name:
                        continue
                    factor, label = str(offsetTable["factor"][oo]), str(offsetTable["label"][oo])
                    reference.setdefault(factor, None)
                    if label == "":
                        continue
                    if offsetTable["isReference"][oo]:
                        reference[factor] = label
                    if not offsetTable["hasOffset"][oo]:
                        continue
                    offsets.setdefault(factor, {})[label] = (float(offsetTable["value"][oo]),
                                                             float(offsetTable["error"][oo]))
                surfaces[name] = PfsLsfSurface(
                    name=name,
                    degree=tuple(int(dd) for dd in surfaceTable["degree"][kk]),
                    domain=tuple(float(dd) for dd in surfaceTable["domain"][kk]),
                    coeffs=np.asarray(surfaceTable["coeffs"][kk], dtype=float),
                    offsets=offsets,
                    reference=reference,
                    rms=float(surfaceTable["rms"][kk]),
                    chi2=float(surfaceTable["chi2"][kk]),
                    ndof=int(surfaceTable["ndof"][kk]),
                    nUsed=int(surfaceTable["nUsed"][kk]),
                )

            fiberGroups = {}
            for factor in groupFactors:
                labels = {int(ff): str(fiberTable[f"group_{factor}"][jj]) for ff, jj in zip(fiberId, rows)
                          if str(fiberTable[f"group_{factor}"][jj])}
                if labels:
                    fiberGroups[factor] = labels

            arms.append(PfsArmLsfData(
                arm=str(armTable["arm"][ii]),
                spectrograph=int(armTable["spectrograph"][ii]),
                length=int(armTable["length"][ii]),
                kind=str(armTable["kind"][ii]),
                fiberId=fiberId,
                surfaces=surfaces,
                extraction=str(armTable["extraction"][ii]),
                blurOrder=int(armTable["blurOrder"][ii]),
                centroidZeroPoint=float(armTable["centroidZeroPoint"][ii]),
                xSplit=float(armTable["xSplit"][ii]),
                affine=None if np.all(np.isnan(affine)) else affine.reshape(3, 2),
                affineMaxError=float(armTable["affineMaxError"][ii]),
                isEngineering=np.array([fiberTable["isEngineering"][jj] for jj in rows], dtype=bool),
                fiberGroups=fiberGroups,
                iisExtraBlur=(None if np.all(np.isnan(iisExtraBlur)) else
                              tuple(float(vv) for vv in iisExtraBlur)),
                gaussianWidth=None if np.all(np.isnan(gaussianWidth)) else gaussianWidth,
                traceKnots=_noneIfEmpty(armTable["traceKnots"][ii]),
                traceValues=np.array(traceValues) if traceValues and len(traceValues[0]) > 0 else None,
                wavelengthKnots=_noneIfEmpty(armTable["wavelengthKnots"][ii]),
                wavelengthValues=(np.array(wavelengthValues)
                                  if wavelengthValues and len(wavelengthValues[0]) > 0 else None),
                metadata=json.loads(str(armTable["metadata"][ii])),
            ))

        inputs = []
        if "LSF_INPUT" in fits:
            table = _columns(fits["LSF_INPUT"])
            for ii in range(len(table["spectrum"])):
                inputs.append(PfsLsfInput(
                    spectrum=int(table["spectrum"][ii]),
                    armIndex=int(table["armIndex"][ii]),
                    fiberId=int(table["fiberId"][ii]),
                    order=int(table["order"][ii]),
                    indexPositions=np.asarray(table["indexPositions"][ii], dtype=int),
                    indexSamples=np.asarray(table["indexSamples"][ii], dtype=float),
                    indexMaxError=float(table["indexMaxError"][ii]),
                    weightMinIndex=int(table["weightMinIndex"][ii]),
                    weightMaxIndex=int(table["weightMaxIndex"][ii]),
                    weightBinSize=int(table["weightBinSize"][ii]),
                    weightValues=np.asarray(table["weightValues"][ii], dtype=np.float32),
                ))
        return cls(arms, inputs)

    def writeFits(self, filename: str) -> None:
        """Write to a FITS file of its own (e.g. a detector's LSF)"""
        fits = HDUList([PrimaryHDU()])
        self.toFits(fits)
        with open(filename, "wb") as fd:
            fits.writeto(fd)

    @classmethod
    def readFits(cls, filename: str) -> "PfsLsf":
        """Read from a FITS file of its own"""
        from astropy.io.fits import open as fitsOpen
        with fitsOpen(filename, memmap=False) as fits:
            lsf = cls.fromFits(fits)
        if lsf is None:
            raise RuntimeError(f"No LSF in {filename}")
        return lsf


def _fixed(values: list, width: int, dtype: type) -> np.ndarray:
    """A 2-D array for a fixed-width FITS array column (which may have no rows)"""
    return np.array(values, dtype=dtype).reshape(len(values), width)


def _stringColumn(name: str, values: List[str]) -> Column:
    """A FITS column of strings, as wide as the longest

    Variable-length string columns lose their spaces when read.
    """
    values = [str(vv) for vv in values]
    width = max([len(vv.encode()) for vv in values] + [1])
    return Column(name, f"{width}A", array=values)


def _columns(hdu: BinTableHDU) -> Dict[str, list]:
    """All the columns of a FITS table, as lists (keeping variable-length arrays)"""
    data = hdu.data
    return {name: list(data[name]) for name in data.columns.names}


def _orEmpty(array: Optional[np.ndarray]) -> np.ndarray:
    """An array, or an empty one for `None`"""
    return np.zeros(0) if array is None else np.asarray(array, dtype=float)


def _rowOrEmpty(array: Optional[np.ndarray], index: int) -> np.ndarray:
    """Row of an array, or an empty one if the array is `None`"""
    return np.zeros(0) if array is None else np.asarray(array[index], dtype=float)


def _noneIfEmpty(array) -> Optional[np.ndarray]:
    """An array, or `None` if it is empty"""
    array = np.asarray(array, dtype=float)
    return None if array.size == 0 else array
