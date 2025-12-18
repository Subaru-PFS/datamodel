import astropy
from enum import Flag, auto, Enum


objects = ["galaxy", "qso", "star"]

stage_to_pfs_sym = {"redshiftSolver": "Z",
                    "lineMeasSolver": "L"}


class ZLWarning(Flag):
    """Enumeration of warning flags used in redshift
    and line measurement solvers.

    These flags indicate various non-fatal issues or assumptions made during
    redshift estimation or line fitting procedures.
    """

    AIR_VACUUM_CONVERSION_IGNORED = auto()
    PDF_PEAK_NOT_FOUND = auto()
    ESTIMATED_STD_FAR_FROM_INPUT = auto()
    LINEMATCHING_REACHED_ENDLOOP = auto()
    FORCED_IGNORELINESUPPORT_TO_FALSE = auto()
    FORCED_CONTINUUM_COMPONENT_TO_FROMSPECTRUM = auto()
    AIR_VACUUM_REACHED_MAX_ITERATIONS = auto()
    ASYMFIT_NAN_PARAMS = auto()
    DELTAZ_COMPUTATION_FAILED = auto()
    INVALID_FOLDER_PATH = auto()
    FORCED_CONTINUUM_TO_NOCONTINUUM = auto()
    FORCED_CONTINUUM_REESTIMATION_TO_NO = auto()
    LESS_OBSERVED_SAMPLES_THAN_AMPLITUDES_TO_FIT = auto()
    LBFGSPP_ERROR = auto()
    PDF_INTEGRATION_WINDOW_TOO_SMALL = auto()
    UNUSED_PARAMETER = auto()
    SPECTRUM_WAVELENGTH_TIGHTER_THAN_PARAM = auto()
    MULTI_OBS_ARBITRARY_LSF = auto()

    FORCED_POWERLAW_TO_ZERO = auto()
    NULL_LINES_PROFILE = auto()
    STD_ESTIMATION_FAILED = auto()
    VELOCITY_FIT_RANGE = auto()
    TOO_LITTLE_PIXELS = auto()

class ZLError(Enum):
    """Enumeration of error codes used in redshift and line measurement solvers.

    These codes indicate various fatal issues that can occur during
    redshift estimation or line fitting procedures.
    """
    NO_ERROR = 0
    INTERNAL_ERROR = 1
    EXTERNAL_LIB_ERROR = 2
    INVALID_SPECTRUM_WAVELENGTH = 3
    INVALID_SPECTRUM_FLUX = 4
    INVALID_NOISE = 5
    INVALID_WAVELENGTH_RANGE = 6
    NEGATIVE_CONTINUUMFIT = 7
    BAD_CONTINUUMFIT = 8
    DEPRECATED__NULL_AMPLITUDES = 9
    PDF_PEAK_NOT_FOUND = 10
    DEPRECATED__MAX_AT_BORDER_PDF = 11
    MISSING_PARAMETER = 12
    DEPRECATED__BAD_PARAMETER_VALUE = 13
    UNKNOWN_ATTRIBUTE = 14
    DEPRECATED__BAD_LINECATALOG = 15
    DEPRECATED__BAD_LOGSAMPLEDSPECTRUM = 16
    DEPRECATED__BAD_COUNTMATCH = 17
    BAD_TEMPLATECATALOG = 18
    INVALID_SPECTRUM = 19
    TEMPLATE_OVERLAP_TOO_SMALL = 20
    IE_DZ_NOT_COMPUTABLE = 21
    DEPRECATED__INCOHERENT_INPUTPARAMETERS = 22
    BAD_CALZETTICORR = 23
    IE_CRANGE_VALUE_OUTSIDERANGE = 24
    IE_CRANGE_VECTBORDERS_OUTSIDERANGE = 25
    IE_CRANGE_NO_INTERSECTION = 26
    INVALID_MERIT_VALUES = 27
    DEPRECATED__TPL_NAME_EMPTY = 28
    IE_EMPTY_LIST = 29
    LESS_OBSERVED_SAMPLES_THAN_AMPLITUDES_TO_FIT = 30
    SPECTRUM_CORRECTION_ERROR = 31
    IE_SCOPESTACK_ERROR = 32
    FLAT_ZPDF = 33
    NULL_MODEL = 34
    IE_INVALID_SPECTRUM_INDEX = 35
    IE_UNKNOWN_AIR_VACUUM_CONVERSION = 36
    BAD_LINE_TYPE = 37
    BAD_LINE_FORCE = 38
    IE_FFT_WITH_PHOTOMETRY_NOTIMPLEMENTED = 39
    IE_MULTIOBS_WITH_PHOTOMETRY_NOTIMPLEMENTED = 40
    MISSING_PHOTOMETRIC_DATA = 41
    MISSING_PHOTOMETRIC_TRANSMISSION = 42
    PDF_NORMALIZATION_FAILED = 43
    DEPRECATED__INSUFFICIENT_TEMPLATE_COVERAGE = 44
    INSUFFICIENT_LSF_COVERAGE = 45
    INVALID_LSF = 46
    SPECTRUM_NOT_LOADED = 47
    LSF_NOT_LOADED = 48
    UNALLOWED_DUPLICATES = 49
    IE_UNSORTED_ARRAY = 50
    INVALID_DIRECTORY = 51
    INVALID_FILEPATH = 52
    IE_INVALID_PARAMETER = 53
    MISSING_CONFIG_OPTION = 54
    BAD_FILEFORMAT = 55
    INCOHERENT_CONFIG_OPTIONS = 56
    ATTRIBUTE_NOT_SUPPORTED = 57
    INCOMPATIBLE_PDF_MODELSHAPES = 58
    DEPRECATED__UNKNOWN_RESULT_TYPE = 59
    RELIABILITY_NEEDS_TENSORFLOW = 60
    OUTPUT_READER_ERROR = 61
    PYTHON_API_ERROR = 62
    DEPRECATED__INVALID_NAME = 63
    IE_INVALID_FILTER_INSTRUCTION = 64
    IE_INVALID_FILTER_KEY = 65
    NO_CLASSIFICATION = 66
    INVALID_PARAMETER_FILE = 67
    DUPLICATED_LINES = 68
    STAGE_NOT_RUN_BECAUSE_OF_PREVIOUS_FAILURE = 69
    LINE_RATIO_UNKNOWN_LINE = 70
    IE_UNSUPPORTED_METHOD = 71
    IE_LSF_NOT_LOADED = 72
    LINE_NOT_FOUND = 73
    IMPORT_ERROR = 74
    LINE_CATALOG_ERROR = 75
    LINEMEAS_CATALOG_ERROR = 76
    IE_UNSUPPORTED_OPERATION = 77


class ZObjectCandidates:
    """Container for redshift or line measurement candidates
    for a given object type.

    Parameters
    ----------
    object_type : `str`
        Object type. Must be one of ["galaxy", "star", "qso"].
    errors : `dict`
        Dictionary containing error codes and messages for Z and L solvers.
    warnings : `dict`
        Dictionary containing warning flags for Z and L solvers.
    candidates : `list` [ `astropy.table.row.Row` ]
        List of parameter dictionaries or an astropy row for a single candidate.
    models : `list` [`np.ndarray`]
        Spectrum models for each candidate over the full wavelength range.
    ln_pdf : `np.ndarray`
        PDF marginalized over all models.
    lines : `np.ndarray` or None
        Line measurements (used only for "galaxy" and "qso").

    Attributes
    ----------
    model : `list` [ `np.ndarray ]
        List of model spectra over full wavelength range.
    parameters : `list` [ `dict` [`str` , `object`] ]
        List of parameter dictionaries for each candidate.
    pdf : `np.ndarray`
        PDF marginalized over all models.
    ZWarning : `ZLWarning`
        Warning flags from the redshift solver.
    ZError : `dict` [`str` , `object` ]
        Dictionary with keys "code" of type ZLError and "message" of type str
    for redshift solver error.
    lines : `np.ndarray`
        Line measurement results (if applicable).
    LWarning : `ZLWarning`
        Warning flags from the line measurement solver (None for stars).
    LError : `dict`
        Dictionary with "code" of type ZLError and "message" of type str
    for line solver error (None for stars).
    """

    def __init__(self, object_type, errors, warnings, candidates,
                 models, ln_pdf, lines):
        self.model = models
        if isinstance(candidates, astropy.table.row.Row):
            self.parameters = [dict(candidates)]
        else:
            self.parameters = [dict(c) for c in candidates]
        self.pdf = ln_pdf
        self.ZWarning = ZLWarning(int(warnings[f"{object_type}ZWarning"]))
        self.ZError = {"code": ZLError(errors[f"{object_type}ZError"]),
                       "message": errors[f"{object_type}ZErrorMessage"],
                       }

        if object_type != "star":
            self.lines = lines
            self.LWarning = ZLWarning(int(warnings[f"{object_type}LWarning"]))
            self.LError = {"code": ZLError(errors[f"{object_type}LError"]),
                           "message": errors[f"{object_type}LErrorMessage"],
                           }
        else:
            self.lines = None
            self.LError = None


class ZClassification:
    """Spectroscopic classification result with probabilities and associated flags.

    Parameters
    ----------
    classification : dict
        Dictionary containing classification name and class probabilities.
    errors : dict
        Dictionary with error code for classification step.
    warnings : dict
        Dictionary with warning flags for classification.

    Attributes
    ----------
    name : `str`
        Assigned spectroscopic class: "GALAXY", "QSO", or "STAR".
    probabilities : `dict [ `str` , `float` ]
        Class membership probabilities, with keys "galaxy", "star", and "QSO".
    error : `dict` [ `str` , `ZLError` ]
        Dictionary with a single key "code" for error code
    during classification.
    warning : `ZLWarning`
        Warning flags raised during classification.

    """

    def __init__(self, classification, errors, warnings):
        self.name = classification["class"]
        self.probabilities = dict()
        for o in ["galaxy", "star"]:
            self.probabilities[o] = classification[f'proba{o.capitalize()}']
        self.probabilities["QSO"] = classification['probaQSO']
        self.error = {"code": ZLError(errors["classificationError"])}
        self.warning = ZLWarning(int(warnings["classificationWarning"]))


class PfsZCandidates:
    """Redshift Candidates for a single object


    Attributes
    ----------
    target : `target.Target`
        Target information
    init_error : `dict` [ `str`, `object` ]
        Initialization error with "code" and "message".
    init_warning : `ZLWarning`
        Warning flags from spectrum initialization.
    galaxy : `ZObjectCandidates`
        Candidates and metadata for the galaxy solver.
    qso : `ZObjectCandidates`
        Candidates and metadata for the QSO solver.
    star : `ZObjectCandidates`
        Candidates and metadata for the star solver.
    classification : `ZClassification`
        Result of spectro classification combining all object types.
    """

    def __init__(self, target, errors, warnings, classification,
                 candidates, models, pdfs, lines):
        self.target = target
        self.init_error = {"code": ZLError(errors["initError"]),
                           "message": errors["initErrorMessage"]}
        self.init_warning = ZLWarning(int(warnings["initWarning"]))

        self.galaxy = ZObjectCandidates("galaxy",
                                        errors,
                                        warnings,
                                        candidates["GALAXY"],
                                        models["GALAXY"],
                                        pdfs["GALAXY"],
                                        lines["GALAXY"])
        self.qso = ZObjectCandidates("QSO",
                                     errors,
                                     warnings,
                                     candidates["QSO"],
                                     models["QSO"],
                                     pdfs["QSO"],
                                     lines["QSO"])
        self.star = ZObjectCandidates("star",
                                      errors,
                                      warnings,
                                      candidates["STAR"],
                                      models["STAR"],
                                      pdfs["STAR"],
                                      None)

        self.classification = ZClassification(classification, errors, warnings)

    @classmethod
    def _readImpl(cls, fits):
        """Implementation for reading from FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            Opened FITS file.

        Returns
        -------
        kwargs : ``dict``
            Keyword arguments for constructing spectrum.
        """
        raise NotImplementedError("file format not used")

    @classmethod
    def readFits(cls, filename):
        """Read from FITS file

        This API is intended for use by the LSST data butler, which handles
        translating the desired identity into a filename.

        Parameters
        ----------
        filename : `str`
            Filename of FITS file.

        Returns
        -------
        self : ``cls``
            Constructed instance, from FITS file.
        """
        with astropy.io.fits.open(filename) as fd:
            data = cls._readImpl(fd)
        return cls(**data)

    @classmethod
    def read(cls, identity, dirName="."):
        """Read from file

        This API is intended for use by science users, as it allows selection
        of the correct file from parameters that make sense, such as which
        catId, objId, etc.

        Parameters
        ----------
        identity : `dict`
            Keyword-value pairs identifying the data of interest.
            Common keywords include ``catId``, ``tract``, ``patch``, ``objId``.
        dirName : `str`, optional
            Directory from which to read.

        Returns
        -------
        self : ``cls``
            Spectrum read from file.
        """
        raise NotImplementedError("File formate not used")

    @classmethod
    def get_warning(cls, fits, object_type, stage):
        """"Retrieve warning from fits
        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            Opened FITS file.
        object_type : `str`
            in value in ['galaxy','star','qso']
        stage : `str`
           solving stage

        Returns:
        warning : `ZLWarning`
           warning flag
        """

        if object_type is None:
            return fits[0].header["INIT_WARNING"]
        w_name = f'{object_type.upper()}_{stage_to_pfs_sym[stage]}WARNING'
        warning = fits[0].header[w_name]
        return warning

    def has_solution(self):
        """Has at least one solver succeeded ?

        Returns
        -------
        has_solution : `bool`
        """
        return (self.init_error["code"].value == 0 and
                self.classification.error["code"].value == 0)

    def get_classified_model(self):
        """Return the model of classified object type top candidate

        Returns
        -------
        classified_model : `np.ndarray`
        """

        if self.has_solution():
            class_ = self.classification.name.lower()
            return getattr(self, class_).model[0]
        return None

    def get_classified_parameters(self):
        """Return the model parameters of classified object type top candidate

        Returns
        -------
        classified_model_parameters : `dict` [`str` , `object`]
        """

        if self.has_solution():
            class_ = self.classification.name.lower()
            return getattr(self, class_).parameters[0]
        return None

    def get_classified_ZWarning(self):
        """Return the redshift solver warning of classified object

        Returns
        -------
        zwarning : `ZLWarning`
        """
        if self.has_solution():
            class_ = self.classification.name.lower()
            return getattr(self, class_).ZWarning
        return None

    def get_classified_LWarning(self):
        """Return the line measurement solver warning of classified object

        Returns
        -------
        lwarning : `ZLWarning`
        """
        if self.has_solution():
            class_ = self.classification.name.lower()
            if class_ != "star":
                return getattr(self, class_).LWarning
        return None

    def get_classified_PDF(self):
        """Return the classified PDF marginalized over all models

        Returns
        -------
        ln_pdf : `np.ndarray`
        """
        if self.has_solution():
            class_ = self.classification.name.lower()
            return getattr(self, class_).pdf
        return None

    def get_classified_lines(self):
        """Return the classified lines measurements

        Returns
        -------
        lines : `np.ndarray`
        """

        if self.has_solution():
            class_ = self.classification.name.lower()
            if class_ != "star":
                return getattr(self, class_).lines
        return None

    def get_classified_LError(self):
        """Return the classified line measurement solver error

        Returns
        -------
        l_error : `dict` [`str` , `object`]
        """

        if self.has_solution():
            class_ = self.classification.name.lower()
            if class_ != "star":
                return getattr(self, class_).LError
        return None
