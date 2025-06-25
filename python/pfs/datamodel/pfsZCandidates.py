import astropy
from enum import Flag, auto

objects = ["galaxy", "qso", "star"]


stage_to_pfs_sym = {"redshiftSolver": "Z",
                    "lineMeasSolver": "L"}


class ZLWarning(Flag):
    """
    Enumeration of warning flags used in redshift and line measurement solvers.

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


class ZObjectCandidates:
    """
    Container for redshift or line measurement candidates for a given object type.

    Parameters
    ----------
    object_type : str
        Object type. Must be one of ["galaxy", "star", "qso"].
    errors : dict
        Dictionary containing error codes and messages for Z and L solvers.
    warnings : dict
        Dictionary containing warning flags for Z and L solvers.
    candidates : list or astropy.table.row.Row
        List of parameter dictionaries or an astropy row for a single candidate.
    models : list of np.ndarray
        Spectrum models for each candidate over the full wavelength range.
    ln_pdf : np.ndarray
        PDF marginalized over all models.
    lines : np.ndarray or None
        Line measurements (used only for "galaxy" and "qso").

    Attributes
    ----------
    model : list of np.ndarray
        List of model spectra over full wavelength range.
    parameters : list of dict
        List of parameter dictionaries for each candidate.
    pdf : np.ndarray
        PDF marginalized over all models.
    ZWarning : ZLWarning
        Warning flags from the redshift solver.
    ZError : dict
        Dictionary with keys "code" and "message" for redshift solver error.
    lines : np.ndarray or None
        Line measurement results (if applicable).
    LWarning : ZLWarning or None
        Warning flags from the line measurement solver (None for stars).
    LError : dict or None
        Dictionary with "code" and "message" for line solver error (None for stars).
    """
    
    def __init__(self, object_type, errors, warnings, candidates, models, ln_pdf, lines):
        self.model = models
        if isinstance(candidates, astropy.table.row.Row):
            self.parameters = [dict(candidates)]
        else:
            self.parameters = [dict(c) for c in candidates]
        self.pdf = ln_pdf
        self.ZWarning = ZLWarning(int(warnings[f"{object_type}ZWarning"]))
        self.ZError = {"code": errors[f"{object_type}ZError"],
                       "message": errors[f"{object_type}ZErrorMessage"],
                       }

        if object_type != "star":
            self.lines = lines
            self.LWarning = ZLWarning(int(warnings[f"{object_type}LWarning"]))
            self.LError = {"code": errors[f"{object_type}LError"],
                           "message": errors[f"{object_type}LErrorMessage"],
                           }
        else:
            self.lines = None
            self.LError = None


class ZClassification:
    """
    Spectroscopic classification result with probabilities and associated flags.

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
    name : str
        Assigned spectroscopic class: "GALAXY", "QSO", or "STAR".
    probabilities : dict
        Class membership probabilities, with keys "galaxy", "star", and "QSO".
    error : dict
        Dictionary with a single key "code" for error code during classification.
    warning : ZLWarning
        Warning flags raised during classification.
    
    """


    def __init__(self, classification, errors, warnings):
        self.name = classification["class"]
        self.probabilities = dict()
        for o in ["galaxy", "star"]:
            self.probabilities[o] = classification[f'proba{o.capitalize()}']
        self.probabilities["QSO"] = classification['probaQSO']
        self.error = {"code": errors["classificationError"]}
        self.warning = ZLWarning(int(warnings["classificationWarning"]))


class PfsZCandidates:
    """Redshift Candidates for a single object


    Parameters
    ----------
    target : object
        Target information
    errors : dict
        Dictionary containing error codes and messages for various stages.
    warnings : dict
        Dictionary containing warning flags for various stages.
    classification : dict
        Classification result with probabilities and class name.
    candidates : dict
        Dictionary of candidate lists for each object type (GALAXY, QSO, STAR).
    models : dict
        Dictionary of models for each object type.
    pdfs : dict
        Dictionary of PDFs for each object type.
    lines : dict
        Dictionary of line measurements for each object type.

    Attributes
    ----------
    target : object
        Target information
    init_error : dict
        Initialization error with "code" and "message".
    init_warning : ZLWarning
        Warning flags from spectrum initialization.
    galaxy : ZObjectCandidates
        Candidates and metadata for the galaxy solver.
    qso : ZObjectCandidates
        Candidates and metadata for the QSO solver.
    star : ZObjectCandidates
        Candidates and metadata for the star solver.
    classification : ZClassification
        Result of spectro classification combining all object types.
    """

    def __init__(self, target, errors, warnings, classification,
                 candidates, models, pdfs, lines):
        self.target = target
        self.init_error = {"code": errors["initError"],
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
        if object_type is None:
            return fits[0].header["INIT_WARNING"]
        w_name = f'{object_type.upper()}_{stage_to_pfs_sym[stage]}WARNING'
        return fits[0].header[w_name]

    def has_solution(self):
        return (self.init_error["code"] == 0 and
                self.classification.error["code"] == 0)

    def get_classified_model(self):
        if self.has_solution():
            class_ = self.classification.name.lower()
            return getattr(self, class_).model[0]
        return None

    def get_classified_parameters(self):
        if self.has_solution():
            class_ = self.classification.name.lower()
            return getattr(self, class_).parameters[0]
        return None

    def get_classified_ZWarning(self):
        if self.has_solution():
            class_ = self.classification.name.lower()
            return getattr(self, class_).ZWarning
        return None

    def get_classified_LWarning(self):
        if self.has_solution():
            class_ = self.classification.name.lower()
            if class_ != "star":
                return getattr(self, class_).LWarning
        return None

    def get_classified_PDF(self):
        if self.has_solution():
            class_ = self.classification.name.lower()
            return getattr(self, class_).pdf
        return None

    def get_classified_lines(self):
        if self.has_solution():
            class_ = self.classification.name.lower()
            if class_ != "star":
                return getattr(self, class_).lines
        return None

    def get_classified_LError(self):
        if self.has_solution():
            class_ = self.classification.name.lower()
            if class_ != "star":
                return getattr(self, class_).LError
        return None
