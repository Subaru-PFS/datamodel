import numpy as np
from .utils import inheritDocstrings
import astropy
from enum import Flag, auto
from astropy.table import Table

objects = ["galaxy","qso","star"]



stage_to_pfs_sym = {"redshiftSolver":"Z",
                    "lineMeasSolver":"L"}

class ZLWarning(Flag):
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


@inheritDocstrings
class ZObjectCandidates:
    """Redshift Candidates for a single object and a spectro classification

    Parameters
    ----------
    model : `list[np.array]`
        spectrum models for each candidate on full wavelength range
    parameters : `dict`
        parameter of the models for each candidate
    pdf : `np.array`
        PDF marginalised over all models, should be used with grids from pfsCoZCandidates
    lines : `np.array`
        Lines measurements (only for qso and galaxy)
    ZWarning: Flag
        warning flags for redshift solver
    ZError: `dict` of string
        dictionnary with two keys, code and message, for redshift solver error
    LWarning: `Flag`
        warning flags for redshift solver
    LError: `dict` of strings
        dictionnary with two keys, code and message, both strings values, for line measurement solver error
    """

    def __init__(self, object_type, errors, warnings, candidates, models, ln_pdf, lines):
        self.model = models
        if type(candidates) == astropy.table.row.Row:
            self.parameters = [dict(candidates)]
        else:
            self.parameters = [dict(c) for c in candidates] 
        self.pdf = ln_pdf
        self.ZWarning = ZLWarning(int(warnings[f"{object_type}ZWarning"]))
        self.ZError = {"code":errors[f"{object_type}ZError"],
                       "message":errors[f"{object_type}ZErrorMessage"],
                       }
        
        if object_type != "star":
            self.lines = lines
            self.LWarning = ZLWarning(int(warnings[f"{object_type}LWarning"]))
            self.LError = {"code":errors[f"{object_type}LError"],
                           "message":errors[f"{object_type}LErrorMessage"],
                           }
        else:
            self.lines = None
            self.LError = None
            
class ZClassification:
    """Spectro classification

    Parameters
    ----------
    name : str
    
       Spectro classification : GALAXY, QSO, STAR
    probabilites: dict
       probabilities to be star,galaxy or qso
    error: str
        Error code
    warning:
        Warning flag
    """
    
    def __init__(self, classification, errors, warnings):
        self.name = classification["class"]
        self.probabilities = dict()
        for o in ["galaxy","star"]:
            self.probabilities[o] = classification[f'proba{o.capitalize()}']
        self.probabilities["QSO"] = classification[f'probaQSO']
        self.error = {"code":errors["classificationError"]}
        self.warning = ZLWarning(int(warnings["classificationWarning"]))


@inheritDocstrings
class PfsZCandidates:
    """Redshift Candidates for a single object


    Parameters
    ----------
    galaxy : `pfs.datamodel.ZObjectCandidates`
        galaxy candidates
    qso : `pfs.datamodel.ZObjectCandidates`
        qso candidates
    star : `pfs.datamodel.ZObjectCandidates`
        star candidates
    classification: `pfs.datamodel.ZClassification`
    init_flags: `Flag`
        warning flags for spectrum initialization
    init_errors: `dict`
        dictionnary with two keys, code and message, both strings values, for initialization error
    
    """

 
    def __init__(self, target, errors, warnings, classification, candidates, models, pdfs, lines):
        self.target = target
        self.init_error = {"code": errors["initError"],
                           "message": errors["initErrorMessage"]}
        self.init_warning  =ZLWarning(int(warnings["initWarning"]))

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
            Keyword-value pairs identifying the data of interest. Common keywords
            include ``catId``, ``tract``, ``patch``, ``objId``.
        dirName : `str`, optional
            Directory from which to read.

        Returns
        -------
        self : ``cls``
            Spectrum read from file.
        """
        filename = os.path.join(dirName, cls.filenameFormat % identity)
        return cls.readFits(filename)

    @classmethod
    def get_candidate_attr_from_fits(cls, object_type, attribute, rank):
        return fits[f'{object_type.upper()}_CANDIDATES'].data[rank][attribute]

    @classmethod
    def get_warning(cls, fits, object_type, stage):
        if object_type is None:
            return fits[0].header["INIT_WARNING"]
        return fits[0].header[f'{object_type.upper()}_{stage_to_pfs_sym[stage]}WARNING']

    def has_solution(self):
        return self.init_error["code"] == 0 and self.classification.error["code"] == 0
    
    def get_classified_model(self):
        if self.has_solution():
            class_ = self.classification.name.lower()
            return getattr(self,class_).model[0]
        return None
    
    def get_classified_parameters(self):
        if self.has_solution():
            class_ = self.classification.name.lower()
            return getattr(self,class_).parameters[0]
        return None

    def get_classified_ZWarning(self):
        if self.has_solution():
            class_ = self.classification.name.lower()
            return getattr(self,class_).ZWarning
        return None

    def get_classified_LWarning(self):
        if self.has_solution():
            class_ = self.classification.name.lower()
            if class_ != "star":
                return getattr(self,class_).LWarning
        return None

    def get_classified_PDF(self):
        if self.has_solution():
            class_ = self.classification.name.lower()
            return getattr(self,class_).pdf
        return None    

    def get_classified_lines(self):
        if self.has_solution():
            class_ = self.classification.name.lower()
            if class_ != "star":
                return getattr(self,class_).lines
        return None    

    def get_classified_LError(self):
        if self.has_solution():
            class_ = self.classification.name.lower()
            if class_ != "star":
                return getattr(self,class_).LError
        return None    
