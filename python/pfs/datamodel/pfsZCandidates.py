import numpy as np
from .utils import inheritDocstrings
import astropy
from enum import Flag, auto

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
        PDF marginalised over all models
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

    def __init__(self, object_type, hdul):
        self.model = list()
        self.parameters = list()
        self.pdf = None

        param_names = hdul[f"{object_type}_CANDIDATES"].data.dtype.names
        for cand in hdul[f"{object_type}_CANDIDATES"].data:
            self.model.append(cand["MODELFLUX"])
            params = dict()
            for p in param_names:
                if p in  ["MODELFLUX","CRANK"]:
                    continue
                params[p]=cand[p]
            self.parameters.append(params)

        self.pdf = np.array(hdul[f"{object_type}_PDF"].data)
        if object_type != "star":
            self.lines = np.array(hdul[f"{object_type}_LINES"].data)
        else:
            self.lines = None

            
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
    
    def __init__(self,class_, probas, error, warning):
        self.name = class_
        self.probabilities = probas
        self.error = error
        self.warning = ZLWarning(warning)


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

    filenameFormat = ("pfsZCandidates-%(catId)05d-%(tract)05d-%(patch)s-%(objId)016x"
                      "-%(nVisit)03d-0x%(pfsVisitHash)016x.fits")
    filenameRegex = r"^pfsZCandidates-(\d{5})-(\d{5})-(.*)-([0-9a-f]{16})-(\d{3})-0x([0-9a-f]{16})\.fits.*$"
    filenameKeys = [("catId", int), ("tract", int), ("patch", str), ("objId", int),
                    ("nVisit", int), ("pfsVisitHash", int)]

    def __init__(self,errors, init_warning, galaxy, qso, star, classification):
        self.init_error = errors["init"]
        self.init_warning = init_warning
        self.galaxy = galaxy
        self.qso = qso
        self.star = star
        self.classification = classification

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
        data = {}
        data["errors"] = dict()
        data["errors"]["init"]=dict()
        data["errors"]["init"]["code"]=fits[0].header["INIT_ERROR"]
        data["errors"]["init"]["message"]=fits[0].header["INIT_ERR"]
        
#        self.init_error=data["errors"]["init"]
        data["init_warning"]=ZLWarning(cls.get_warning(fits,None,"init"))
        for o in objects:
            od = ZObjectCandidates(o,fits)
            stage = "redshiftSolver"
            data["errors"][f"{o}_{stage}"]=dict()
            data["errors"][f"{o}_{stage}"]["code"]=fits[0].header[f"{o.upper()}_ZERROR"]
            data["errors"][f"{o}_{stage}"]["message"]=fits[0].header[f"{o.upper()[0]}_ZERR"]
            setattr(od,"ZError",data["errors"][f"{o}_{stage}"])
            setattr(od,"ZWarning",ZLWarning(cls.get_warning(fits,o,stage)))
            if o != "star":
                stage = "lineMeasSolver"
                data["errors"][f"{o}_{stage}"] = dict()
                data["errors"][f"{o}_{stage}"]["code"]=fits[0].header[f"{o.upper()}_LERROR"]
                data["errors"][f"{o}_{stage}"]["message"]=fits[0].header[f"{o.upper()[0]}_LERR"]
                setattr(od,"LError",data["errors"][f"{o}_{stage}"])
                setattr(od,"LWarning",ZLWarning(cls.get_warning(fits,o,stage)))            
            data[o]=od
        

        probabilities = dict()
        for o in objects:
            probabilities[f"{o}_proba"]=fits["CLASSIFICATION"].header[f"P_{o.upper()}"]
        data["classification"]=ZClassification(fits["CLASSIFICATION"].header["CLASS"],
                                               probabilities,
                                               fits[0].header["CLASSIFICATION_ERROR"],
                                               fits[0].header["CLASSIFICATION_WARNING"])
        return data
        
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

