# cython: language_level = 3
# distutils: language = c++

from libcpp.map cimport pair, map
from cython.operator cimport dereference as deref

from analysisg.core.structs cimport meta_t
from analysisg.core.meta cimport meta
from analysisg.core.tools cimport *

try:
    import pyAMI
    from pyAMI.client import Client
    from pyAMI.api import list_datasets as atlas

except ModuleNotFoundError: pass
except NameError: pass
import warnings
import signal

def _sig(signum, frame): return ""
signal.signal(signal.SIGALRM, _sig)
signal.alarm(30)
warnings.filterwarnings("ignore")

cdef class ami_client:
    cdef file
    cdef client
    cdef public bool is_cached

    def __init__(self): pass
    def __cinit__(self):
        warnings.filterwarnings("ignore")
        try: self.client = Client("atlas")
        except: self.client = None
        self.file = None

    cdef list list_datasets(self, int dsids, str type_ = "DAOD_TOPQ1"):
        cdef str i
        cdef int idx
        cdef dict command = {}
        command["client"] = self.client
        command["type"] = type_
        command["dataset_number"] = None
        command["dataset_number"] = [str(dsids)]
        try: print(atlas(**command))
        except TypeError: pass

cdef class Meta:
    def __cinit__(self):
        self.ptr = new meta()

    def __init__(self): pass
    def __dealloc__(self): del self.ptr
    cdef __meta__(self, meta* _meta):
        cdef ami_client ami = ami_client()
        self.ptr.meta_data = _meta.meta_data
        ami.list_datasets(_meta.meta_data.dsid)

    # Attributes with getter and setter
    @property
    def dsid(self) -> int:
        return self.ptr.meta_data.dsid

    @dsid.setter
    def dsid(self, int val):
        self.ptr.meta_data.dsid = val

    @property
    def amitag(self) -> str:
        return env(self.ptr.meta_data.AMITag)

    @amitag.setter
    def amitag(self, str val):
        self.ptr.meta_data.AMITag = enc(val)

    @property
    def generators(self) -> str:
        return env(self.ptr.meta_data.generators)

    @generators.setter
    def generators(self, str val):
        val = val.replace("+", " ")
        self.ptr.meta_data.generators = enc(val)

    @property
    def isMC(self) -> bool:
        return self.ptr.meta_data.isMC

    @isMC.setter
    def isMC(self, bool val):
        self.ptr.meta_data.isMC = val

    @property
    def derivationFormat(self) -> str:
        return env(self.ptr.meta_data.derivationFormat)

    @derivationFormat.setter
    def derivationFormat(self, str val):
        self.ptr.meta_data.derivationFormat = enc(val)

    @property
    def eventNumber(self) -> double:
        return self.ptr.meta_data.eventNumber

    @eventNumber.setter
    def eventNumber(self, double val):
        self.ptr.meta_data.eventNumber = val

    @property
    def ecmEnergy(self) -> float:
        return self.ptr.meta_data.ecmEnergy

    @ecmEnergy.setter
    def ecmEnergy(self, val: Union[str, float]):
        self.ptr.meta_data.ecmEnergy = float(val)

    @property
    def genFiltEff(self) -> float:
        return self.ptr.meta_data.genFiltEff

    @genFiltEff.setter
    def genFiltEff(self, val: Union[str, float]):
        self.ptr.meta_data.genFiltEff = float(val)

    @property
    def completion(self) -> float:
        return self.ptr.meta_data.completion

    @completion.setter
    def completion(self, val: Union[str, float]):
        self.ptr.meta_data.completion = float(val)

    @property
    def beam_energy(self) -> float:
        return self.ptr.meta_data.beam_energy

    @beam_energy.setter
    def beam_energy(self, val: Union[str, float]):
        self.ptr.meta_data.beam_energy = float(val)

    @property
    def crossSection(self) -> float:
        return self.ptr.meta_data.crossSection

    @crossSection.setter
    def crossSection(self, val: Union[str, float]):
        try: self.ptr.meta_data.crossSection = float(val)
        except ValueError: self.ptr.meta_data.crossSection = -1

    @property
    def crossSection_mean(self) -> float:
        return self.ptr.meta_data.crossSection_mean

    @crossSection_mean.setter
    def crossSection_mean(self, val):
        try: self.ptr.meta_data.crossSection_mean = float(val)
        except ValueError: self.ptr.meta_data.crossSection_mean = -1

    @property
    def totalSize(self) -> float:
        return self.ptr.meta_data.totalSize

    @totalSize.setter
    def totalSize(self, val: Union[str, float]):
        self.ptr.meta_data.totalSize = float(val)

    @property
    def nFiles(self) -> int:
        return self.ptr.meta_data.nFiles

    @nFiles.setter
    def nFiles(self, val: Union[str, int]):
        self.ptr.meta_data.nFiles = int(val)

    @property
    def run_number(self) -> int:
        return self.ptr.meta_data.run_number

    @run_number.setter
    def run_number(self, val: Union[str, int]):
        self.ptr.meta_data.run_number = int(val)

    @property
    def totalEvents(self) -> int:
        return self.ptr.meta_data.totalEvents

    @totalEvents.setter
    def totalEvents(self, val: Union[str, int]):
        self.ptr.meta_data.totalEvents = int(val)

    @property
    def datasetNumber(self) -> int:
        return self.ptr.meta_data.datasetNumber

    @datasetNumber.setter
    def datasetNumber(self, val: Union[str, int]):
        self.ptr.meta_data.datasetNumber = int(val)

    @property
    def identifier(self) -> str:
        return env(self.ptr.meta_data.identifier)

    @identifier.setter
    def identifier(self, str val) -> str:
        self.ptr.meta_data.identifier = enc(val)

    @property
    def prodsysStatus(self) -> str:
        return env(self.ptr.meta_data.prodsysStatus)

    @prodsysStatus.setter
    def prodsysStatus(self, str val) -> str:
        self.ptr.meta_data.prodsysStatus = enc(val)

    @property
    def dataType(self) -> str:
        return env(self.ptr.meta_data.dataType)

    @dataType.setter
    def dataType(self, str val) -> str:
        self.ptr.meta_data.dataType = enc(val)

    @property
    def version(self) -> str:
        return env(self.ptr.meta_data.version)

    @version.setter
    def version(self, str val):
        self.ptr.meta_data.version = enc(val)

    @property
    def PDF(self) -> str:
        return env(self.ptr.meta_data.PDF)

    @PDF.setter
    def PDF(self, str val):
        self.ptr.meta_data.PDF = enc(val)

    @property
    def AtlasRelease(self) -> str:
        return env(self.ptr.meta_data.AtlasRelease)

    @AtlasRelease.setter
    def AtlasRelease(self, str val):
        self.ptr.meta_data.AtlasRelease = enc(val)

    @property
    def principalPhysicsGroup(self) -> str:
        return env(self.ptr.meta_data.principalPhysicsGroup)

    @principalPhysicsGroup.setter
    def principalPhysicsGroup(self, str val):
        self.ptr.meta_data.principalPhysicsGroup = enc(val)

    @property
    def physicsShort(self) -> str:
        return env(self.ptr.meta_data.physicsShort)

    @physicsShort.setter
    def physicsShort(self, str val):
        self.ptr.meta_data.physicsShort = enc(val)

    @property
    def generatorName(self) -> str:
        return env(self.ptr.meta_data.generatorName)

    @generatorName.setter
    def generatorName(self, str val):
        self.ptr.meta_data.generatorName = enc(val)

    @property
    def geometryVersion(self) -> str:
        return env(self.ptr.meta_data.geometryVersion)

    @geometryVersion.setter
    def geometryVersion(self, str val):
        self.ptr.meta_data.geometryVersion = enc(val)

    @property
    def conditionsTag(self) -> str:
        return env(self.ptr.meta_data.conditionsTag)

    @conditionsTag.setter
    def conditionsTag(self, str val) -> str:
        self.ptr.meta_data.conditionsTag = enc(val)

    @property
    def generatorTune(self) -> str:
        return env(self.ptr.meta_data.generatorTune)

    @generatorTune.setter
    def generatorTune(self, str val):
        self.ptr.meta_data.generatorTune = enc(val)

    @property
    def amiStatus(self) -> str:
        return env(self.ptr.meta_data.amiStatus)

    @amiStatus.setter
    def amiStatus(self, str val):
        self.ptr.meta_data.amiStatus = enc(val)

    @property
    def beamType(self) -> str:
        return env(self.ptr.meta_data.beamType)

    @beamType.setter
    def beamType(self, str val):
        self.ptr.meta_data.beamType = enc(val)

    @property
    def productionStep(self) -> str:
        return env(self.ptr.meta_data.productionStep)

    @productionStep.setter
    def productionStep(self, str val):
        self.ptr.meta_data.productionStep = enc(val)

    @property
    def projectName(self) -> str:
        return env(self.ptr.meta_data.projectName)

    @projectName.setter
    def projectName(self, str val):
        self.ptr.meta_data.projectName = enc(val)

    @property
    def statsAlgorithm(self) -> str:
        return env(self.ptr.meta_data.statsAlgorithm)

    @statsAlgorithm.setter
    def statsAlgorithm(self, str val):
        self.ptr.meta_data.statsAlgorithm = enc(val)

    @property
    def genFilterNames(self) -> str:
        return env(self.ptr.meta_data.genFilterNames)

    @genFilterNames.setter
    def genFilterNames(self, str val) -> str:
        self.ptr.meta_data.genFilterNames = enc(val)

    @property
    def file_type(self) -> str:
        return env(self.ptr.meta_data.file_type)

    @file_type.setter
    def file_type(self, str val):
        self.ptr.meta_data.file_type = enc(val)

    #@property
    #def DatasetName(self) -> str:
    #    return env(self.ptr.DatasetName())

    #@DatasetName.setter
    #def DatasetName(self, str val):
    #    self.ptr.meta_data.DatasetName = enc(val)

    @property
    def logicalDatasetName(self) -> str:
        return env(self.ptr.meta_data.logicalDatasetName)

    @logicalDatasetName.setter
    def logicalDatasetName(self, str val):
        self.ptr.meta_data.logicalDatasetName = enc(val)

    @property
    def event_index(self) -> int:
        return self.ptr.meta_data.event_index

    @event_index.setter
    def event_index(self, int val):
        self.ptr.meta_data.event_index = val


    # constant properties
    @property
    def original_name(self) -> str:
        return env(self.ptr.meta_data.original_name)

    @property
    def original_path(self) -> str:
        return env(self.ptr.meta_data.original_path)

    #@property
    #def hash(self) -> str:
    #    self.ptr.Hash()
    #    return env(self.ptr.hash)

    @property
    def keywords(self) -> list:
        return [env(i) for i in self.ptr.meta_data.keywords]

    @property
    def weights(self) -> list:
        return [env(i) for i in self.ptr.meta_data.weights]

    @property
    def keyword(self) -> list:
        return [env(i) for i in self.ptr.meta_data.keyword]

    @property
    def found(self) -> bool:
        return self.ptr.meta_data.found

    @property
    def config(self) -> dict:
        cdef pair[string, string] it
        cdef dict out = {}
        for it in self.ptr.meta_data.config: out[env(it.first)] = env(it.second)
        return out

    #@property
    #def GetLengthTrees(self) -> dict:
    #    cdef map[string, int] x = self.ptr.GetLength()
    #    cdef pair[string, int] it
    #    cdef dict out = {env(it.first) : it.second for it in x}
    #    return out

    #@property
    #def MissingTrees(self) -> list:
    #    return [env(i) for i in self.ptr.meta_data.mis_trees]

    #@property
    #def MissingBranches(self) -> list:
    #    return [env(i) for i in self.ptr.meta_data.mis_branches]

    #@property
    #def MissingLeaves(self) -> list:
    #    return [env(i) for i in self.ptr.meta_data.mis_leaves]

    #@property
    #def DAODList(self) -> list:
    #    cdef vector[string] out = self.ptr.DAODList()
    #    cdef string i
    #    return [env(i) for i in out]

    @property
    def Files(self) -> dict:
        cdef pair[int, string] it
        cdef dict out = {}
        for it in self.ptr.meta_data.inputfiles:
            out[it.first] = env(it.second)
        return out

    #@property
    #def DAOD(self) -> str:
    #    return self.IndexToSample(self.ptr.event_index)

    @property
    def fileGUID(self) -> dict:
        cdef pair[string, int] it
        cdef dict output = {}
        cdef string guid
        for it in self.ptr.meta_data.LFN:
            guid = self.ptr.meta_data.fileGUID.at(it.second)
            output[env(it.first)] = env(guid)
        return output

    @property
    def events(self) -> dict:
        cdef pair[string, int] it
        cdef dict output = {}
        for it in self.ptr.meta_data.LFN:
            output[env(it.first)] = self.ptr.meta_data.events.at(it.second)
        return output

    @property
    def fileSize(self) -> dict:
        cdef pair[string, int] it
        cdef dict output = {}
        for it in self.ptr.meta_data.LFN:
            output[env(it.first)] = self.ptr.meta_data.filesSize.at(it.second)
        return output

    @property
    def sample_name(self):
        return env(self.ptr.meta_data.sample_name)

    @sample_name.setter
    def sample_name(self, str val):
        self.ptr.meta_data.sample_name = enc(val)










