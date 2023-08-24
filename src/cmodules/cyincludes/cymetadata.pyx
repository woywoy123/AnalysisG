# distuils: language = c++
# cython: language_level = 3

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map, pair
from libcpp cimport bool

from cymetadata cimport CyMetaData
from cytypes cimport meta_t

try:
    import pyAMI
    import pyAMI.client
    import pyAMI.atlas.api as atlas
except ModuleNotFoundError: pass

from uproot.exceptions import KeyInFileError
import warnings
import uproot
import json
import signal

cdef string enc(str val): return val.encode("UTF-8")
cdef str env(string val): return val.decode("UTF-8")

cdef class MetaData:

    cdef CyMetaData* ptr
    cdef public bool loaded
    cdef public client
    cdef public str sampletype
    cdef public bool scan_ami
    cdef meta_t meta

    def __cinit__(self):
        self.ptr = new CyMetaData()
        self.client = None

    def __init__(self, str file, bool scan_ami = True, str sampletype = "DAOD_TOPQ1"):
        self.original_input = file
        self.sampletype = sampletype
        self.loaded = True
        self.scan_ami = scan_ami

        if not scan_ami: self._getMetaData(file)
        else:
            warnings.filterwarnings("ignore")
            try: self.client = pyAMI.client.Client("atlas")
            except: pass
            self.loaded = self._getMetaData(file)
        self.ptr.Import(self.meta)

    def __dealloc__(self): del self.ptr
    def __getstate__(self) -> meta_t: return self.ptr.Export()
    def __setstate__(self, meta_t val): self.ptr.Import(val)
    def __hash__(self): return int(self.hash[:8], 0)
    def __eq__(self, other):
        if not issubclass(other, MetaData): return False
        else: return self.hash == other.hash

    def __str__(self) -> str:
        cdef str i = ""
        cdef str k
        for k in self.__dir__():
            if k.startswith("_"): continue
            i += k + " : " + str(getattr(self, k)) + "\n"
        return i

    def _getMetaData(self, str file) -> bool:
        if not file.endswith(".root"): return False
        if not self._file_data(file): return False
        if not self._file_tracker(file): return False
        if not self._file_truth(file): return False
        if not self.scan_ami: return True
        if not self._search(): return False
        return True

    def _get(self, dict command) -> dict:
        cdef dict x, out
        cdef str l

        out = {}
        x = [k for k in uproot.iterate(**command)][0]
        for l in command["expressions"]:
            try: out[l] = x[l].tolist()[0]
            except KeyError: pass
        return out

    def _file_data(self, str i) -> bool:
        cdef dict command = {
                "files" : i + ":sumWeights",
                "expressions" :  ["dsid", "AMITag", "generators"],
                "how" : dict, "library" : "np"
        }

        cdef bool k1 = True
        cdef bool k2 = True
        cdef bool k3 = True

        cdef dict out = self._get(command)
        try: self.dsid = out["dsid"]
        except KeyError: k1 = False
        self.hash

        try: self.amitag = out["AMITag"]
        except KeyError: k2 = False

        try: self.generators = out["generators"]
        except KeyError: k3 = False
        if not k1 and not k2 and not k3: return False
        return True

    def _file_tracker(self, str i) -> bool:
        cdef dict command = {
                "files" : i + ":AnalysisTracking",
                "expressions" : ["jsonData"],
                "step_size" : 100, "how" : dict,
                "library" : "np"
        }

        cdef list f
        cdef dict this
        cdef int index = 0

        try: f = self._get(command)["jsonData"].split("\n")
        except KeyError: return False
        if len(f) == 0: return False

        t = json.loads("\n".join([k for k in f if "BtagCDIPath" not in k]))
        this = t["inputConfig"]
        self.isMC = this["isMC"]
        self.derivationFormat = this["derivationFormat"]

        this = t["configSettings"]
        for x in this: self.ptr.addconfig(enc(x), enc(this[x]))

        for x in t["inputFiles"]:
            self.ptr.addsamples(index, x[1], enc(x[0]))
            index += x[1]
        return True

    def _file_truth(self, str i) -> bool:
        cdef dict command = {
                "files" : i + ":truth",
                "expressions" : ["eventNumber"],
                "how" : dict, "library" : "np"
        }
        try: self.eventNumber = self._get(command)["eventNumber"]
        except KeyError: return False
        return True

    def _populate(self, inpt):
        for key in inpt:
            if key == "run_number": continue
            if key == "keywords": continue
            if key == "keyword": continue
            if key == "weights": continue
            setattr(self, key, inpt[key])

        self.run_number = inpt["run_number"][1:-1]
        self.meta.keywords = [enc(v) for v in inpt["keywords"].replace(" ", "").split(",")]
        self.meta.keyword  = [enc(v) for v in inpt["keyword"].replace(" ", "").split(",")]
        self.meta.weights  = [enc(v.lstrip(" ").rstrip(" ")) for v in inpt["weights"].split("|")][:-1]

    def _search(self) -> bool:
        def _sig(signum, frame): return ""

        if self.client is None: return False
        signal.signal(signal.SIGALRM, _sig)
        signal.alarm(10)
        warnings.filterwarnings("ignore")

        cdef list x
        try:
            x = atlas.list_datasets(
                self.client, dataset_number = [str(self.dsid)], type = self.sampletype
            )
        except pyAMI.exception.Error: return False

        cdef dict t
        cdef list tags
        cdef str sample, v
        cdef int index

        tags = list(set(self.amitag.split("_")))
        for i in x:
            try: sample = i["ldn"]
            except KeyError: continue

            if len([v for v in tags if v in sample]) != len(tags): continue

            self.DatasetName = sample
            self.meta.found = True
            self._populate(atlas.get_dataset_info(self.client, sample)[0])

            index = 0
            for k in atlas.list_files(self.client, sample):
                self.meta.LFN[enc(k["LFN"])] = index
                self.meta.fileGUID.push_back(enc(k["fileGUID"]))
                self.meta.events.push_back(int(k["events"]))
                self.meta.fileSize.push_back(float(k["fileSize"]))
                index += 1
        return self.meta.found

    def IndexToSample(self, int index) -> str:
        return env(self.ptr.IndexToSample(index))

    def ProcessKeys(self, dict val):
        cdef vector[string] keys = [enc(i) for i in val["keys"]]
        cdef int num = val["num"][0]
        self.ptr.processkeys(keys, num)

    def _findmissingkeys(self): 
        self.ptr.FindMissingKeys()

    def _MakeGetter(self) -> dict:
        cdef map[string, vector[string]] get = self.ptr.MakeGetter()
        cdef pair[string, vector[string]] it
        cdef string i
        cdef dict output = {}
        for it in get:
            output[env(it.first)] = [env(i) for i in it.second]
        return output

    # Attributes with getter and setter
    @property
    def Trees(self) -> list:
        return [env(i) for i in self.meta.req_trees]

    @Trees.setter
    def Trees(self, list val):
        self.meta.req_trees = [enc(i) for i in val]

    @property
    def Branches(self) -> list:
        return [env(i) for i in self.meta.req_branches]

    @Branches.setter
    def Branches(self, list val):
        self.meta.req_branches = [enc(i) for i in val]

    @property
    def Leaves(self) -> list:
        return [env(i) for i in self.meta.req_leaves]

    @Leaves.setter
    def Leaves(self, list val):
        self.meta.req_leaves = [enc(i) for i in val]

    @property
    def original_input(self) -> str:
        return env(self.meta.original_input)

    @original_input.setter
    def original_input(self, str val):
        cdef list oth = val.split("/")
        self.meta.original_input = enc(val)
        self.meta.original_name = enc(oth[-1])
        self.meta.original_path = enc("/".join(oth[:-1]))

    @property
    def dsid(self) -> int:
        return self.meta.dsid

    @dsid.setter
    def dsid(self, int val):
        self.meta.dsid = val

    @property
    def amitag(self) -> str:
        return env(self.meta.AMITag)

    @amitag.setter
    def amitag(self, str val):
        self.meta.AMITag = enc(val)

    @property
    def generators(self) -> str:
        return env(self.meta.generators)

    @generators.setter
    def generators(self, str val):
        val = val.replace("+", " ")
        self.meta.generators = enc(val)

    @property
    def isMC(self) -> bool:
        return self.meta.isMC

    @isMC.setter
    def isMC(self, bool val):
        self.meta.isMC = val

    @property
    def derivationFormat(self) -> str:
        return env(self.meta.derivationFormat)

    @derivationFormat.setter
    def derivationFormat(self, str val):
        self.meta.derivationFormat = enc(val)

    @property
    def eventNumber(self) -> int:
        return self.meta.eventNumber

    @eventNumber.setter
    def eventNumber(self, int val):
        self.meta.eventNumber = val

    @property
    def ecmEnergy(self) -> float:
        return self.meta.ecmEnergy

    @ecmEnergy.setter
    def ecmEnergy(self, val: Union[str, float]):
        self.meta.ecmEnergy = float(val)

    @property
    def genFiltEff(self) -> float:
        return self.meta.genFiltEff

    @genFiltEff.setter
    def genFiltEff(self, val: Union[str, float]):
        self.meta.genFiltEff = float(val)

    @property
    def completion(self) -> float:
        return self.meta.completion

    @completion.setter
    def completion(self, val: Union[str, float]):
        self.meta.completion = float(val)

    @property
    def beam_energy(self) -> float:
        return self.meta.beam_energy

    @beam_energy.setter
    def beam_energy(self, val: Union[str, float]):
        self.meta.beam_energy = float(val)

    @property
    def crossSection(self) -> float:
        return self.meta.crossSection

    @crossSection.setter
    def crossSection(self, val: Union[str, float]):
        self.meta.crossSection = float(val)

    @property
    def crossSection_mean(self) -> float:
        return self.meta.crossSection_mean

    @crossSection_mean.setter
    def crossSection_mean(self, val: Union[str, float]):
        self.meta.crossSection_mean = float(val)

    @property
    def totalSize(self) -> float:
        return self.meta.totalSize

    @totalSize.setter
    def totalSize(self, val: Union[str, float]):
        self.meta.totalSize = float(val)

    @property
    def nFiles(self) -> int:
        return self.meta.nFiles

    @nFiles.setter
    def nFiles(self, val: Union[str, int]):
        self.meta.nFiles = int(val)

    @property
    def run_number(self) -> int:
        return self.meta.run_number

    @run_number.setter
    def run_number(self, val: Union[str, int]):
        self.meta.run_numer = int(val)

    @property
    def totalEvents(self) -> int:
        return self.meta.totalEvents

    @totalEvents.setter
    def totalEvents(self, val: Union[str, int]):
        self.meta.totalEvents = int(val)

    @property
    def datasetNumber(self) -> int:
        return self.meta.datasetNumber

    @datasetNumber.setter
    def datasetNumber(self, val: Union[str, int]):
        self.meta.datasetNumber = int(val)

    @property
    def identifier(self) -> str:
        return env(self.meta.identifier)

    @identifier.setter
    def identifier(self, str val) -> str:
        self.meta.identifier = enc(val)

    @property
    def prodsysStatus(self) -> str:
        return env(self.meta.prodsysStatus)

    @prodsysStatus.setter
    def prodsysStatus(self, str val) -> str:
        self.meta.prodsysStatus = enc(val)

    @property
    def dataType(self) -> str:
        return env(self.meta.dataType)

    @dataType.setter
    def dataType(self, str val) -> str:
        self.meta.dataType = enc(val)

    @property
    def version(self) -> str:
        return env(self.meta.version)

    @version.setter
    def version(self, str val):
        self.meta.version = enc(val)

    @property
    def PDF(self) -> str:
        return env(self.meta.PDF)

    @PDF.setter
    def PDF(self, str val):
        self.meta.PDF = enc(val)

    @property
    def AtlasRelease(self) -> str:
        return env(self.meta.AtlasRelease)

    @AtlasRelease.setter
    def AtlasRelease(self, str val):
        self.meta.AtlasRelease = enc(val)

    @property
    def principalPhysicsGroup(self) -> str:
        return env(self.meta.principalPhysicsGroup)

    @principalPhysicsGroup.setter
    def principalPhysicsGroup(self, str val):
        self.meta.principalPhysicsGroup = enc(val)

    @property
    def physicsShort(self) -> str:
        return env(self.meta.physicsShort)

    @physicsShort.setter
    def physicsShort(self, str val):
        self.meta.physicsShort = enc(val)

    @property
    def generatorName(self) -> str:
        return env(self.meta.generatorName)

    @generatorName.setter
    def generatorName(self, str val):
        self.meta.generatorName = enc(val)

    @property
    def geometryVersion(self) -> str:
        return env(self.meta.geometryVersion)

    @geometryVersion.setter
    def geometryVersion(self, str val):
        self.meta.geometryVersion = enc(val)

    @property
    def conditionsTag(self) -> str:
        return env(self.meta.conditionsTag)

    @conditionsTag.setter
    def conditionsTag(self, str val) -> str:
        self.meta.conditionsTag = enc(val)

    @property
    def generatorTune(self) -> str:
        return env(self.meta.generatorTune)

    @generatorTune.setter
    def generatorTune(self, str val):
        self.meta.generatorTune = enc(val)

    @property
    def amiStatus(self) -> str:
        return env(self.meta.amiStatus)

    @amiStatus.setter
    def amiStatus(self, str val):
        self.meta.amiStatus = enc(val)

    @property
    def beamType(self) -> str:
        return env(self.meta.beamType)

    @beamType.setter
    def beamType(self, str val):
        self.meta.beamType = enc(val)

    @property
    def productionStep(self) -> str:
        return env(self.meta.productionStep)

    @productionStep.setter
    def productionStep(self, str val):
        self.meta.productionStep = enc(val)

    @property
    def projectName(self) -> str:
        return env(self.meta.projectName)

    @projectName.setter
    def projectName(self, str val):
        self.meta.projectName = enc(val)

    @property
    def statsAlgorithm(self) -> str:
        return env(self.meta.statsAlgorithm)

    @statsAlgorithm.setter
    def statsAlgorithm(self, str val):
        self.meta.statsAlgorithm = enc(val)

    @property
    def genFilterNames(self) -> str:
        return env(self.meta.genFilterNames)

    @genFilterNames.setter
    def genFilterNames(self, str val) -> str:
        self.meta.genFilterNames = enc(val)

    @property
    def file_type(self) -> str:
        return env(self.meta.file_type)

    @file_type.setter
    def file_type(self, str val):
        self.meta.file_type = enc(val)

    @property
    def DatasetName(self) -> str:
        return env(self.ptr.DatasetName())

    @DatasetName.setter
    def DatasetName(self, str val):
        self.meta.DatasetName = enc(val)

    @property
    def event_index(self) -> int:
        return self.meta.event_index

    @event_index.setter
    def event_index(self, int val):
        self.meta.event_index = val




    # constant properties
    @property
    def original_name(self) -> str:
        return env(self.meta.original_name)

    @property
    def original_path(self) -> str:
        return env(self.meta.original_path)

    @property
    def hash(self) -> str:
        self.ptr.Hash()
        return env(self.ptr.hash)

    @property
    def keywords(self) -> list:
        return [env(i) for i in self.meta.keywords]

    @property
    def weights(self) -> list:
        return [env(i) for i in self.meta.weights]

    @property
    def keyword(self) -> list:
        return [env(i) for i in self.meta.keyword]

    @property
    def found(self) -> bool:
        return self.meta.found

    @property
    def config(self) -> dict:
        cdef pair[string, string] it
        cdef dict out = {}
        for it in self.meta.config: out[env(it.first)] = env(it.second)
        return out

    @property
    def GetLengthTrees(self) -> dict:
        cdef map[string, int] x = self.ptr.GetLength()
        cdef pair[string, int] it
        cdef dict out = {env(it.first) : it.second for it in x}
        return out

    @property
    def MissingTrees(self) -> list:
        return [env(i) for i in self.meta.mis_trees]

    @property
    def MissingBranches(self) -> list:
        return [env(i) for i in self.meta.mis_branches]

    @property
    def MissingLeaves(self) -> list:
        return [env(i) for i in self.meta.mis_leaves]

    @property
    def DAODList(self) -> list:
        cdef vector[string] out = self.ptr.DAODList()
        cdef string i
        return [env(i) for i in out]

    @property
    def Files(self) -> dict:
        cdef pair[int, string] it
        cdef dict out = {}
        for it in self.meta.inputfiles:
            out[it.first] = env(it.second)
        return out

    @property
    def DAOD(self) -> str:
        return self.IndexToSample(self.ptr.event_index)

    @property
    def fileGUID(self) -> dict:
        cdef pair[string, int] it
        cdef dict output = {}
        cdef string guid
        for it in self.meta.LFN:
            guid = self.meta.fileGUID.at(it.second)
            output[env(it.first)] = env(guid)
        return output

    @property
    def events(self) -> dict:
        cdef pair[string, int] it
        cdef dict output = {}
        for it in self.meta.LFN:
            output[env(it.first)] = self.meta.events.at(it.second)
        return output

    @property
    def fileSize(self) -> dict:
        cdef pair[string, int] it
        cdef dict output = {}
        for it in self.meta.LFN:
            output[env(it.first)] = self.meta.filesSize.at(it.second)
        return output
