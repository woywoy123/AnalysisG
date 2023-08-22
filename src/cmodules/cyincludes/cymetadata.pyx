# distuils: language = c++
# cython: language_level = 3

from cymetadata cimport CyMetaData, ExportMetaData
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map, pair
from libcpp cimport bool

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
    cdef public str _hash
    cdef public bool scan_ami

    def __cinit__(self):
        self.ptr = new CyMetaData()
        self.client = None

    def __init__(self, str file, bool scan_ami = True, str sampletype = "DAOD_TOPQ1"):
        self.ptr.original_input = enc(file)
        self.ptr.original_name = enc(file.split("/")[-1])
        self.ptr.original_path = enc("/".join(file.split("/")[:-1]))

        self.sampletype = sampletype
        self.loaded = True
        self.scan_ami = scan_ami

        if not scan_ami:
            self._getMetaData(file)
            return

        warnings.filterwarnings("ignore")
        try: self.client = pyAMI.client.Client("atlas")
        except: pass
        self.loaded = self._getMetaData(file)

    def __dealloc__(self):
        del self.ptr


    def __getstate__(self):
        cdef ExportMetaData tmp = self.ptr.MakeMapping()
        return tmp


    def __str__(self) -> str:
        cdef str i = ""
        cdef str k
        for k in self.__dir__():
            if k.startswith("_"): continue
            i += k + " : " + str(getattr(self, k)) + "\n"
        return i

    def __hash__(self):
        self.ptr.hashing()
        return int(env(self.ptr.hash)[:8], 0)

    def __eq__(self, other):
        if not issubclass(other.__class__, MetaData): return False
        return self._hash == other._hash

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
        try: self.ptr.dsid = out["dsid"]
        except KeyError: k1 = False
        self.ptr.hashing()

        try: self.ptr.AMITag = enc(out["AMITag"])
        except KeyError: k2 = False

        try: self.ptr.generators = enc(out["generators"].replace("+", " "))
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
        self.ptr.isMC = this["isMC"]
        self.ptr.derivationFormat = enc(this["derivationFormat"])

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
        try: self.ptr.eventNumber = self._get(command)["eventNumber"]
        except KeyError: return False
        return True

    def _populate(self, inpt):
        self.ptr.ecmEnergy         = float(inpt["ecmEnergy"])
        self.ptr.genFiltEff        = float(inpt["genFiltEff"])
        self.ptr.completion        = float(inpt["completion"])
        self.ptr.beam_energy       = float(inpt["beam_energy"])
        self.ptr.crossSection      = float(inpt["crossSection"])
        self.ptr.crossSection_mean = float(inpt["crossSection_mean"])
        self.ptr.totalSize         = float(inpt["totalSize"])

        self.ptr.nFiles        = int(inpt["nFiles"])
        self.ptr.run_number    = int(inpt["run_number"][1:-1])
        self.ptr.totalEvents   = int(inpt["totalEvents"])
        self.ptr.datasetNumber = int(inpt["datasetNumber"])

        self.ptr.identifier            = enc(inpt["identifier"])
        self.ptr.prodsysStatus         = enc(inpt["prodsysStatus"])
        self.ptr.dataType              = enc(inpt["dataType"])
        self.ptr.version               = enc(inpt["version"])
        self.ptr.PDF                   = enc(inpt["PDF"])
        self.ptr.AtlasRelease          = enc(inpt["AtlasRelease"])
        self.ptr.principalPhysicsGroup = enc(inpt["principalPhysicsGroup"])
        self.ptr.physicsShort          = enc(inpt["physicsShort"])
        self.ptr.generatorName         = enc(inpt["generatorName"])
        self.ptr.geometryVersion       = enc(inpt["geometryVersion"])
        self.ptr.conditionsTag         = enc(inpt["conditionsTag"])
        self.ptr.generatorTune         = enc(inpt["generatorTune"])
        self.ptr.amiStatus             = enc(inpt["amiStatus"])
        self.ptr.beamType              = enc(inpt["beamType"])
        self.ptr.productionStep        = enc(inpt["productionStep"])
        self.ptr.projectName           = enc(inpt["projectName"])
        self.ptr.statsAlgorithm        = enc(inpt["statsAlgorithm"])
        self.ptr.genFilterNames        = enc(inpt["genFilterNames"])
        self.ptr.file_type             = enc(inpt["file_type"])

        self.ptr.keywords = [enc(v) for v in inpt["keywords"].replace(" ", "").split(",")]
        self.ptr.keyword  = [enc(v) for v in inpt["keyword"].replace(" ", "").split(",")]
        self.ptr.weights  = [enc(v.lstrip(" ").rstrip(" ")) for v in inpt["weights"].split("|")][:-1]

    def _search(self) -> bool:
        def _sig(signum, frame): return ""

        if self.client is None: return False
        signal.signal(signal.SIGALRM, _sig)
        signal.alarm(10)
        warnings.filterwarnings("ignore")

        cdef list x
        try:
            x = atlas.list_datasets(
                self.client, dataset_number = [str(self.ptr.dsid)], type = self.sampletype
            )
        except pyAMI.exception.Error: return False

        cdef dict t
        cdef list tags
        cdef str sample, v, tag
        cdef int index

        tag = env(self.ptr.AMITag)
        tags = list(set(tag.split("_")))
        for i in x:
            try: sample = i["ldn"]
            except KeyError: continue

            if len([v for v in tags if v in sample]) != len(tags): continue

            self.ptr.DatasetName = enc(sample)
            self.ptr.found = True
            self._populate(atlas.get_dataset_info(self.client, sample)[0])

            index = 0
            for k in atlas.list_files(self.client, sample):
                self.ptr.LFN[enc(k["LFN"])] = index
                self.ptr.fileGUID.push_back(enc(k["fileGUID"]))
                self.ptr.events.push_back(int(k["events"]))
                self.ptr.fileSize.push_back(float(k["fileSize"]))
                index += 1
        return self.ptr.found

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

    @property
    def GetLengthTrees(self) -> dict:
        cdef map[string, int] x = self.ptr.GetLength()
        cdef pair[string, int] it
        cdef dict out = {env(it.first) : it.second for it in x}
        return out

    @property
    def MissingTrees(self) -> list:
        return [env(i) for i in self.ptr.mis_trees]

    @property
    def MissingBranches(self) -> list:
        return [env(i) for i in self.ptr.mis_branches]

    @property
    def MissingLeaves(self) -> list:
        return [env(i) for i in self.ptr.mis_leaves]

    @property
    def Trees(self) -> list: return [env(i) for i in self.ptr.req_trees]

    @Trees.setter
    def Trees(self, list val): self.ptr.req_trees = [enc(i) for i in val]

    @property
    def Branches(self) -> list: return [env(i) for i in self.ptr.req_branches]

    @Branches.setter
    def Branches(self, list val): self.ptr.req_branches = [enc(i) for i in val]

    @property
    def Leaves(self) -> list: return [env(i) for i in self.ptr.req_leaves]

    @Leaves.setter
    def Leaves(self, list val): self.ptr.req_leaves = [enc(i) for i in val]


    @property
    def inputName(self) -> str: return env(self.ptr.original_input)

    @property
    def dsid(self) -> int: return self.ptr.dsid

    @property
    def amitag(self) -> str: return env(self.ptr.AMITag)

    @property
    def generators(self) -> str: return env(self.ptr.generators)

    @property
    def isMC(self) -> bool: return self.ptr.isMC

    @property
    def derivationFormat(self) -> str: return env(self.ptr.derivationFormat)

    @property
    def Files(self) -> dict:
        cdef pair[int, string] it
        cdef dict out = {}
        for it in self.ptr.inputfiles: out[it.first] = env(it.second)
        return out

    @property
    def config(self) -> dict:
        cdef pair[string, string] it
        cdef dict out = {}
        for it in self.ptr.config: out[env(it.first)] = env(it.second)
        return out

    @property
    def eventNumber(self) -> int: return self.ptr.eventNumber

    @eventNumber.setter
    def eventNumber(self, int val): self.ptr.eventNumber = val

    @property
    def event_index(self) -> int: return self.ptr.event_index

    @event_index.setter
    def event_index(self, int val): self.ptr.event_index = val

    @property
    def found(self) -> bool: return self.ptr.found

    @property
    def DatasetName(self) -> str:
        cdef str dsid, data, i
        cdef list out
        if self.ptr.DatasetName.size() == 0:
            dsid = str(self.dsid)
            out = [i for i in self.Files.values() if dsid in i]
            if len(out) == 0: return env(self.ptr.DatasetName)
            data = [i for i in out[0].split("/") if dsid in i][0]
            self.ptr.DatasetName = enc(data)
        return env(self.ptr.DatasetName)

    @property
    def ecmEnergy(self) -> float: return self.ptr.ecmEnergy

    @property
    def genFiltEff(self) -> float: return self.ptr.genFiltEff

    @property
    def completion(self) -> float: return self.ptr.completion

    @property
    def beam_energy(self) -> float: return self.ptr.beam_energy

    @property
    def cross_section(self) -> float: return self.ptr.crossSection

    @property
    def cross_section_mean(self) -> float: return self.ptr.crossSection_mean

    @property
    def total_size(self) -> float: return self.ptr.totalSize

    @property
    def nFiles(self) -> int: return self.ptr.nFiles

    @property
    def run_number(self) -> int: return self.ptr.run_number

    @property
    def totalEvents(self) -> int: return self.ptr.totalEvents

    @property
    def datasetNumber(self) -> int: return self.ptr.datasetNumber

    @property
    def identifier(self) -> str: return env(self.ptr.identifier)

    @property
    def prodsysStatus(self) -> str: return env(self.ptr.prodsysStatus)

    @property
    def dataType(self) -> str: return env(self.ptr.dataType)

    @property
    def version(self) -> str: return env(self.ptr.version)

    @property
    def PDF(self) -> str: return env(self.ptr.PDF)

    @property
    def AtlasRelease(self) -> str: return env(self.ptr.AtlasRelease)

    @property
    def principalPhysicsGroup(self) -> str: return env(self.ptr.principalPhysicsGroup)

    @property
    def physicsShort(self) -> str: return env(self.ptr.physicsShort)

    @property
    def generatorName(self) -> str: return env(self.ptr.generatorName)

    @property
    def geometryVersion(self) -> str: return env(self.ptr.geometryVersion)

    @property
    def conditionsTag(self) -> str: return env(self.ptr.conditionsTag)

    @property
    def generatorTune(self) -> str: return env(self.ptr.generatorTune)

    @property
    def amiStatus(self) -> str: return env(self.ptr.amiStatus)

    @property
    def beamType(self) -> str: return env(self.ptr.beamType)

    @property
    def productionStep(self) -> str: return env(self.ptr.productionStep)

    @property
    def projectName(self) -> str: return env(self.ptr.projectName)

    @property
    def statsAlgorithm(self) -> str: return env(self.ptr.statsAlgorithm)

    @property
    def genFilterNames(self) -> str: return env(self.ptr.genFilterNames)

    @property
    def file_type(self) -> str: return env(self.ptr.file_type)

    @property
    def keywords(self) -> list: return [env(i) for i in self.ptr.keywords]

    @property
    def weights(self) -> list: return [env(i) for i in self.ptr.weights]

    @property
    def keyword(self) -> list: return [env(i) for i in self.ptr.keyword]

    @property
    def DAODList(self) -> list:
        cdef pair[string, int] it
        cdef list l = [env(it.first) for it in self.ptr.LFN]
        if len(l): return l
        cdef str key
        return [key.split("/")[-1] for key in self.Files.values()]

    @property
    def DAOD(self) -> str:
        cdef str out = env(self.ptr.IndexToSample(self.ptr.event_index))
        if len(out): return out.split("/")[-1]
        return env(self.ptr.original_name)

    @property
    def FilePath(self) -> str: return env(self.ptr.original_path)

    @property
    def fileGUID(self) -> dict:
        cdef pair[string, int] it
        return {env(it.first) : self.ptr.fileGUID.at(it.second) for it in self.ptr.LFN}

    @property
    def events(self) -> dict:
        cdef pair[string, int] it
        return {env(it.first) : self.ptr.events.at(it.second) for it in self.ptr.LFN}

    @property
    def fileSize(self) -> dict:
        cdef pair[string, int] it
        return {env(it.first) : self.ptr.fileSize.at(it.second) for it in self.ptr.LFN}
