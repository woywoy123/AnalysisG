# distutils: language=c++
# cython: language_level=3

from cython.operator cimport dereference as deref
from libcpp.string cimport string
from libcpp.map cimport pair, map
from libcpp.vector cimport vector

from AnalysisG import auth_pyami
from AnalysisG.core.structs cimport meta_t, weights_t
from AnalysisG.core.tools cimport *
from AnalysisG.core.meta cimport *
from AnalysisG.core.notification cimport *

import pyAMI.client
import pyAMI.httpclient
import pyAMI_atlas.api
import http.client
import pickle
import h5py

class httpx(pyAMI.httpclient.HttpClient):

    def __init__(self, config):
        super(httpx, self).__init__(config)

    def connect(self, endpoint):
        self.endpoint = endpoint
        cdef dict chn = {"certfile" : self.config.cert_file, "keyfile" : self.config.key_file}
        cdef dict confx = {"host" : str(self.endpoint["host"]), "port" : int(self.endpoint["port"])}
        confx["context"] = self.create_unverified_context()
        confx["context"].load_cert_chain(**chn)
        self.connection = http.client.HTTPSConnection(**confx)

class atlas(pyAMI.client.Client):

    def __init__(self):
        super(atlas, self).__init__("atlas-replica")
        self.httpClient = None
        try: self.httpClient = httpx(self.config)
        except: print("Failed to Authenticate to PyAMI.")
        self.authenticated = self.httpClient is not None

cdef class ami_client:

    def __cinit__(self):
        self.client = atlas()
        self.nf = new notification()
        self.nf.prefix = b"PyAMI-MetaScan"

    def __init__(self): self.type_ = "DAOD_TOPQ1"
    def __dealloc__(self): del self.nf

    cdef bool loadcache(self, Meta obj):
        self.dsids = []
        self.datas = {}
        self.infos = {}
        self.file_cache = None
        cdef str dsidr = str(obj.dsid)
        try: self.file_cache = h5py.File(obj.MetaCachePath, "a")
        except FileNotFoundError: self.file_cache = h5py.File(obj.MetaCachePath, "w")
        except OSError: self.file_cache = h5py.File(obj.MetaCachePath, "w")
        except: return False

        cdef string cached_dsid
        try: cached_dsid = enc(self.file_cache[dsidr].attrs["dsids"])
        except KeyError: pass

        cdef string cached_maps
        try: cached_maps = enc(self.file_cache[dsidr].attrs["datasets"])
        except KeyError: pass

        cdef string cached_info
        try: cached_info = enc(self.file_cache[dsidr].attrs["infos"])
        except KeyError: pass

        if cached_dsid.size(): self.dsids = pickle.loads(tools().decode64(&cached_dsid))
        if cached_maps.size(): self.datas = pickle.loads(tools().decode64(&cached_maps))
        if cached_info.size(): self.infos = pickle.loads(tools().decode64(&cached_info))
        return len(self.dsids)*len(self.datas)*len(self.infos)

    cdef void savecache(self, Meta obj):
        cdef str dsidr = str(obj.dsid)

        cdef string cached_dsid = pickle.dumps(self.dsids)
        cdef string cached_maps = pickle.dumps(self.datas)
        cdef string cached_info = pickle.dumps(self.infos)

        try: ref = self.file_cache.create_dataset(dsidr, (1), dtype = h5py.ref_dtype)
        except: ref = self.file_cache[dsidr]
        ref.attrs["dsids"] = tools().encode64(&cached_dsid)
        ref.attrs["datasets"] = tools().encode64(&cached_maps)
        ref.attrs["infos"] = tools().encode64(&cached_info)
        self.file_cache.close()

    cdef void dressmeta(self, Meta obj, str dset_name):
        cdef dict info = dict(self.infos[dset_name][0])
        cdef dict files = {}
        for l in [dict(k) for k in self.datas[dset_name]]: files[l["LFN"]] = l


        cdef list keys = [
                "logicalDatasetName", "identifier", "nFiles", "totalEvents", "totalSize", "dataType", "prodsysStatus", "completion", "ecmEnergy",
                "PDF", "version", "AtlasRelease", "crossSection", "genFiltEff", "datasetNumber", "physicsShort", "generatorName", "geometryVersion",
                "conditionsTag", "generatorTune", "amiStatus", "beamType", "productionStep", "projectName", "statsAlgorithm", "beam_energy",
                "crossSection_mean", "file_type", "genFilterNames", "run_number", "principalPhysicsGroup"
        ]

        for i in keys:
            try: setattr(obj, i, info[i])
            except KeyError: continue

        obj.DatasetName  = info["logicalDatasetName"]
        obj.keywords     = info["keywords"].split(", ")
        obj.keyword      = info["keyword"].split(", ")
        obj.events       = [int(files[i]["events"]) for i in obj.Files.values()]
        obj.fileSize     = [int(files[i]["fileSize"]) for i in obj.Files.values()]
        obj.fileGUID     = [enc(files[i]["fileGUID"]) for i in obj.Files.values()]

        try: obj.kfactor = info["kFactor@PMG"]
        except KeyError: obj.kfactor = 1

        try: obj.weights = info["weights"].split(" | ")[:-1]
        except KeyError: pass
        obj.found = True

    cdef void list_datasets(self, Meta obj):
        if not self.client.authenticated: return
        cdef str dsidr = str(obj.dsid)
        cdef list ami_tags = list(set(obj.amitag.split("_")))
        cdef dict command = {"client" : self.client, "type" : self.type_, "dataset_number" : dsidr}
        cdef bool hit = self.loadcache(obj)
        if not hit:
            try: self.dsids = pyAMI_atlas.api.list_datasets(**command)
            except:
                auth_pyami()
                self.client = atlas()
                self.list_datasets(obj)
                return

            self.nf.success(enc("DSID not in cache, fetched from PyAMI: " + dsidr))
        else: self.nf.success(enc("DSID cache hit for: " + dsidr))

        cdef str i
        cdef bool fset
        cdef list files
        for k in self.dsids:
            fset = False
            dset_name = k["ldn"]
            tag = dset_name.split(".")[-1]
            for i in ami_tags:
                if i not in tag: continue
                fset = True
                break

            if dset_name not in self.datas:
                self.datas[dset_name] = pyAMI_atlas.api.list_files(self.client, dset_name)
                self.nf.success(enc("Fetched Dataset MetaData for " + dset_name))
                hit = False

            if dset_name not in self.infos:
                self.infos[dset_name] = pyAMI_atlas.api.get_dataset_info(self.client, dset_name)
                hit = False

            if not fset: continue
            is_dataset = True
            files = self.datas[dset_name]
            for i in list(obj.Files.values()): is_dataset *= len([1 for t in files if t["LFN"] == i])
            if not is_dataset: continue
            if obj.found: continue
            self.dressmeta(obj, dset_name)
        if hit: return
        self.savecache(obj)

cdef class Meta:

    def __cinit__(self):
        self.ptr = new meta()
        self.loaded = False

    def __init__(self, inpt = None):
        if inpt is None: return
        for i in [i for i in self.__dir__() if not i.startswith("__")]:
            try: setattr(self, i, inpt[i])
            except KeyError: continue
            except: continue
        self.ptr.meta_data = <meta_t>inpt["ptr"]
        self.ptr.metacache_path = enc(inpt["MetaCachePath"])

    def __dealloc__(self): del self.ptr

    cdef __meta__(self, meta* _meta):
        cdef ami_client ami = ami_client()
        if not self.loaded: self.ptr.meta_data = _meta.meta_data
        cdef map[string, weights_t] wx = _meta.meta_data.misc
        if wx.size(): self.ptr.meta_data.misc = wx

        ami.list_datasets(self)
        _meta.meta_data = self.ptr.meta_data

    def __reduce__(self):
        cdef list keys = [i for i in self.__dir__() if not i.startswith("__")]
        cdef dict out = {"ptr" : self.ptr.meta_data}
        out |= {i : getattr(self, i) for i in keys if not callable(getattr(self, i))}
        return (self.__class__, (out,))

    def __str__(self):
        out = ""
        for i in self.__dir__():
            if i.startswith("__"): continue
            try: out += i + " -> " + str(getattr(self, i)) + "\n"
            except: pass
        return out

    def expected_events(self, float lumi = 140.1):
        cdef float s = self.crossSection
        if s < 0: return 0
        else: return s*lumi

    def GetSumOfWeights(self, str name):
        cdef float f = self.ptr.meta_data.misc[enc(name)].processed_events_weighted
        if not f: return 1
        return f

    def FetchMeta(self, int dsid, str amitag):
        self.dsid = dsid
        self.amitag = amitag
        self.__meta__(self.ptr)

    def hash(self, str val): return env(self.ptr.hash(enc(val)))

    @property
    def MetaCachePath(self): return env(self.ptr.metacache_path)

    @MetaCachePath.setter
    def MetaCachePath(self, str val): self.ptr.metacache_path = enc(val)

    @property
    def SumOfWeights(self): return self.ptr.meta_data.misc

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
    def ecmEnergy(self, val):
        self.ptr.meta_data.ecmEnergy = float(val)

    @property
    def genFiltEff(self) -> float:
        return self.ptr.meta_data.genFiltEff

    @property
    def kfactor(self) -> float:
        return self.ptr.meta_data.kfactor

    @kfactor.setter
    def kfactor(self, val): self.ptr.meta_data.kfactor = float(val)

    @genFiltEff.setter
    def genFiltEff(self, val):
        self.ptr.meta_data.genFiltEff = float(val)

    @property
    def completion(self) -> float:
        return self.ptr.meta_data.completion

    @completion.setter
    def completion(self, val):
        self.ptr.meta_data.completion = float(val)

    @property
    def beam_energy(self) -> float:
        return self.ptr.meta_data.beam_energy

    @beam_energy.setter
    def beam_energy(self, val):
        self.ptr.meta_data.beam_energy = float(val)

    @property
    def crossSection(self) -> float:
        return self.ptr.meta_data.crossSection*(10**6)

    @crossSection.setter
    def crossSection(self, val):
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
    def totalSize(self, val):
        self.ptr.meta_data.totalSize = float(val)

    @property
    def nFiles(self) -> int:
        return self.ptr.meta_data.nFiles

    @nFiles.setter
    def nFiles(self, val):
        self.ptr.meta_data.nFiles = int(val)

    @property
    def run_number(self):
        return self.ptr.meta_data.run_number

    @run_number.setter
    def run_number(self, val):
        if isinstance(val, str): val = eval(val)
        if not isinstance(val, list): val = [val]
        self.ptr.meta_data.run_number = val

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

    @property
    def DatasetName(self) -> str:
        return env(self.ptr.meta_data.DatasetName)

    @DatasetName.setter
    def DatasetName(self, str val):
        self.ptr.meta_data.DatasetName = enc(val)

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


    @property
    def keywords(self) -> list:
        return [env(i) for i in self.ptr.meta_data.keywords]

    @keywords.setter
    def keywords(self, val):
        self.ptr.meta_data.keywords = [enc(i) for i in val]

    @property
    def weights(self) -> list:
        return [env(i) for i in self.ptr.meta_data.weights]

    @weights.setter
    def weights(self, val):
        self.ptr.meta_data.weights = [enc(i) for i in val]

    @property
    def keyword(self) -> list:
        return [env(i) for i in self.ptr.meta_data.keyword]

    @keyword.setter
    def keyword(self, val):
        self.ptr.meta_data.keyword = [enc(i) for i in val]

    @property
    def found(self) -> bool:
        return self.ptr.meta_data.found

    @found.setter
    def found(self, val):
        self.ptr.meta_data.found = val


    @property
    def config(self) -> dict:
        cdef dict out = {}
        cdef pair[string, string] it
        for it in self.ptr.meta_data.config: out[env(it.first)] = env(it.second)
        return out

    @property
    def Files(self) -> dict:
        cdef dict out = {}
        cdef pair[int, string] it
        for it in self.ptr.meta_data.inputfiles: out[it.first] = env(it.second)
        return out

    @property
    def fileGUID(self) -> dict:
        cdef pair[string, int] it
        cdef dict output = {}
        cdef string guid
        for it in self.ptr.meta_data.LFN:
            guid = self.ptr.meta_data.fileGUID.at(it.second)
            output[env(it.first)] = env(guid)
        return output


    @fileGUID.setter
    def fileGUID(self, list val):
        self.ptr.meta_data.fileGUID = val

    @property
    def events(self) -> dict:
        cdef pair[string, int] it
        cdef dict output = {}
        for it in self.ptr.meta_data.LFN:
            output[env(it.first)] = self.ptr.meta_data.events.at(it.second)
        return output

    @events.setter
    def events(self, list val):
        if not self.ptr.meta_data.LFN.size():
            fi = list(self.Files.values())
            for i in range(len(fi)): self.ptr.meta_data.LFN[enc(fi[i])] = i
        self.ptr.meta_data.events = val

    @property
    def fileSize(self) -> dict:
        cdef dict output = {}
        cdef pair[string, int] it
        for it in self.ptr.meta_data.LFN:
            output[env(it.first)] = self.ptr.meta_data.fileSize.at(it.second)
        return output

    @fileSize.setter
    def fileSize(self, list val):
        self.ptr.meta_data.fileSize = val

    @property
    def sample_name(self):
        return env(self.ptr.meta_data.sample_name)

    @sample_name.setter
    def sample_name(self, str val):
        self.ptr.meta_data.sample_name = enc(val)


