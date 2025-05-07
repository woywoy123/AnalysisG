/**
 * @file meta.pyx
 * @brief Provides metadata management for the AnalysisG framework.
 */

# distutils: language=c++
# cython: language_level=3

from cython.operator cimport dereference as deref
from libcpp.string cimport string
from libcpp.map cimport pair, map
from libcpp.vector cimport vector
from cython.parallel cimport prange

from AnalysisG import auth_pyami
from AnalysisG.core.structs cimport meta_t, weights_t
from AnalysisG.core.tools cimport *
from AnalysisG.core.meta cimport *
from AnalysisG.core.notification cimport *

import pyAMI.client
import pyAMI.httpclient
try: import pyAMI_atlas.api
except: pass
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
        cdef dict files = {}
        for l in [dict(k) for k in self.datas[dset_name]]: files[l["LFN"]] = l
        try: obj.events  = [int(files[i]["events"]) for i in obj.Files.values()]
        except KeyError: return

        cdef list keys = [
                "logicalDatasetName", "identifier",
                "nFiles", "totalEvents", "totalSize",
                "dataType", "prodsysStatus", "completion", "ecmEnergy",
                "PDF", "version", "AtlasRelease", "crossSection",
                "genFiltEff", "datasetNumber", "physicsShort",
                "generatorName", "geometryVersion",
                "conditionsTag", "generatorTune", "amiStatus", "beamType",
                "productionStep", "projectName", "statsAlgorithm", "beam_energy",
                "crossSection_mean", "file_type", "genFilterNames", "run_number",
                "principalPhysicsGroup"
        ]
        cdef dict info = dict(self.infos[dset_name][0])
        for i in keys:
            try: setattr(obj, i, info[i])
            except KeyError: pass
            except ValueError: pass

        obj.DatasetName  = info["logicalDatasetName"]
        obj.keywords     = info["keywords"].split(", ")
        obj.keyword      = info["keyword"].split(", ")
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
        cdef list ami_tags = sum([i.split(".") for i in list(set(obj.amitag.split("_")))], [])
        cdef dict command = {"client" : self.client, "type" : self.type_, "dataset_number" : dsidr, "show_archived" : True}
        cdef bool hit = self.loadcache(obj)

        if not hit:
            command["type"] = "DAOD_" + obj.derivationFormat
            try:
                self.dsids = pyAMI_atlas.api.list_datasets(**command)
                for f in obj.Files.values():
                    for k in pyAMI_atlas.api.get_file(self.client, f):
                        if k["logicalDatasetName"] not in self.datas:
                            self.datas[k["logicalDatasetName"]] = []
                            self.dsids += [{"ldn" : k["logicalDatasetName"]}]
                        self.datas[k["logicalDatasetName"]] += [k]
            except:
                auth_pyami()
                self.client = atlas()
                self.list_datasets(obj)
                return

            self.nf.success(enc("DSID not in cache, fetched from PyAMI: " + dsidr))
        else: self.nf.success(enc("DSID cache hit for: " + dsidr))

        cdef bool fset
        cdef list files
        cdef str i, dset_name, tag

        cdef list srch = list(obj.Files.values())
        cdef int cx = 0
        cdef int lxn = len(srch)

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

            if not fset or obj.found: continue
            self.dressmeta(obj, dset_name)

        if not obj.found:
            for k in self.dsids:
                dset_name = k["ldn"]

                # ----- Fall back strategy ----- #
                try: files = [t["LFN"] for t in self.datas[dset_name] if "LFN" in t]
                except KeyError: continue

                cx = 0
                for i in files: cx += 1 if i in srch else 0
                if cx != lxn: continue

                self.dressmeta(obj, dset_name)
                if obj.found: break
        if hit: return
        self.savecache(obj)


cdef class MetaLookup:

    def __cinit__(self):
        self.metadata = {}
        self.matched  = {}

        self.meta = None
        self.luminosity = 140.1

    def __init__(self, MetaLookup data = None):
        if data is None: return
        self.metadata = data.metadata

    cdef Meta __find__(self, str inpt):
        try: self.meta = self.matched[inpt]
        except: self.meta = None
        if self.meta is not None: return self.meta

        cdef str i
        cdef Meta mtl
        for i in self.metadata:
            mtl = self.metadata[i]
            if mtl.hash(inpt) not in i: continue
            self.matched[inpt] = mtl
            self.meta = mtl
            return mtl
        return None

    def __call__(self, inpt):
        cdef str ds
        try: ds = inpt.decode("utf-8")
        except: ds = inpt
        self.__find__(ds)
        return self

    @property
    def DatasetName(self): return self.meta.DatasetName
    @property
    def CrossSection(self): return self.meta.crossSection
    @property
    def ExpectedEvents(self): return self.meta.crossSection*self.luminosity
    @property
    def SumOfWeights(self): return 1
    @property
    def GenerateData(self): return Data(self)

cdef class Data:

    def __cinit__(self): pass
    def __init__(self, mtl): self._meta = mtl

    def __add__(self, Data other):
        self.weights = other._weights
        self.data    = other._data
        return self

    def __radd__(self, other):
        if not isinstance(other, int): return self.__add__(other)
        return self._meta.GenerateData.__add__(self)

    cdef void __populate__(self, dict inpt, map[string, vector[float]]* ptx):
        cdef int i
        cdef string fname, key
        cdef vector[float] val
        cdef list names = list(inpt)
        for i in range(len(names)):
            try: fname = names[i].encode("utf-8")
            except: fname = names[i]
            val = inpt[names[i]]
            key = self._meta(fname).DatasetName.encode("utf-8")
            deref(ptx)[fname].insert(deref(ptx)[fname].end(), val.begin(), val.end())
            if self.sumofweights[key].count(fname): continue
            self.sumofweights[key][fname] = self._meta.SumOfWeights
            self.expected_events[key] = self._meta.ExpectedEvents

    cdef void __rescale__(self, vector[float]* ptx):
        cdef int i
        cdef float scale
        cdef vector[float] lst
        cdef pair[string, float] ix
        cdef pair[string, map[string, float]] itr
        for itr in self.sumofweights:
            scale = sum([ix.second for ix in itr.second])
            scale = self.expected_events[itr.first]/scale
            lst   = sum([self._weights[ix.first] for ix in itr.second], [])
            for i in prange(lst.size(), nogil = True): lst[i] = lst[i]*scale
            ptx.insert(ptx.end(), lst.begin(), lst.end())

    @property
    def weights(self):
        cdef vector[float] tx
        self.__rescale__(&tx)
        return tx

    @weights.setter
    def weights(self, dict val): self.__populate__(val, &self._weights)

    @property
    def data(self):
        cdef vector[float] out
        cdef pair[string, vector[float]] itx
        for itx in self._data: out.insert(out.end(), itx.second.begin(), itx.second.end())
        return out

    @data.setter
    def data(self, dict val): self.__populate__(val, &self._data)


/**
 * @class Meta
 * @brief Handles metadata properties and their interactions.
 */
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
        """
        @brief Gets the dataset ID.
        @return An integer representing the dataset ID.
        """
        return self.ptr.meta_data.dsid

    @dsid.setter
    def dsid(self, int val):
        """
        @brief Sets the dataset ID.
        @param val An integer representing the dataset ID.
        """
        self.ptr.meta_data.dsid = val

    @property
    def amitag(self) -> str:
        """
        @brief Gets the AMI tag.
        @return A string representing the AMI tag.
        """
        return env(self.ptr.meta_data.AMITag)

    @amitag.setter
    def amitag(self, str val):
        """
        @brief Sets the AMI tag.
        @param val A string representing the AMI tag.
        """
        self.ptr.meta_data.AMITag = enc(val)

    @property
    def generators(self) -> str:
        """
        @brief Gets the generator names.
        @return A string representing the generator names.
        """
        return env(self.ptr.meta_data.generators)

    @generators.setter
    def generators(self, str val):
        """
        @brief Sets the generator names.
        @param val A string representing the generator names.
        """
        val = val.replace("+", " ")
        self.ptr.meta_data.generators = enc(val)

    @property
    def isMC(self) -> bool:
        """
        @brief Checks if the dataset is Monte Carlo.
        @return A boolean indicating if the dataset is Monte Carlo.
        """
        return self.ptr.meta_data.isMC

    @isMC.setter
    def isMC(self, bool val):
        """
        @brief Sets the Monte Carlo status of the dataset.
        @param val A boolean indicating if the dataset is Monte Carlo.
        """
        self.ptr.meta_data.isMC = val

    @property
    def derivationFormat(self) -> str:
        """
        @brief Gets the derivation format.
        @return A string representing the derivation format.
        """
        return env(self.ptr.meta_data.derivationFormat)

    @derivationFormat.setter
    def derivationFormat(self, str val):
        """
        @brief Sets the derivation format.
        @param val A string representing the derivation format.
        """
        self.ptr.meta_data.derivationFormat = enc(val)

    @property
    def eventNumber(self) -> double:
        """
        @brief Gets the event number.
        @return A double representing the event number.
        """
        return self.ptr.meta_data.eventNumber

    @eventNumber.setter
    def eventNumber(self, double val):
        """
        @brief Sets the event number.
        @param val A double representing the event number.
        """
        self.ptr.meta_data.eventNumber = val

    @property
    def ecmEnergy(self) -> float:
        """
        @brief Gets the center-of-mass energy.
        @return A float representing the center-of-mass energy.
        """
        return self.ptr.meta_data.ecmEnergy

    @ecmEnergy.setter
    def ecmEnergy(self, val):
        """
        @brief Sets the center-of-mass energy.
        @param val A float representing the center-of-mass energy.
        """
        self.ptr.meta_data.ecmEnergy = float(val)

    @property
    def genFiltEff(self) -> float:
        """
        @brief Gets the generator filter efficiency.
        @return A float representing the generator filter efficiency.
        """
        return self.ptr.meta_data.genFiltEff

    @genFiltEff.setter
    def genFiltEff(self, val):
        """
        @brief Sets the generator filter efficiency.
        @param val A float representing the generator filter efficiency.
        """
        self.ptr.meta_data.genFiltEff = float(val)

    @property
    def kfactor(self) -> float:
        """
        @brief Gets the k-factor.
        @return A float representing the k-factor.
        """
        return self.ptr.meta_data.kfactor

    @kfactor.setter
    def kfactor(self, val):
        """
        @brief Sets the k-factor.
        @param val A float representing the k-factor.
        """
        self.ptr.meta_data.kfactor = float(val)

    @property
    def completion(self) -> float:
        """
        @brief Gets the completion percentage.
        @return A float representing the completion percentage.
        """
        return self.ptr.meta_data.completion

    @completion.setter
    def completion(self, val):
        """
        @brief Sets the completion percentage.
        @param val A float representing the completion percentage.
        """
        self.ptr.meta_data.completion = float(val)

    @property
    def beam_energy(self) -> float:
        """
        @brief Gets the beam energy.
        @return A float representing the beam energy.
        """
        return self.ptr.meta_data.beam_energy

    @beam_energy.setter
    def beam_energy(self, val):
        """
        @brief Sets the beam energy.
        @param val A float representing the beam energy.
        """
        self.ptr.meta_data.beam_energy = float(val)

    @property
    def crossSection(self) -> float:
        """
        @brief Gets the cross-section.
        @return A float representing the cross-section in microbarns.
        """
        return self.ptr.meta_data.crossSection*(10**6)

    @crossSection.setter
    def crossSection(self, val):
        """
        @brief Sets the cross-section.
        @param val A float representing the cross-section in microbarns.
        """
        try: self.ptr.meta_data.crossSection = float(val)
        except ValueError: self.ptr.meta_data.crossSection = -1

    @property
    def generatorName(self) -> str:
        """
        @brief Gets the generator name from the metadata.
        @return A string representing the generator name.
        """
        return env(self.ptr.meta_data.generatorName)

    @generatorName.setter
    def generatorName(self, str val):
        """
        @brief Sets the generator name in the metadata.
        @param val A string representing the generator name.
        """
        self.ptr.meta_data.generatorName = enc(val)

    # Additional properties and methods documented similarly...
