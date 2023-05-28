#distutils: language = c++
# cython: language_level=3
from ROOT cimport CySampleTracer, CyROOT, CyEvent
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map, pair
from libcpp cimport bool
import h5py
import os
import sys
import shutil
import pickle 
import codecs
from tqdm import tqdm
from typing import Union 
from AnalysisG.Tools import Code, Threading

cdef class Event:
    cdef CyEvent* ptr
    cdef public list _instance
    cdef public bool _demanded
    def __cinit__(self):
        self.ptr = NULL
        self._demanded = False

    def __init__(self): self._instance = []
    def _wrap(self, val): self._instance.append(val)
    def __setstate__(self, inpt): setattr(self, "_instance", inpt["_instance"])
    def __getattr__(self, attr):
        self.demand()
        try: return getattr(self._instance[0], attr)
        except AttributeError: pass
        return getattr(self._instance[1], attr)

    def __getstate__(self):
        out = {}
        out["_instance"] = self._instance
        return out
    
    def __eq__(self, other) -> bool:
        if isinstance(self, str): return False
        if isinstance(other, str): return False
        if self.hash != other.hash: return False
        if self.Tree != other.Tree: return False
        try: return self.Event == other.Event
        except AttributeError: pass
        try: return self.Graph == other.Graph 
        except AttributeError: pass
        return False

    def demand(self) -> None:
        if self._demanded: return
        try: self._instance[0] = pickle.loads(self._instance[0])
        except TypeError: pass
        self._demanded = True
        return 
 
    @property
    def index(self) -> int: return self.ptr.EventIndex 

    @property
    def Tree(self) -> str: return self.ptr.Tree.decode("UTF-8")

    @property
    def Graph(self) -> bool: return self.ptr.Graph
    
    @property
    def Event(self) -> bool: return self.ptr.Event

    @property
    def hash(self) -> str: return self.ptr.hash.decode("UTF-8")

    @property 
    def ROOT(self) -> str: return self.ptr.ROOT.decode("UTF-8")
   
    @property
    def CachePath(self) -> str: return self.ptr.ROOTFile.CachePath.decode("UTF-8")

    @property
    def TrainMode(self) -> str: return self.ptr.TrainMode.decode("UTF-8")

    @TrainMode.setter
    def TrainMode(self, val): self.ptr.TrainMode = <string>val.encode("UTF-8")

cdef class SampleTracer:
    cdef CySampleTracer* ptr
    cdef vector[string] _itv
    cdef map[string, string] _HashMeta
    cdef dict HashMeta
    cdef dict HashMap
    cdef int _its
    cdef int _ite
    cdef str OutDir 
    cdef bool _EventCache 
    cdef bool _DataCache
    cdef str _SampleName
    cdef dict _Codes

    def __cinit__(self):
        self.ptr = new CySampleTracer()
        self.HashMap = {}
        self.HashMeta = {}
        self.OutDir = os.getcwd()
        self._SampleName = ""
        self._Codes = {}

    def __dealloc__(self): del self.ptr
    def __init__(self): pass
    
    def __contains__(self, str key) -> bool:
        if key.endswith(".root"): key = os.path.abspath(key)
        if self.ptr.ContainsROOT(key.encode("UTF-8")): return True 
        if self.ptr.ContainsHash(key.encode("UTF-8")): return True
        if key in self.HashMeta: return True
        return False
    
    def __getitem__(self, str key):
        cdef string _key = key.encode("UTF-8")
        cdef string _i
        cdef vector[string] r
        cdef str daod
        cdef list out = []

        r = self.ptr.ROOTtoHashList(_key)
        if r.size() > 0: 
            for _i in r:
                daod = self._HashMeta[_i].decode("UTF-8")
                if len(daod) != 18: 
                    self.RestoreMeta
                    daod = self._HashMeta[_i].decode("UTF-8")
                ev = Event()
                ev.ptr = self.ptr.HashToEvent(_i)
                ev._wrap(self.HashMap[_i.decode("UTF-8")])
                ev._wrap(self.HashMeta[daod])
                out.append(ev)
            return out
        
        if self.ptr.ContainsHash(_key): 
            daod = self._HashMeta[_key].decode("UTF-8")
            if len(daod) != 18: 
                self.RestoreMeta
                daod = self._HashMeta[_key].decode("UTF-8")
            ev = Event()
            ev._wrap(self.HashMap[key])
            ev._wrap(self.HashMeta[daod])
            ev.ptr = self.ptr.HashToEvent(_key)
            return ev

        if key.endswith(".root"): key = os.path.abspath(key)
        if key in self.HashMeta: 
            for daod in self.HashMeta[key].Files:
                try: out += self[self.HashMeta[key].thisSet + daod]
                except: continue
            return out
        return out

    def __len__(self) -> int:
        if not self.EventCache and not self.DataCache: self._itv = self.ptr.HashList()
        else: self._itv = self.ptr.GetCacheType(self.EventCache, self.DataCache)
        self._ite = self._itv.size()
        self.ptr.length = self._ite
        self.RestoreMeta
        return self.ptr.length

    def __add__(self, other) -> SampleTracer:
        if isinstance(self, int): return other
        if isinstance(other, int): return self
        cdef SampleTracer s = self
        cdef SampleTracer o = other
        cdef SampleTracer t = self.clone
        del t.ptr
        t.ptr = s.ptr[0] + o.ptr
        t.HashMap |= s.HashMap
        t.HashMap |= o.HashMap
        t.HashMeta |= s.HashMeta
        t.HashMeta |= o.HashMeta
        t._Codes |= s._Codes
        t._Codes |= o._Codes
        t.RestoreMeta
        return t
    
    def __iadd__(self, other):
        cdef SampleTracer s = self
        cdef SampleTracer o = other
        s.ptr = s.ptr[0] + o.ptr
        s.HashMap |= o.HashMap
        s.HashMeta |= o.HashMeta
        s._Codes |= o._Codes
        s.Threads = <int>self.Threads
        return s
    
    @property
    def clone(self):
        v = self.__new__(self.__class__)
        v.__init__()
        return v
    
    def HashToROOT(self, str key) -> str: return self._HashMeta[key.encode("UTF-8")].decode("UTF-8")
    def _decoder(self, str inpt): return pickle.loads(codecs.decode(inpt.encode("UTF-8"), "base64"))
    def _encoder(self, inpt) -> str: return codecs.encode(pickle.dumps(inpt), "base64").decode()
     
    @property
    def Threads(self) -> int: return self.ptr.Threads

    @property
    def chnk(self) -> int: return self.ptr.ChunkSize

    @Threads.setter
    def Threads(self, int val): self.ptr.Threads = val

    @chnk.setter
    def chnk(self, int val): self.ptr.ChunkSize = val
    
    @property
    def OutputDirectory(self): return self.OutDir
    
    @OutputDirectory.setter
    def OutputDirectory(self, str val): self.OutDir = val
    
    @property
    def EventCache(self) -> bool: return self._EventCache 

    @EventCache.setter
    def EventCache(self, bool val): self._EventCache = val

    @property
    def DataCache(self) -> bool: return self._DataCache 

    @DataCache.setter
    def DataCache(self, bool val): self._DataCache = val
    
    @property
    def SampleName(self) -> str: return self._SampleName 

    @SampleName.setter
    def SampleName(self, str val): self._SampleName = val

    @property
    def todict(self) -> dict: return {i.hash : i for i in self}

    def FastHashSearch(self, list hashes) -> dict:
        cdef vector[string] v
        cdef int i
        cdef string key
        cdef bool fnd

        for i in range(len(hashes)): v.push_back(hashes[i].encode("UTF-8"))
        return {key.decode("UTF-8") : fnd for key, fnd in self.ptr.FastSearch(v)}

    def AddEvent(self, dict Events):
        cdef str i, p
        cdef dict x
        cdef list hashes = list(Events)
        if len(self._Codes) == 0: 
            p = hashes[0]
            cl = Code(pickle.loads(Events[p]["pkl"]))
            x = cl.clone.Objects
            self._Codes |= {t._Name : t.purge for t in [Code(x[p]) for p in x]}
            self._Codes[cl._Name] = cl.purge
            del cl
        
        x = self.FastHashSearch(hashes)    
        for i in x:
            if x[i]: continue
            _ev = new CyEvent()
            _ev.Tree = Events[i]["Tree"].encode("UTF-8")
            _ev.hash = i.encode("UTF-8")
            _ev.ROOT = Events[i]["ROOT"].encode("UTF-8")
            _ev.EventIndex = <unsigned int>Events[i]["index"]
            _ev.Event = True
            self.ptr.AddEvent(_ev) 
            self.HashMap[i] = Events[i]["pkl"]
            self.HashMeta[Events[i]["Meta"].ROOTName] = Events[i]["Meta"]

    def AddGraph(self, Events):
        cdef int i;
        cdef str hash; 
        cdef CyEvent* event;
        for i in range(len(Events)):
            hash, evnt = Events[i]
            self.HashMap[hash] = pickle.dumps(evnt)
            event = self.ptr.HashToEvent(<string>hash.encode("UTF-8"))
            event.Graph = True 
            event.Event = False

    @property
    def DumpTracer(self):
        cdef pair[string, CyROOT*] root
        cdef pair[string, CyEvent*] event
        cdef CyROOT* r
        cdef CyEvent* e
        cdef str _meta
      
        try: os.mkdir(self.OutDir)
        except FileExistsError: pass
        try: os.mkdir(self.OutDir + "/Tracer")
        except FileExistsError: pass
 
        for root in self.ptr._ROOTMap:
            r = root.second
            f = h5py.File(self.OutDir + "/Tracer/" + r.CachePath.decode("UTF-8") + ".hdf5", "a")
            try: ref_s = f.create_dataset("SampleNames", (1), dtype = h5py.ref_dtype)
            except ValueError: ref_s = f["SampleNames"]
            if self._SampleName != "": ref_s.attrs[self._SampleName] = True

            for _meta in self.HashMeta:
                if not self.HashMeta[_meta].MatchROOTName(root.first.decode("UTF-8")): continue
                try: ref_s = f.create_dataset("MetaData", (1), dtype = h5py.ref_dtype)
                except ValueError: ref_s = f["MetaData"]
                ref_s.attrs[_meta] = self._encoder(self.HashMeta[_meta])
           
            for event in r.HashMap:
                e = event.second
                try: ref = f.create_dataset(event.first.decode("UTF-8"), (1), dtype = h5py.ref_dtype)
                except ValueError: ref = f[event.first.decode("UTF-8")]
                
                ref.attrs["Tree"] = e.Tree
                ref.attrs["TrainMode"] = e.TrainMode
                ref.attrs["Hash"] = e.hash
                ref.attrs["ROOT"] = e.ROOT
                ref.attrs["EventIndex"] = e.EventIndex
                ref.attrs["CachedEvent"] = e.CachedEvent
                ref.attrs["CachedGraph"] = e.CachedGraph

            f.close()
    
    @property
    def DumpEvents(self):
        def Function(inpt, _prgbar):
            lock, bar = _prgbar
            for j in range(len(inpt)):
                inpt[j] = [inpt[j][0], codecs.encode(inpt[j][1], "base64").decode()]
                with lock: bar.update(1)
            return inpt

        cdef CyEvent* e
        cdef CyROOT* r 
        cdef str out, l
        cdef file = ""

        try: os.mkdir(self.OutDir)
        except FileExistsError: pass

        th = Threading([[i, self.HashMap[i]] for i in self.HashMap], Function, self.Threads)
        th.Title = "TRACER::DUMPING"
        th.Start
        cdef dict Encodes = {i[0] : i[1] for i in th._lists}
        del th 

        _, bar = self._MakeBar(len(self), "TRACER::DUMPING EVENTS")
        for i in self.HashMap:
            e = self.ptr.HashToEvent(<string>i.encode("UTF-8"))
            r = self.ptr.HashToCyROOT(<string>i.encode("UTF-8"))
            if e.CachedEvent and e.Event: continue
            elif e.CachedGraph and e.Graph: continue
            
            bar.update(1)
            if e.Event: out = self.OutDir + "/EventCache/"
            elif e.Graph: out = self.OutDir + "/DataCache/"
            else: continue
            
            if file == "":
                try: os.mkdir(out)
                except FileExistsError: pass

            out += r.CachePath.decode("UTF-8")
            if file != out: f = h5py.File(out + ".hdf5", "a")
            try: ref = f.create_dataset(i, (1), dtype = h5py.ref_dtype)
            except ValueError: ref = f[i]
            
            if e.Event: e.CachedEvent = True
            else: e.CachedGraph = True
            ref.attrs["Event"] = Encodes[i]

            file = out if file != "" else file
            if file != out: 
                try: ref = f.create_dataset("code", (1), dtype = h5py.ref_dtype)
                except ValueError: ref = f["code"]
                for l in self._Codes: ref.attrs[l] = self._encoder(self._Codes[l])
                f.close()
        del bar
        self.DumpTracer

    @property
    def RestoreTracer(self):
        cdef str i, k
        cdef CyROOT* R
        cdef CyEvent* E
        cdef vector[string] search
        cdef dict maps 
        cdef list get

        for i in [self.OutDir + "/Tracer/" + i for i in os.listdir(self.OutDir + "/Tracer/") if ".hdf5" in i]:
            f = h5py.File(i)
            try:
                if self._SampleName == "": raise KeyError
                try: f["SampleNames"].attrs[self._SampleName]
                except KeyError: continue
            except KeyError: pass
            self.HashMeta |= {i : self._decoder(f["MetaData"].attrs[i]) for i in f["MetaData"].attrs}
            
            maps = self.FastHashSearch([k for k in f if k != "SampleNames" and k != "code" and k != "MetaData"])
            for i in maps:
                if maps[i]: continue
                ref = f[i]
                E = new CyEvent()
                E.Tree = <string>ref.attrs["Tree"].encode("UTF-8")
                E.TrainMode = <string>ref.attrs["TrainMode"].encode("UTF-8")
                E.ROOT = <string>ref.attrs["ROOT"].encode("UTF-8")
                E.EventIndex = <unsigned int>ref.attrs["EventIndex"]
                E.CachedEvent = ref.attrs["CachedEvent"]
                E.CachedGraph = ref.attrs["CachedGraph"]
                E.Hash()
                self.ptr.AddEvent(E)
  
    @property
    def RestoreMeta(self):
        cdef string _daod, _dataset
        cdef pair[string, CyROOT*] _i
        cdef str _key

        for _i in self.ptr._ROOTHash:
            if self._HashMeta[_i.first].decode("UTF-8") != "": continue
            string = _i.second.SourcePath + _i.second.Filename 
            for _key in self.HashMeta:
                if not self.HashMeta[_key].MatchROOTName(string.decode("UTF-8")): continue 
                break
            self._HashMeta[_i.first] = _key.encode("UTF-8")
    
    @property
    def RestoreEvents(self):
        def Function(inpt, _prgbar):
            lock, bar = _prgbar
            for j in range(len(inpt)):
                inpt[j][1] = codecs.decode(inpt[j][1].encode("UTF-8"), "base64")
                with lock: bar.update(1)
            return inpt 
        
        self.RestoreTracer 
        self.RestoreMeta
        cdef pair[string, CyROOT*] c
        cdef pair[string, CyEvent*] e
        cdef CyROOT* r_
        cdef CyEvent* e_
        cdef str get, k
        cdef bool DataCache, EventCache
        cdef dict files = {}
        cdef list reco = []

        for c in self.ptr._ROOTMap:
            DataCache = self.DataCache 
            EventCache = self.EventCache

            r_ = c.second
            get = self.OutDir
            get += "/DataCache/" if DataCache else "/EventCache/"
            get += r_.CachePath.decode("UTF-8") + ".hdf5"
            if not os.path.isfile(get): 
                get = get.replace("/DataCache/", "/EventCache/")
                DataCache = False
                EventCache = True
            if not os.path.isfile(get): continue
            
            f = h5py.File(get)
            get = "TRACER::READING EVENTS (" + ("DataCache" if DataCache else "EventCache") + "): "
            _, bar = self._MakeBar(r_.HashMap.size(), get + c.first.decode("UTF-8").split("/")[-1])
            
            if not DataCache: 
                try: os.mkdir("./tmp")
                except FileExistsError: pass
                for k in f["code"].attrs:
                    Code = self._decoder(f["code"].attrs[k])
                    k = Code._File.split("/")[-1]
                    mk = open("./tmp/" + k, "w")
                    mk.write(Code._FileCode)
                    mk.close()
                sys.path.append("./tmp/")

            for e in r_.HashMap:
                bar.update(1)
                e_ = e.second   
                if e_.CachedEvent == False and e_.CachedGraph == False: continue
                elif e_.CachedGraph == False and DataCache: continue
                elif e_.CachedEvent == False and EventCache: continue
                get = e.first.decode("UTF-8")
                self.HashMap[get] = f[get].attrs["Event"]
                if EventCache: e_.Event = True
                if DataCache: e_.Graph = True
                reco.append([get, f[get].attrs["Event"]])

        if len(reco) == 0: return 
        th = Threading(reco, Function, self.Threads)
        th.Title = "TRACER::DECODING"
        th.Start

        cdef list t
        for t in th._lists: self.HashMap[t[0]] = t[1]
        del th

    def ForceTheseHashes(self, inpt: Union[dict, list]) -> None:
        cdef list smpl = inpt if isinstance(inpt, list) else list(inpt)
        cdef str i
        
        self._itv.clear()
        for i in smpl: self._itv.push_back(i.encode("UTF-8"))
        self._ite = self._itv.size()

    def MarkTheseHashes(self, inpt: Union[dict, list], str label) -> None:
        cdef list smpl = inpt if isinstance(inpt, list) else list(inpt)
        cdef string lab = label.encode("UTF-8")
        cdef string _i
        cdef CyEvent* ev
        cdef str i

        for i in inpt:
            _i = i.encode("UTF-8")
            if not self.ptr.ContainsHash(_i): continue
            ev = self.ptr.HashToEvent(_i)
            ev.TrainMode = lab

    def _MakeBar(self, inpt: Union[int], CustTitle: Union[None, str] = None):
        _dct = {}
        _dct["desc"] = f'Progress {self.Caller}' if CustTitle is None else CustTitle
        _dct["leave"] = False
        _dct["colour"] = "GREEN"
        _dct["dynamic_ncols"] = True
        _dct["total"] = inpt
        return (None, tqdm(**_dct))

    def __preiteration__(self): pass

    def __iter__(self):
        if self.__preiteration__(): 
            self._its = 0
            self._ite = 0
            return self
        if self.ptr.length == 0: len(self)
        self._its = 0
        return self

    def __next__(self):
        if self._its == self._ite: raise StopIteration
        cdef string _hash = self._itv[self._its]
        cdef str hash = _hash.decode("UTF-8")
        
        ev = Event()
        ev._wrap(self.HashMap[hash])
        ev._wrap(self.HashMeta[self._HashMeta[_hash].decode("UTF-8")])
        ev.ptr = self.ptr[0].HashToEvent(_hash)
        self._its += 1
        return ev
        
