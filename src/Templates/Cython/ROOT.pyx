#distutils: language = c++
# cython: language_level=3
from ROOT cimport CySampleTracer, CyROOT, CyEvent
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map, pair
from libcpp cimport bool
import h5py
import os
import pickle 
import codecs
from tqdm import tqdm
from typing import Union 

cdef class Event:
    cdef CyEvent* ptr
    cdef public list _instance

    def __cinit__(self):
        self.ptr = NULL

    def __init__(self):
        self._instance = []
 
    def _wrap(self, val):
        self._instance.append(val)
   
    def __getattr__(self, attr):
        return getattr(self._instance[0], attr)

    def __getstate__(self):
        out = {}
        out["_instance"] = self._instance
        return out
    
    def __setstate__(self, inpt):
        setattr(self, "_instance", inpt["_instance"])

    def __eq__(self, other) -> bool:
        if isinstance(self, str): return False
        if isinstance(other, str): return False
        if self.hash != other.hash: return False
        try: return self.Event == other.Event
        except AttributeError: return True

    @property
    def index(self) -> int: return self.ptr.EventIndex 

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

cdef class SampleTracer:
    cdef CySampleTracer* ptr
    cdef vector[string] _itv
    cdef dict HashMap
    cdef int _its
    cdef int _ite
    cdef str OutDir 
    cdef bool _EventCache 
    cdef bool _DataCache
    cdef str _SampleName

    def __cinit__(self):
        self.ptr = new CySampleTracer()
        self.HashMap = {}
        self.OutDir = os.getcwd()
        self._SampleName = ""

    def __dealloc__(self):
        del self.ptr

    def __init__(self):
        pass

    def __contains__(self, str key) -> bool:
        if key.endswith(".root"): key = os.path.abspath(key)
        if self.ptr.ContainsROOT(key.encode("UTF-8")): return True 
        if self.ptr.ContainsHash(key.encode("UTF-8")): return True 
        return False
    
    def __getitem__(self, str key):
        cdef string i = key.encode("UTF-8"); 
        cdef vector[string] r
        cdef list out = []

        if key.endswith(".root"): key = os.path.abspath(key)
        if self.ptr.ContainsROOT(i): 
            r = self.ptr.ROOTtoHashList(i)
            for i in r:
                ev = Event()
                ev._wrap(self.HashMap[i.decode("UTF-8")])
                ev.ptr = self.ptr.HashToEvent(i)
                out.append(ev)
            return out
        if self.ptr.ContainsHash(i): 
            ev = Event()
            ev._wrap(self.HashMap[key])
            ev.ptr = self.ptr.HashToEvent(<string>key.encode("UTF-8"))
            return ev
        return False

    def __len__(self) -> int:
        if self.ptr.length == 0: self.ptr.length = self.ptr._ROOTHash.size()
        return self.ptr.length

    def __add__(self, other) -> SampleTracer:
        if isinstance(self, int): return other
        if isinstance(other, int): return self

        cdef SampleTracer s = self
        cdef SampleTracer o = other
        cdef SampleTracer t = self.clone
        del t.ptr
        t.ptr = s.ptr[0] + o.ptr
        for i in s.HashMap: t.HashMap[i] = s.HashMap[i] 
        for i in o.HashMap: t.HashMap[i] = o.HashMap[i]
        return t
    
    def __iadd__(self, other):
        cdef SampleTracer s = self
        cdef SampleTracer o = other
        s.ptr = s.ptr[0] + o.ptr
        for i in o.HashMap: s.HashMap[i] = o.HashMap[i]
        return s

    def HashToROOT(self, str key) -> str:
        return self.ptr.HashToROOT(key.encode("UTF-8")).decode("UTF-8")

    @property
    def clone(self):
        v = self.__new__(self.__class__)
        v.__init__()
        return v
    
    @property
    def Threads(self) -> int: return self.ptr.Threads

    @property
    def chnk(self) -> int: return self.ptr.ChunkSize

    @Threads.setter
    def Threads(self, int val): self.ptr.Threads = val

    @chnk.setter
    def chnk(self, int val): self.ptr.ChunkSize = val
    
    @property
    def OutputDirectory(self):
        return self.OutDir
    
    @OutputDirectory.setter
    def OutputDirectory(self, str val):
        self.OutDir = val
    
    @property
    def EventCache(self) -> bool:
        return self._EventCache 

    @EventCache.setter
    def EventCache(self, val) -> bool:
        self._EventCache = val

    @property
    def DataCache(self) -> bool:
        return self._DataCache 

    @DataCache.setter
    def DataCache(self, val) -> bool:
        self._DataCache = val
    
    @property
    def SampleName(self) -> str:
        return self._SampleName 

    @SampleName.setter
    def SampleName(self, val) -> str:
        self._SampleName = val

    def FastHashSearch(self, hashes) -> dict:
        cdef vector[string] v
        cdef int i
        cdef string key
        cdef bool fnd

        for i in range(len(hashes)): v.push_back(hashes[i].encode("UTF-8"))
        return {key.decode("UTF-8") : fnd for key, fnd in self.ptr.FastSearch(v)}

    def AddEvent(self, Events, root, index):
        cdef int i; 
        for i in range(len(Events)):
            if Events[i].index == -1: Events[i].index = index
            Events[i].hash = root
            
            if self.ptr.ContainsHash(Events[i].hash.encode("UTF-8")): continue
            _ev = new CyEvent()
            _ev.Tree = Events[i].Tree.encode("UTF-8")
            _ev.hash = Events[i].hash.encode("UTF-8")
            _ev.ROOT = root.encode("UTF-8")
            _ev.EventIndex = <unsigned int>Events[i].index
            _ev.Event = True
            
            self.ptr.AddEvent(_ev) 
            self.HashMap[Events[i].hash] = Events[i]
    
    def AddGraph(self, Events):
        cdef int i;
        cdef str hash; 
        cdef CyEvent* event;
        for i in range(len(Events)):
            hash, evnt = Events[i]
            self.HashMap[hash] = evnt 
            event = self.ptr.HashToEvent(<string>hash.encode("UTF-8"))
            event.Graph = True 
            event.Event = False
    
    @property
    def DumpTracer(self):
        cdef pair[string, CyROOT*] root
        cdef pair[string, CyEvent*] event; 
        cdef CyROOT* r
        cdef CyEvent* e
       
        try: os.mkdir(self.OutDir + "/Tracer/")
        except FileExistsError: pass

        for root in self.ptr._ROOTMap:
            r = root.second
            f = h5py.File(self.OutDir + "/Tracer/" + r.CachePath.decode("UTF-8") + ".hdf5", "a")
            try: ref = f.create_dataset("SampleNames", (1), dtype = h5py.ref_dtype)
            except ValueError: ref = f["SampleNames"]
            if self._SampleName != "": ref.attrs[self._SampleName] = True

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
        cdef CyEvent* e
        cdef str out
        cdef str file = ""
        
        _, bar = self._MakeBar(len(self), "DUMPING EVENTS")
        for i in self:
            e = self.ptr.HashToEvent(<string>i.hash.encode("UTF-8"))

            if e.CachedEvent and i.Event: continue
            elif e.CachedGraph and i.Graph: continue
            
            bar.update(1)
            if i.Event: out = self.OutDir + "/EventCache/"
            elif i.Graph: out = self.OutDir + "/DataCache/"
            else: continue
            
            if file == "":
                try: os.mkdir(out)
                except FileExistsError: pass

            out += i.CachePath
            if file != out: f = h5py.File(out + ".hdf5", "a")
             
            try: ref = f.create_dataset(i.hash, (1), dtype = h5py.ref_dtype)
            except ValueError: ref = f[i.hash]

            if i.Event: e.CachedEvent = True
            else: e.CachedGraph = True
            ref.attrs["Event"] = codecs.encode(pickle.dumps(i), "base64").decode() 

            file = out if file != "" else file
            if file != out: f.close()
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

            maps = self.FastHashSearch([k for k in f if k != "SampleNames"])
            get = [i for i in maps if not maps[i]]
            if len(get) == 0: continue
            for i in get:
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
    def RestoreEvents(self):
        self.RestoreTracer 
        cdef pair[string, CyROOT*] c
        cdef pair[string, CyEvent*] e
        cdef CyROOT* r_
        cdef CyEvent* e_
        cdef str get
        cdef bool DataCache, EventCache

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
            get = "RESTORING EVENTS (" + ("DataCache" if DataCache else "EventCache") + "): "
            _, bar = self._MakeBar(r_.HashMap.size(), get + c.first.decode("UTF-8"))
            for e in r_.HashMap:
                bar.update(1)
                e_ = e.second   
                if e_.CachedEvent == False and e_.CachedGraph == False: continue
                elif e_.CachedGraph == False and DataCache: continue
                elif e_.CachedEvent == False and EventCache: continue
                get = e.first.decode("UTF-8")
                self.HashMap[get] = pickle.loads(codecs.decode(f[get].attrs["Event"].encode("UTF-8"), "base64"))
                if EventCache: e_.Event = True
                if DataCache: e_.Graph = True
            del bar

    def _MakeBar(self, inpt: Union[int], CustTitle: Union[None, str] = None):
        _dct = {}
        _dct["desc"] = f'Progress {self.Caller}' if CustTitle is None else CustTitle
        _dct["leave"] = False
        _dct["colour"] = "GREEN"
        _dct["dynamic_ncols"] = True
        _dct["total"] = inpt
        return (None, tqdm(**_dct))

    def __preiteration__(self):
        pass

    def __iter__(self):
        if self.__preiteration__(): 
            self._its = 0
            self._ite = 0
            return self

        self._itv = self.ptr.HashList()
        self._its = 0
        self._ite = self._itv.size()
        return self

    def __next__(self):
        if self._its == self._ite: raise StopIteration
        cdef str hash = self._itv[self._its].decode("UTF-8")
        ev = Event()
        ev._wrap(self.HashMap[hash])
        ev.ptr = self.ptr[0].HashToEvent(<string>hash.encode("UTF-8"))
        self._its += 1
        return ev
        
