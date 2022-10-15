from AnalysisTopGNN.Generators import EventGenerator, GraphGenerator
from AnalysisTopGNN.IO import UnpickleObject, PickleObject
from AnalysisTopGNN.Events import Event, EventGraphTruthTopChildren
from AnalysisTopGNN.Samples import SampleTracer

def TestEventGenerator(Files):
    File1 = Files[0]
    File2 = Files[1] 

    Ev = EventGenerator(File1) 
    Ev.Event = Event
    Ev.SpawnEvents()
    Ev.CompileEvent()
    PickleObject(Ev, "TMP1")

    T = EventGenerator(File2) 
    T.Event = Event
    T.SpawnEvents()
    T.CompileEvent()
    PickleObject(T, "TMP2")

    T = EventGenerator(Files) 
    T.Event = Event
    T.SpawnEvents()
    T.CompileEvent()
    PickleObject(T, "TMP3")

    ev1 = UnpickleObject("TMP1")
    ev2 = UnpickleObject("TMP2")
    ev3 = UnpickleObject("TMP3")

    x = SampleTracer(ev1)
    y = SampleTracer(ev2)
    z = SampleTracer(ev3) 

    p = sum([x, y])
    
    assert len(p) == len(z)
    print("PASSED: SAME LENGTH", len(p), len(z))

    for i, j in zip(p, z):
        assert i.EventIndex == j.EventIndex
    print("PASSED: CONSISTENT INDEX")

    for i, j in zip(p, z):
        assert i.Filename == j.Filename
    print("PASSED: SAME FILENAMES")
    
    for i, j in zip(p, z):
        assert len(i.Trees["nominal"].DetectorParticles) == len(j.Trees["nominal"].DetectorParticles)
    print("PASSED: SAME NUMBER OF PARTICLES PER EVENT")

    ev1 = UnpickleObject("TMP1")
    ev2 = UnpickleObject("TMP2")
    ev3 = UnpickleObject("TMP3")

    x = SampleTracer(ev1)
    y = SampleTracer(ev2)
    z = SampleTracer(ev3) 

    p = sum([x, y, SampleTracer(ev3)])
    
    assert len(p) == len(z)
    print("PASSED: SAME LENGTH", len(p), len(z))

    for i, j in zip(p, z):
        assert i.EventIndex == j.EventIndex
    print("PASSED: CONSISTENT INDEX")

    for i, j in zip(p, z):
        assert i.Filename == j.Filename
    print("PASSED: SAME FILENAMES")
    
    for i, j in zip(p, z):
        assert len(i.Trees["nominal"].DetectorParticles) == len(j.Trees["nominal"].DetectorParticles)
    print("PASSED: SAME NUMBER OF PARTICLES PER EVENT")

    return True 


def TestGraphGenerator(Files):
    from AnalysisTopGNN.Generators import GraphGenerator
    from AnalysisTopGNN.Events import EventGraphTruthTopChildren

    def Test(a):
        return a.eta

    File1 = Files[0]
    File2 = Files[1] 

    Ev = EventGenerator(File1) 
    Ev.Event = Event
    Ev.SpawnEvents()
    Ev.CompileEvent()
    
    Gr = GraphGenerator()
    Gr.ImportTracer(Ev) 
    Gr.AddNodeFeature(Test)
    Gr.EventGraph = EventGraphTruthTopChildren
    Gr.CompileEventGraph()
    PickleObject(Gr, "TMP1")

    T = EventGenerator(File2) 
    T.Event = Event
    T.SpawnEvents()
    T.CompileEvent()

    Gr = GraphGenerator()
    Gr.ImportTracer(T) 
    Gr.AddNodeFeature(Test)
    Gr.EventGraph = EventGraphTruthTopChildren
    Gr.CompileEventGraph()
    PickleObject(Gr, "TMP2")

    T = EventGenerator(Files) 
    T.Event = Event
    T.SpawnEvents()
    T.CompileEvent()

    Gr = GraphGenerator()
    Gr.ImportTracer(T) 
    Gr.AddNodeFeature(Test)
    Gr.EventGraph = EventGraphTruthTopChildren
    Gr.CompileEventGraph()
    PickleObject(Gr, "TMP3")

    ev1 = UnpickleObject("TMP1")
    ev2 = UnpickleObject("TMP2")
    ev3 = UnpickleObject("TMP3")

    x = SampleTracer(ev1)
    y = SampleTracer(ev2)
    z = SampleTracer(ev3) 

    p = sum([x, y])
    
    assert len(p) == len(z)
    print("PASSED: SAME LENGTH", len(p), len(z))

    for i, j in zip(p, z):
        assert i.EventIndex == j.EventIndex
    print("PASSED: CONSISTENT INDEX")

    for i, j in zip(p, z):
        assert i.Filename == j.Filename
    print("PASSED: SAME FILENAMES")
    
    for i, j in zip(p, z):
        assert i.Trees["nominal"].num_nodes == j.Trees["nominal"].num_nodes
    print("PASSED: SAME NUMBER OF PARTICLES PER EVENT")

    ev1 = UnpickleObject("TMP1")
    ev2 = UnpickleObject("TMP2")
    ev3 = UnpickleObject("TMP3")

    x = SampleTracer(ev1)
    y = SampleTracer(ev2)
    z = SampleTracer(ev3) 

    p = sum([x, y, SampleTracer(ev3)])
    
    assert len(p) == len(z)
    print("PASSED: SAME LENGTH", len(p), len(z))

    for i, j in zip(p, z):
        assert i.EventIndex == j.EventIndex
    print("PASSED: CONSISTENT INDEX")

    for i, j in zip(p, z):
        assert i.Filename == j.Filename
    print("PASSED: SAME FILENAMES")
    
    for i, j in zip(p, z):
        assert i.Trees["nominal"].num_nodes == j.Trees["nominal"].num_nodes
    print("PASSED: SAME NUMBER OF PARTICLES PER EVENT")

    return True 




import torch 
import h5py
import numpy as np


import hashlib 

class Hashing:

    def __init__(self):
        pass
    
    def MD5(self, inpt):
        return str(hashlib.md5(inpt.encode("utf-8")).hexdigest())


from AnalysisTopGNN.Tools import Tools
from AnalysisTopGNN.Tools import Threading

class HDF5(Tools):

    def __init__(self):
        self._File = None
        self.Filename = "UNTITLED"
        self._ext = ".hdf5"
        self._iter = -1
        self._obj = {}
        self.Threads = 12
        self.chnk = 100
    
    def Start(self, Name = False, Mode = "w"):
        self._File = h5py.File(self.Filename + self._ext, mode = Mode, track_order = True)
        if Mode == "w":
            self.__IncrementRefSet(Name)
    
    def __IncrementRefSet(self, Name = False):
        self._iter += 1
        if Name:
            name = Name 
        else:
            name = self._iter
        if str(name) in self._File:
            return 
        self._Ref = self._File.create_dataset(str(name), (1, ), dtype = h5py.ref_dtype)

    def __AddToDataSet(self, RefName, Key, Val = None):
        if Val == None:
            self._Ref.attrs[RefName] = Key
            return  
        if "AnalysisTopGNN" in str(type(Val)).split("'")[1]:
            self.DumpObject(Val)
            Val = str(hex(id(Val)))
        self._Ref.attrs[RefName + "." + Key] = Val

    def __Contains(self, key):
        return True if key in self._Ref.attrs else False

    def __Store(self, ObjPath, objectaddress, Key, Val):
            
            if self.__Contains(objectaddress) == False:
                self.__AddToDataSet(objectaddress, ObjPath)
            
            if isinstance(Val, str):
                return self.__AddToDataSet(objectaddress, Key, Val)
            elif isinstance(Val, int):
                return self.__AddToDataSet(objectaddress, Key, Val)
            elif isinstance(Val, float):
                return self.__AddToDataSet(objectaddress, Key, Val)
            elif isinstance(Val, dict):
                for i in Val:
                    self.__AddToDataSet(objectaddress, Key + "-" + i, Val[i])
                return 
            elif isinstance(Val, list):
                for i in range(len(Val)):
                    self.__AddToDataSet(objectaddress, Key + "#" + str(i), Val[i]) 
                return 
            print(ObjPath, objectaddress, Key, Val, type(Val))

    def DumpObject(self, obj, Name = False):
        if self._iter == -1:
            self.Start(Name = Name, Mode = "w")
        if Name:
            self.__IncrementRefSet(Name)

        objname = str(type(obj)).split("'")[1]
        objectaddress = str(hex(id(obj)))
        for i in obj.__dict__:
            self.__Store(objname, objectaddress, i, obj.__dict__[i])
        return True
    
    def MultiThreadedDump(self, ObjectDict):
        if isinstance(ObjectDict, dict) == False:
            return 
       
        self.mkdir("_TMP")
        def function(inpt):
            out = []
            for i in inpt:
                h = HDF5()
                h.Filename = "_TMP/" + i[0] 
                h.DumpObject(i[1])
                out.append(["_TMP/" + i[0] + self._ext, i[0]]) 
            return out

        inpo = [[name, ObjectDict[name]] for name in ObjectDict]
        TH = Threading(inpo, function, self.Threads, self.chnk)
        TH.VerboseLevel = 3
        TH.Start()
        with h5py.File(self.Filename + self._ext, "w") as dst:
            for i in enumerate(TH.lists):
                with h5py.File(i[0]) as src:
                    dst[i[1]] = src[i[1]]

    def End(self):
        self._File.close()
        self.__init__()
    
    def __BuildContainer(self, obj, attr, i, Name, typ):
        if typ == "-":
            ins = {}
        if typ == "#":
            ins = []
        attr = attr.split(typ)
        r = self._Ref.attrs[i]
        val = {attr[1] : r if r not in self._obj[Name] else self._obj[Name][r][1]}

        if attr[0] not in obj.__dict__:
            setattr(obj, attr[0], ins)
        elif isinstance(obj.__dict__[attr[0]], type(ins)) == False:
            setattr(obj, attr[0], ins)
        v = getattr(obj, attr[0])
        if typ == "-":
            v |= val
        if typ == "#":
            v += val
    
    def RebuildObject(self, Name):
        if self._iter == -1:
            self.Start(Mode = "r")
            self._Ref = self._File[Name]
            self._iter = 0
            self._obj[Name] = {}
            objstruc = {n : self._Ref.attrs[n] for n in self._Ref.attrs}
            self._obj[Name] |= {n : self.GetObjectFromString(".".join(objstruc[n].split(".")[:-1]), objstruc[n].split(".")[-1]) for n in objstruc if "." not in n}
            return self.RebuildObject(Name)
        
        for i in self._Ref.attrs:
            val = i.split(".")

            # --- the variable "de" is the default value for an object. See if this causes a problem.
            de, obj = self._obj[Name][val[0]]
            if len(val) == 1:
                continue

            attr = val[1]
            if "-" in attr:
                self.__BuildContainer(obj, attr, i, Name, "-")
                continue
            elif "#" in attr:
                self.__BuildContainer(obj, attr, i, Name, "#")
                continue
            setattr(obj, attr, self._Ref.attrs[i])
        return list(self._obj[Name].values())[0][1]

    def __iter__(self):
        self.Start(Mode = "r")
        self._names = [n for i in self._F]
        return self
    def __next__(self):
        self._iter = -1
        if len(self._names) == 0:
            raise StopIteration()
        name = self._names.pop()
        return (name, self.RebuildObject(name))

def TestEventGeneratorDumper(Files):

    #File1 = Files[0]

    #Ev = EventGenerator(File1) 
    #Ev.Event = Event
    #Ev.SpawnEvents()
    #Ev.CompileEvent()
    #PickleObject(Ev, "TMP1")

    ev1 = UnpickleObject("TMP1")
    hdf = HDF5()
    Objects = {}
    for i in ev1:
        Objects[i.Filename] = i
    hdf.MultiThreadedDump(Objects)


    #hdf.DumpObject(i, i.Filename)
    
    #    out = hdf.RebuildObject(i.Filename) 
    #    c = out + i 
    #    for k in c:
    #        print(k)
        

