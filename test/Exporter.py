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
class HDF5(Tools):

    def __init__(self):
        self._File = None
        self.Filename = "UNTITLED"
        self._ext = ".hdf5"
        self._iter = -1
    
    def Start(self, Mode = "w"):
        self._File = h5py.File(self.Filename + self._ext, mode = Mode, track_order = True)
        self.__IncrementRefSet()
    
    def __IncrementRefSet(self):
        self._iter += 1
        self.RefSet = self._File.create_dataset(str(self._iter), (1, ), dtype = h5py.ref_dtype)

    def __CreateDataSet(self, RefName):
        self._Ref.attrs[RefName] = self._File.create_dataset(RefName, data = h5py.Empty(None)).ref

    def DumpObject(self, obj, Name):
        if self._iter == -1:
            self.Start()
        print(obj.__module__, type(obj).__qualname__)
        print(Name)





def TestEventGeneratorDumper(Files):

    #File1 = Files[0]

    #Ev = EventGenerator(File1) 
    #Ev.Event = Event
    #Ev.SpawnEvents()
    #Ev.CompileEvent()
    #PickleObject(Ev, "TMP1")

    ev1 = UnpickleObject("TMP1")
    hdf = HDF5()
    hdf.Start("w")
    for i in ev1:
        hdf.DumpObject(i, i.Filename)

        break
        

