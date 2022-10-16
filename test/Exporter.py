from AnalysisTopGNN.Generators import EventGenerator, GraphGenerator
from AnalysisTopGNN.IO import UnpickleObject, PickleObject
from AnalysisTopGNN.Events import Event, EventGraphTruthTopChildren
from AnalysisTopGNN.Samples import SampleTracer
from AnalysisTopGNN.IO import HDF5

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


def TestEventGeneratorDumper(Files):

    File1 = Files[0]

    Ev = EventGenerator(File1) 
    Ev.Event = Event
    Ev.SpawnEvents()
    Ev.CompileEvent()
    hdf = HDF5()

    Objects = {}
    it = 0
    for i in Ev:
        Objects[i.Filename] = i
        it+=1 
        if it == 24:
            break
    hdf.MultiThreadedDump(Objects)
    for name, obj in hdf:
        print(name, obj.Trees["nominal"].TruthTops, obj)
    return True

def TestGraphGeneratorDumper(Files):
    from AnalysisTopGNN.Generators import GraphGenerator
    from AnalysisTopGNN.Events import EventGraphTruthTopChildren

    def Test(a):
        return a.eta

    File1 = Files[0]

    Ev = EventGenerator(File1) 
    Ev.Event = Event
    Ev.SpawnEvents()
    Ev.CompileEvent()
    
    Gr = GraphGenerator()
    Gr.ImportTracer(Ev) 
    Gr.AddNodeFeature(Test)
    Gr.EventGraph = EventGraphTruthTopChildren
    Gr.CompileEventGraph()

    hdf = HDF5()
    Objects = {}
    it = 0
    for i in Gr:
        Objects[i.Filename] = i
        it+=1 
        if it == 24:
            break
    hdf.MultiThreadedDump(Objects)
    
    for name, obj in hdf:
        print(name, obj.Trees["nominal"], obj)
    return True
