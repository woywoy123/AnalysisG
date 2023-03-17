from AnalysisTopGNN.Generators import EventGenerator, GraphGenerator, Analysis
from AnalysisTopGNN.Events import Event, EventGraphChildren
from ExampleSelection import Example, Example2

smpl = "./TestCaseFiles/Sample/"
Files = {smpl + "tttt" : ["output.root"], smpl + "tttt_m400" : ["smpl1.root", "smpl2.root", "smpl3.root"]}
def test_eventgenerator():
    EvtGen = EventGenerator(Files)
    EvtGen.EventStop = 3
    EvtGen.EventStart = 1
    EvtGen.Event = Event
    EvtGen.SpawnEvents()
   
    assert len(EvtGen) != 0
    for event in EvtGen:
        assert event == EvtGen[event.Filename]

def _fx(a):
    return 1

def test_eventgraph():
    Ev = EventGenerator(Files)
    Ev.Event = Event 
    Ev.EventStop = 100
    Ev.SpawnEvents()
    
    Gr = GraphGenerator()
    Gr += Ev
    Gr.EventGraph = EventGraphChildren
    Gr.AddGraphFeature(_fx)
    Gr.TestFeatures(len(Ev))
    Gr.CompileEventGraph()
   
    assert len(Gr) == len(Ev)
    for i in Gr:
        assert i.Compiled
        assert Gr.HashToROOT(i.Filename) != None

def test_merge_eventgenerator():
    f = list(Files)
    File0 = {f[0] : Files[f[0]]}
    File1 = {f[1] : Files[f[1]]}

    ev0 = EventGenerator(File0)
    ev0.Event = Event
    ev0.EventStart = 0
    ev0.EventStop = 10
    ev0.SpawnEvents()
    
    ev1 = EventGenerator(File1)
    ev1.Event = Event
    ev1.EventStart = 0
    ev1.EventStop = 10
    ev1.SpawnEvents()
  
    combined = EventGenerator(Files)
    combined.Event = Event
    combined.EventStart = 0
    combined.EventStop = 10
    combined.SpawnEvents()

    Object0 = {}
    for i in ev0:
        Object0[i.Filename] = i
    
    Object1 = {}
    for i in ev1:
        Object1[i.Filename] = i
    
    ObjectSum = {}
    for i in combined:
        ObjectSum[i.Filename] = i
    assert len(ObjectSum) == len(Object0) + len(Object1)
    
    combined += ev0
    combined += ev1
    assert len(combined) == len(ev0) + len(ev1)
    
    for i in Object0:
        assert ObjectSum[i] == Object0[i]
        print("Error: i", i, " obj", ObjectSum[i])

    for i in Object1:
        assert ObjectSum[i] == Object1[i]
        print("Error: i", i, " obj", ObjectSum[i])

    combined = combined + ev0 + ev1
    ObjectSum = {}
    for i in combined:
        ObjectSum[i.Filename] = i
    
    for i in Object0:
        assert ObjectSum[i] == Object0[i]
        print("Error: i", i, " obj", ObjectSum[i])

    for i in Object1:
        assert ObjectSum[i] == Object1[i]
        print("Error: i", i, " obj", ObjectSum[i])


    def EventGen(Dir, Name):
        Ana = Analysis()
        Ana.ProjectName = "TMPProject"
        Ana.InputSample(Name, Dir)
        Ana.EventCache = True
        Ana.Event = Event
        Ana.Threads = 4
        Ana.chnk = 4
        Ana.EventStart = 0
        Ana.EventStop = 10
        Ana.DumpHDF5 = True
        Ana.DumpPickle = True
        Ana.Launch()
        return Ana

    ev = EventGen(File0, "Tops")
    ev += EventGen(File1, "Top")
    
    for i in ev:
        assert ev[i.Filename].Filename == i.Filename
    ev.rm("TMPProject")

def test_analysis():
    AnaE = Analysis()
    AnaE.ProjectName = "Project"
    AnaE.InputSample("bsm-4t", smpl + Files[smpl + "tttt_m400"][0])
    AnaE.InputSample("t", smpl + Files[smpl + "tttt_m400"][1])
    AnaE.Event = Event
    AnaE.EventCache = True
    AnaE.Threads = 2
    AnaE.EventStop = 100
    AnaE.DumpPickle = True
    AnaE.Launch()
    
    it = 0
    for i in AnaE:
        it += 1
    assert it == len(AnaE)

    AnaE.rm("Project")

#def test_selection():
#    AnaE = Analysis()
#    AnaE.ProjectName = "Project"
#    AnaE.InputSample("bsm-4t", smpl + Files[smpl + "tttt_m400"][0])
#    AnaE.InputSample("t", smpl + Files[smpl + "tttt_m400"][1])
#    AnaE.AddSelection("Example", Example)
#    AnaE.AddSelection("Example2", Example2())
#    AnaE.MergeSelection("Example")
#    AnaE.MergeSelection("Example2")
#    AnaE.Event = Event
#    AnaE.EventCache = True
#    AnaE.DumpPickle = True
#    AnaE.Threads = 2
#    AnaE.VerboseLevel = 1
#    AnaE.Launch()
#    
#    it = 0
#    c = 0
#    y = []
#    for i in AnaE:
#        c += len(i.Trees["nominal"].TopChildren)
#        t = Example2()
#        t._EventPreprocessing(i)
#        y.append(t)
#        if it == 10:
#            break
#        it += 1
#    l = len(y) 
#    x = sum(y)
#    assert l == len(x._TimeStats)
#    assert l == x._CutFlow["Success->Example"]
#    assert l*4 == len(x.Top["Truth"])
#    assert len(x.Children["Truth"]) == c
#    AnaE.rm("Project")
#
