from AnalysisG.Templates import ApplyFeatures
from AnalysisG.Submission import Condor
from AnalysisG.Events import GraphChildren
from AnalysisG.Events import Event
from AnalysisG import Analysis

from examples.ExampleSelection import Example, Example2
from examples.EventOther import EventOther
from models.CheatModel import CheatModel
from examples.Graph import DataGraph
from examples.Event import EventEx

smpl = "./samples/"
Files = {
    smpl + "sample1": ["smpl1.root"],
    smpl + "sample2": ["smpl1.root", "smpl2.root", "smpl3.root"],
}

def _template(default=True):
    AnaE = Analysis()
    AnaE.ProjectName = "Project"
    if default == True:
        AnaE.InputSample("Sample1", smpl + "sample1/" + Files[smpl + "sample1"][0])
        AnaE.InputSample("Sample2", smpl + "sample2/" + Files[smpl + "sample2"][1])
    else:
        AnaE.InputSample(**default)
    AnaE.Threads = 2
    AnaE.Verbose = 1
    return AnaE

def test_condor():
    con = Condor()
    con.OutputDirectory = "Project"
    con.ProjectName = "Project_Condor"
    con.PythonVenv = "$PythonGNN"
    con.Verbose = 1

    ana = _template()
    ana.Event = EventEx
    con.AddJob("smpl1", ana, waitfor = ["smpl3", "smpl2"])
    con.AddJob("smpl2", ana, waitfor = ["smpl3"])
    con.AddJob("smpl3", ana)

    Ana_s1 = _template({"Name": "smpl1"})
    Ana_s1.Event = EventEx
    con.AddJob("example1_1", Ana_s1, waitfor=["smpl1", "smpl2", "smpl3", "example1_2"])

    Ana_s2 = _template({"Name": "smpl2"})
    Ana_s2.Event = EventEx
    con.AddJob("example1_2", Ana_s2, waitfor=["smpl3"])

    Ana_s3 = _template({"Name": "smpl3"})
    Ana_s3.Event = EventEx
    con.AddJob("example1_3", Ana_s3, waitfor=["smpl1", "smpl2", "smpl3", "example1_1", "example1_2"])

    Ana_sum = _template()
    Ana_sum.Event = EventEx
    con.AddJob("sum", Ana_sum, waitfor=["example1_1", "example1_2", "example1_3"])

    con.LocalRun()
    con.rm("Project")

def test_dumping_events():
    con = Condor()
    con.PythonVenv = "$PythonGNN"
    con.ProjectName = "Project"


    Ana = _template()
    Ana.Event = EventEx()
    Ana.EventCache = True
    con.AddJob("example", Ana)
    con.LocalRun()

    x = []
    Ana2 = _template()
    Ana2.EventCache = True
    Ana2.EventName = "EventEx"
    Ana2.Launch()
    for i in Ana2: x.append(i.hash)
    assert len(x) != 0

    Ana2.rm("Project")


def Feat(a):
    return 1

def test_dumping_graphs():

    con = Condor()
    con.PythonVenv = "$PythonGNN"
    con.ProjectName = "Project"

    Ana = _template()
    Ana.AddGraphFeature(Feat)
    Ana.Graph = DataGraph
    Ana.EventName = "EventEx"
    Ana.EventCache = True
    Ana.DataCache = True
    con.AddJob("Data", Ana, waitfor=["Events"])

    Ana = _template()
    Ana.Event = EventEx
    Ana.EventCache = True
    con.AddJob("Events", Ana)
    con.LocalRun()

    print("____")
    Ana = _template()
    Ana.DataCache = True
    Ana.GraphName = "DataGraph"
    Ana.Launch()
    x = []
    for i in Ana:
        if not i.Graph: continue
        assert i.GraphName == "DataGraph"
        x.append(i.hash)
        assert i.G_Feat
        assert not len(i.Errors)
    assert len(x) != 0
    Ana.rm("Project")


def test_dumping_event_selection():
    con = Condor()
    con.PythonVenv = "$PythonGNN"
    con.ProjectName = "Project"

    Ana_1 = _template(
        {"Name": "smpl1", "SampleDirectory": {smpl + "sample2": ["smpl1.root"]}}
    )
    Ana_1.Event = EventEx
    Ana_1.EventCache = True

    Ana_2 = _template(
        {"Name": "smpl2", "SampleDirectory": {smpl + "sample2": ["smpl2.root"]}}
    )
    Ana_2.Event = EventEx
    Ana_2.EventCache = True

    Ana_3 = _template(
        {"Name": "smpl3", "SampleDirectory": {smpl + "sample2": ["smpl3.root"]}}
    )
    Ana_3.Event = EventEx
    Ana_3.EventCache = True

    con.AddJob("smpl1", Ana_1)
    con.AddJob("smpl2", Ana_2)
    con.AddJob("smpl3", Ana_3)

    Ana_s1 = _template({"Name": "smpl1"})
    Ana_s1.AddSelection(Example)
    Ana_s1.EventName = "EventEx"
    Ana_s1.EventCache = True
    con.AddJob("example1_1", Ana_s1, waitfor=["smpl1", "smpl2", "smpl3"])

    Ana_s2 = _template({"Name": "smpl2"})
    Ana_s2.AddSelection(Example)
    Ana_s2.EventName = "EventEx"
    Ana_s2.EventCache = True
    con.AddJob("example1_2", Ana_s2, waitfor=["smpl1", "smpl2", "smpl3"])

    Ana_s3 = _template({"Name": "smpl3"})
    Ana_s3.AddSelection(Example)
    Ana_s3.EventName = "EventEx"
    Ana_s3.EventCache = True
    con.AddJob("example1_3", Ana_s3, waitfor=["smpl1", "smpl2", "smpl3"])

    con.LocalRun()

    Ana_T = _template({"Name" : None})
    Ana_T.ProjectName = "Project"
    Ana_T.SelectionName = "Example"
    Ana_T.Threads = 1

    x = []
    for i in Ana_T:
        if not i.release_selection(): continue
        assert i.selection
        assert i.sample_name
        x.append(i)
    assert len(x)
    Ana_T.rm("Project")

def test_dumping_optimization():
    con = Condor()
    con.PythonVenv = "$PythonGNN"
    con.ProjectName = "Project"

    Ana_1 = _template(
        {"Name": "smpl1", "SampleDirectory": {smpl + "sample2": ["smpl1.root"]}}
    )
    Ana_1.Event = Event
    Ana_1.EventCache = True

    Ana_2 = _template(
        {"Name": "smpl2", "SampleDirectory": {smpl + "sample2": ["smpl2.root"]}}
    )
    Ana_2.Event = Event
    Ana_2.EventCache = True

    Ana_3 = _template(
        {"Name": "smpl3", "SampleDirectory": {smpl + "sample2": ["smpl3.root"]}}
    )
    Ana_3.Event = Event
    Ana_3.EventCache = True

    con.AddJob("smpl1", Ana_1)
    con.AddJob("smpl2", Ana_2)
    con.AddJob("smpl3", Ana_3)

    AnaD_1 = _template({"Name": "smpl1"})
    ApplyFeatures(AnaD_1, "TruthChildren")
    AnaD_1.DataCache = True
    AnaD_1.EventCache = True
    AnaD_1.Tree = "nominal"
    AnaD_1.EventName = "Event"
    AnaD_1.Graph = GraphChildren

    AnaD_2 = _template({"Name": "smpl2"})
    ApplyFeatures(AnaD_2, "TruthChildren")
    AnaD_2.DataCache = True
    AnaD_2.EventCache = True
    AnaD_2.Tree = "nominal"
    AnaD_2.EventName = "Event"
    AnaD_2.Graph = GraphChildren

    AnaD_3 = _template({"Name": "smpl3"})
    ApplyFeatures(AnaD_3, "TruthChildren")
    AnaD_3.DataCache = True
    AnaD_3.EventCache = True
    AnaD_3.Tree = "nominal"
    AnaD_3.EventName = "Event"
    AnaD_3.Graph = GraphChildren

    con.AddJob("Dsmpl1", AnaD_1, waitfor=["smpl1"])
    con.AddJob("Dsmpl2", AnaD_2, waitfor=["smpl2"])
    con.AddJob("Dsmpl3", AnaD_3, waitfor=["smpl3"])

    AnaOp = _template({"Name": "smpl1"})
    AnaOp.InputSample(**{"Name": "smpl2"})
    AnaOp.InputSample(**{"Name": "smpl3"})
    AnaOp.RunName = "run-1"
    AnaOp.GraphName = "GraphChildren"
    AnaOp.Tree = "nominal"
    AnaOp.DataCache = True
    AnaOp.kFolds = 4
    AnaOp.Epochs = 10
    AnaOp.Optimizer = "ADAM"
    AnaOp.Scheduler = "ExponentialLR"
    AnaOp.OptimizerParams = {"lr": 0.001}
    AnaOp.SchedulerParams = {"gamma": 1}
    AnaOp.Device = "cuda"
    AnaOp.BatchSize = 2
    AnaOp.ModelParams = {"test" : "t"}
    AnaOp.Model = CheatModel
    AnaOp.DebugMode = True

    con.AddJob("run-1", AnaOp, waitfor=["Dsmpl1", "Dsmpl2", "Dsmpl3"])
    con.LocalRun()
    con.rm("Project")


if __name__ == "__main__":
    #test_condor()
    #test_dumping_events()
    #test_dumping_graphs()
    #test_dumping_event_selection()
    test_dumping_optimization()
