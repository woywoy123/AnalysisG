from AnalysisG import Analysis 
from AnalysisG.Submission import Condor
from examples.Event import EventEx
from examples.Graph import DataGraph
from examples.ExampleSelection import Example, Example2

smpl = "./samples/"
Files = {
            smpl + "sample1" : ["smpl1.root"], 
            smpl + "sample2" : ["smpl1.root", "smpl2.root", "smpl3.root"]
}

def _template(default = True):
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

def test_dumping_events():
    Ana = _template()
    Ana.Event = EventEx()   

    con = Condor()   
    con.PythonVenv = "$PythonGNN"
    con.ProjectName = "Project"
    con.AddJob("example", Ana)
    con.DumpCondorJobs 
    
    Ana.rm("./Project")


def Feat(a):
    return 1

def test_dumping_graphs():
    Ana = _template()
    Ana.EventGraph = DataGraph 
    Ana.AddGraphFeature(Feat)

    con = Condor()   
    con.PythonVenv = "$PythonGNN"
    con.ProjectName = "Project"
    con.AddJob("example", Ana)
    con.DumpCondorJobs 
    
    Ana.rm("./Project")
    
def test_dumping_event_graphs():
    Ana = _template()
    Ana.EventGraph = DataGraph 
    Ana.AddGraphFeature(Feat)

    Ana2 = _template()
    Ana2.Event = EventEx
    Ana2.EventCache = True

    con = Condor()   
    con.PythonVenv = "$PythonGNN"
    con.ProjectName = "Project"
    con.AddJob("example", Ana, waitfor = ["event"])
    con.AddJob("event", Ana2)
    con.DumpCondorJobs 
    Ana2.rm("./Project")
    
def test_dumping_event_selection():
    
    con = Condor()
    con.PythonVenv = "$PythonGNN"
    con.ProjectName = "Project"

    Ana_1 = _template({"Name" : "smpl1", "SampleDirectory" : {smpl + "sample2" : ["smpl1.root"]}})
    Ana_1.Event = EventEx 
    Ana_1.EventCache = True

    Ana_2 = _template({"Name" : "smpl2", "SampleDirectory" : {smpl + "sample2" : ["smpl2.root"]}})
    Ana_2.Event = EventEx 
    Ana_2.EventCache = True

    Ana_3 = _template({"Name" : "smpl3", "SampleDirectory" : {smpl + "sample2" : ["smpl3.root"]}})
    Ana_3.Event = EventEx 
    Ana_3.EventCache = True
     
    con.AddJob("smpl1", Ana_1)        
    con.AddJob("smpl2", Ana_2)        
    con.AddJob("smpl3", Ana_3)        

    Ana_s1 = _template({"Name" : "smpl1"})
    Ana_s1.AddSelection("example1", Example)
    con.AddJob("example1_1", Ana_s1, waitfor = ["smpl1", "smpl2", "smpl3"])    
   
    Ana_s2 = _template({"Name" : "smpl2"})
    Ana_s2.AddSelection("example1", Example)
    con.AddJob("example1_2", Ana_s2, waitfor = ["smpl1", "smpl2", "smpl3"])
 
    Ana_s3 = _template({"Name" : "smpl3"})
    Ana_s3.AddSelection("example1", Example)
    con.AddJob("example1_3", Ana_s3, waitfor = ["smpl1", "smpl2", "smpl3"])   
   
    Ana_sum = Analysis()
    Ana_sum.MergeSelection("example1")
    
    con.AddJob("sum", Ana_sum, waitfor = ["example1_1", "example1_2", "example1_3"])
    con.DumpCondorJobs
    con.TestCondorShell 

    Ana_T = _template({"Name" : "smpl1"})
    Ana_T.InputSample("smpl2")
    Ana_T.InputSample("smpl3")
    Ana_T.OutputDirectory = "./Project/CondorDump"
    Ana_T.Event = EventEx
    Ana_T.Launch

    from AnalysisG.IO import UnpickleObject
    x = UnpickleObject('./Project/CondorDump/Project/Selections/Merged/example1')
    assert list(x.CutFlow.values())[0] == len(Ana_T)
    con.rm("./Project")

if __name__ == "__main__":
    #test_dumping_events()
    #test_dumping_graphs()
    #test_dumping_event_graphs()
    test_dumping_event_selection()
    pass
