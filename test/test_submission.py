from AnalysisG import Analysis 
from AnalysisG.Submission import Condor
from examples.Event import EventEx
from examples.Graph import DataGraph

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
    
 




if __name__ == "__main__":
    #test_dumping_events()
    #test_dumping_graphs()
    test_dumping_event_graphs()
