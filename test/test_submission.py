from AnalysisG import Analysis 
from AnalysisG.Submission import Condor
from AnalysisG.Events import Event, GraphChildren
from examples.Event import Event as EventEx

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

def test_dumping():
    Ana = _template()
    Ana.Event = EventEx()
    assert len(Ana.ExportAnalysisScript) != 0
 
    Ana = _template()
    Ana.Event = EventEx()   
    Ana._condor = True
    print(Ana.__Selection__)  

    con = Condor()   
    con.ProjectName = "Project"
    con.AddJob("example", Ana)
    con.DumpCondorJobs 


if __name__ == "__main__":
    test_dumping()
