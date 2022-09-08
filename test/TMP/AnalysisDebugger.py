from AnalysisRebuild import AnalysisNew
from DelphesDebug.DelphesEvent import Event
from DelphesDebug.DelphesEventGraph import EventGraphTruthTopChildren

def TestEventGenerator(File, Name):
    def Test(ev):
        return 0


    Ana = AnalysisNew()
    Ana.InputSample(Name, File)
    Ana.InputSample("ttbar", [File])

    Ana.EventCache = False
    Ana.DataCache = True
    Ana.Event = Event
    Ana.Tree = "Delphes"
    Ana.CPUThreads = 10
    Ana.EventEnd = 100
    Ana.ProjectName = "ExampleProject"
    Ana.EventGraph = EventGraphTruthTopChildren
    Ana.AddNodeFeature("Test", Test)
    Ana.Launch()



if __name__ == '__main__':
    Dir = "/CERN/Delphes/"
    
    TestEventGenerator(Dir, "Delphes")

