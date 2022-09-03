from AnalysisRebuild import AnalysisNew
from DelphesDebug.DelphesEvent import Event

def TestEventGenerator(File, Name):
    
    Ana = AnalysisNew()
    Ana.InputSample(Name, File)
    Ana.EventCache = True
    Ana.Event = Event
    Ana.Launch()



if __name__ == '__main__':
    Dir = "/home/tnom6927/Dokumente/Project/Analysis/bsm4tops-gnn-analysis/DebugGNN/AnalysisTopGNN/tag_1_delphes_events.root"
    
    TestEventGenerator(Dir, "Delphes")

