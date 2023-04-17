from AnalysisG.Generators import EventGenerator
from AnalysisG.Events.Events.Event import Event

def test_event_generator():

    root1 = "/home/tnom6927/Downloads/samples/Dilepton/output.root" 
    #root1 = "./samples/sample1/smpl1.root"

    EvtGen = EventGenerator({root1: []})
    EvtGen.EventStop = 3
    EvtGen.EventStart = 1
    EvtGen.Event = Event
    EvtGen.MakeEvents
    

if __name__ == "__main__":
    test_event_generator()
