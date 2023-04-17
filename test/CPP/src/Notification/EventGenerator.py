from .Notification import Notification 

class _EventGenerator(Notification):
    
    def __init__(self):
        pass

    @property
    def CheckEventImplementation(self):
        if self.Event == None:
            ex = "Or do: from AnalysisTopGNN.Events import Event"
            self.Failure("="*len(ex))
            self.Failure("No Event Implementation Provided.")
            self.Failure("var = " + self.Caller.capitalize() + "()")
            self.Failure("var.Event")
            self.Failure("See src/Events/Event.py or 'tutorial'")
            self.Failure("="*len(ex))
            return False
        return True
    
    @property
    def CheckROOTFiles(self):
        if len(self.MergeListsInDict(self.Files)) != 0: return True
        mes = "No .root files found."
        self.Failure("="*len(mes))
        self.Failure(mes)
        self.Failure("="*len(mes))
        return False

    @property
    def ObjectCollectFailure(self):
        mess = "Can't Collect Particle Objects in event.Objects..."
        self.Failure("="*len(mess))
        self.Failure(mess)
        self.Failure("="*len(mess))
        return self

