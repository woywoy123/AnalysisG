from .Notification import Notification

class EventGenerator(Notification):

    def __init__(self):
        pass
    
    def CheckEventImplementation(self):
        if self.Event == None:
            ex = "Or do: from AnalysisTopGNN.Events import Event"
            self.Failure("="*len(ex))
            self.Failure("No Event Implementation Provided.")
            self.Failure("See src/EventTemplates/Event.py")
            self.FailureExit(ex)
      
    def CheckROOTFiles(self):
        if len(self.MergeListsInDict(self.Files)) == 0:
            mes = "No .root files found."
            self.Failure("="*len(mes))
            self.FailureExit(mes)
