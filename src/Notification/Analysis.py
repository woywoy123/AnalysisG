from .Notification import Notification

class Analysis(Notification):

    def __init__(self):
        pass

    def EmptyDirectoryWarning(self, Directory):
        pass
   
    def StartingAnalysis(self):
        string = "---" + " Starting Project: " + self.ProjectName + " ---"
        self.Success("="*len(string))
        self.Success(string)
        self.Success("="*len(string))

    def FoundCache(self, Directory, Files):
        if len(Files) == 0:
            self.Warning("No cache was found under " + Directory)
            return  
        self.Success("Found cache: ")
        self.Success("\n -> " + "\n -> ".join(Files))
    
    def MissingTracer(self, Directory):
        self.Failure("Tracer not found under: " + Directory + " skipping...")

    def NoEventImplementation(self):
        ex = "Or do: from AnalysisTopGNN.Events import Event"
        self.Failure("="*len(ex))
        self.Failure("No Event Implementation Provided.")
        self.Failure("See src/EventTemplates/Event.py")
    
    def NoEventGraphImplementation(self):
        message = "EventGraph not defined (obj.EventGraph). See implementations (See src/Events/EventGraphs.py)"
        self.Failure("="*len(message))


