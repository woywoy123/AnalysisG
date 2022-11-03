from .Notification import Notification

class Analysis(Notification):

    def __init__(self):
        pass

    def EmptyDirectoryWarning(self, Directory):
        pass

    def EmptySampleList(self):
        if len(self) == 0:
            string = "No samples found in cache. Exiting..."
            self.Failure("="*len(string))
            self.FailureExit(string)
   
    def StartingAnalysis(self):
        string = "---" + " Starting Project: " + self.ProjectName + " ---"
        self.Success("="*len(string))
        self.Success(string)
        self.Success("="*len(string))

    def FoundCache(self, Directory, Files):
        if len(Files) == 0:
            self.Warning("No cache was found under " + Directory)
            return True
        return False
    
    def MissingTracer(self, Directory):
        self.Failure("Tracer not found under: " + Directory + " skipping...")

    def NoEventImplementation(self):
        ex = "Or do: from AnalysisTopGNN.Events import Event"
        self.Failure("="*len(ex))
        self.Failure("No Event Implementation Provided.")
        self.Failure("See src/EventTemplates/Event.py")
        self.Failure(ex)
        self.Failure("="*len(ex))
        self.FailureExit("Exiting...")
    
    def NoEventGraphImplementation(self):
        message = "EventGraph not defined (obj.EventGraph). See implementations (See src/Events/EventGraphs.py)"
        self.Failure("="*len(message))
    
    def FoundFiles(self, Files):
        if len(Files) == 0:
            return 
        
        trig = True 
        for i in self.DictToList(Files):
            if i.endswith(".root") and trig:
                string = "!!--- ADDING TO SAMPLE COLLECTION ---"
            if "DataCache" in i and trig:
                string = "!!--- FOUND DATA CACHE ---"
            if "EventCache" in i and trig:
                string = "!!--- FOUND EVENT CACHE ---"

            if trig:
                self.Success("-"*len(string))
                self.Success(string)
                self.Success("-"*len(string))
                trig = False
            self.Success("!!-> " + i)
    
    def CantGenerateTrainingSample(self):
        string = "Can't generate training sample, please choose either 'EventCache' or 'DataCache'"
        self.Failure("="*len(string))
        self.FailureExit(string)

    def FileNotFoundWarning(self, Directory, Name):
        pass

    def CheckPercentage(self):
        gr = False
        if self.TrainingPercentage > 100:
            self.TrainingPercentage = 80
            gr = "greater than 100%"
        if self.TrainingPercentage <= 0:
            self.TrainingPercentage = 80
            gr = "less than 0%"
        
        if gr:
            self.Warning("Specified Training Percentage " + gr + ". Setting to 80%")
