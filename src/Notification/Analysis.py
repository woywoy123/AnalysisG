from .Notification import Notification
from time import sleep

class _Analysis(Notification):
    
    def __init__(self):
        pass
    
    @property
    def _WarningPurge(self):
        self.Warning("'PurgeCache' enabled! You have 10 seconds to cancel.")
        self.Warning("Directory (DataCache/EventCache): " + self.OutputDirectory)
        _, bar = self._MakeBar(10, "PURGE-TIMER")
        for i in range(10): 
            sleep(1)
            bar.update(1)
        self.rm(self.OutputDirectory + "/EventCache") 
        self.rm(self.OutputDirectory + "/DataCache")         
        self.rm(self.OutputDirectory + "/Tracer")

    @property
    def _BuildingCache(self):
        if self.EventCache: 
            self.mkdir(self.OutputDirectory + "/EventCache")
            self.Success("Created EventCache under: " + self.OutputDirectory)
        if self.DataCache:  
            self.mkdir(self.OutputDirectory + "/DataCache")
            self.Success("Created DataCache under: " + self.OutputDirectory)

    @property
    def _CheckForTracer(self):
        f = self.ls(self.OutputDirectory + "/Tracer/")
        if len(f) == 0 and (self.EventCache or self.DataCache): return not self.Warning("Tracer directory not found. Generating")

        tracers = {i.split("-")[0] : i.split("-")[1].replace(".hdf5", ".root") for i in f}
        matched = {}
        for i in self.SampleMap:
            matched |= {self.Hash(t + "/") : [t + "/" + l for l in self.SampleMap[i][t]] for t in self.SampleMap[i]}
        self.WhiteSpace()
        if len(tracers) > 0: self.Success("Tracer directory found: ")
        for i in tracers:
            try: f = matched[i]
            except KeyError: 
                self.Success("-> " + i + "-" + tracers[i].replace(".root", ".hdf5"))
                continue
            x = [t for t in f if t.endswith(tracers[i])]
            if len(x) == 0: continue
            pth = "/".join(x[0].split("/")[:-1])
            self.Success("!-> Path: " + pth)
            for l in x: self.Success("!-> " + l.replace(pth, ""))
        return True

    @property
    def EmptySampleList(self):
        if len(self) != 0: return False
        string = "No samples found in cache. Exiting..."
        self.Failure("="*len(string))
        self.Failure(string)
        self.Failure("="*len(string))
        return True

    @property
    def StartingAnalysis(self):
        string1 = "---" + " Starting Project: " + self.ProjectName + " ---"
        string = ""
        string += " > EventGenerator < :: " if self.Event != None else ("> EventCache < :: " if self.EventCache else "")
        string += " > GraphGenerator < :: " if self.EventGraph != None else ("> DataCache < :: " if self.DataCache else "")
#        string += "> TrainingSampleGenerator < :: " if self.TrainingSampleName else ""
#        string += "> Optimization < :: " if self.Model != None else ""
#        string += "> ModelEvaluator < :: " if len(self._ModelDirectories) != 0 or self.PlotNodeStatistics else ""
        string += " > Selections < :: " if len(self.Selections) != 0 else ""
        string += " > Merging Selections < :: " if len(self.Merge) != 0 else ""
        
        l = len(string) if len(string1) < len(string) else len(string1)
        self.Success("="*l)
        self.Success(string1)
        self.Success(string)
        self.Success("="*l)


