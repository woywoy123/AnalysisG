from .Notification import Notification
from time import sleep
import h5py


class _Analysis(Notification):
    def __init__(self):
        pass

    def _WarningPurge(self):
        self.Warning("'PurgeCache' enabled! You have 10 seconds to cancel.")
        self.Warning("Directory (DataCache/EventCache): " + self.OutputDirectory)
        _, bar = self._MakeBar(10, "PURGE-TIMER")
        for i in range(10):
            sleep(1)
            bar.update(1)
        self.rm(self.OutputDirectory + "/EventCache")
        self.rm(self.OutputDirectory + "/GraphCache")
        self.rm(self.OutputDirectory + "/Tracer")
        self.rm(self.OutputDirectory + "/Training/DataSets")

    def _BuildingCache(self):
        if self.EventCache:
            self.mkdir(self.WorkingPath + "EventCache")
            self.Success("Created EventCache under: " + self.WorkingPath)
        if self.DataCache:
            self.mkdir(self.WorkingPath + "GraphCache")
            self.Success("Created GraphCache under: " + self.WorkingPath)

    def _CheckForTracer(self):
        tracers = self.ListFilesInDir(self.WorkingPath + "Tracer/*", ".hdf5")
        if len(tracers) == 0 and (self.EventCache or self.DataCache):
            message = "Tracer directory not found. Generating"
            return not self.Warning(message)
        elif len(tracers) == 0:
            message = "No Tracer directory found... Generating just samples without cache!"
            return not self.Warning(message)
        if len(tracers) > 0: self.Success("!Found Tracers")
        else: self.Warning("Missing Tracers")
        self.WhiteSpace()
        return tracers

    def EmptySampleList(self):
        if self.len != 0: return False
        string = "No samples found in cache. Checking again..."
        self.Failure("=" * len(string))
        self.Failure(string)
        self.Failure("=" * len(string))
        return True

    def StartingAnalysis(self):
        string1 = "---" + " Starting Project: " + self.ProjectName + " ---"
        string = ""
        if self.Event is not None: string += " > EventGenerator < ::"
        elif self.EventCache: string += " > EventCache < ::"

        if self.Graph is not None: string += " > GraphGenerator < ::"
        elif self.DataCache: string += " > DataCache < ::"



        #string += "> SampleGenerator < :: " if self.kFolds else ""
        #string += "> Optimization < :: " if self.Model != None else ""
        #string += " > Selections < :: " if len(self.Selections) != 0 else ""
        #string += " > Merging Selections < :: " if len(self.Merge) != 0 else ""

        l = len(string) if len(string1) < len(string) else len(string1)
        self.Success("=" * l)
        self.Success(string1)
        self.Success(string)
        self.Success("=" * l)
