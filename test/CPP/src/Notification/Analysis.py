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

    @property
    def _BuildingCache(self):
        if self.EventCache: 
            self.mkdir(self.OutputDirectory + "/EventCache")
            self.Success("Created EventCache under: " + self.OutputDirectory)
        if self.DataCache:  
            self.mkdir(self.OutputDirectory + "/DataCache")
            self.Success("Created DataCache under: " + self.OutputDirectory)
