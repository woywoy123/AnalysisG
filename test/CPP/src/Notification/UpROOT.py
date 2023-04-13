from .Notification import Notification
class _UpROOT(Notification):

    def __init__(self):
        pass
    
    @property
    def InvalidROOTFileInput(self):
        self.Failure("Invalid Input. Provide either a string/list of ROOT file/s")
    
    
    def ReadingFile(self, Name):
        self.Success("!!!(Reading) -> " + Name.split("/")[-1] + 
                     " (" + ", ".join([t + " - " + str(self._Reader[t].num_entries) for t in self.Trees]) + ")")

    def CheckValidKeys(self, requested, found, Type):
        if len(requested) == 0: return 
        
        for i in requested:
            if len([k for k in found if i in k]) > 0: continue
            
            if Type not in self._missed: self._missed[Type] = []
            self._missed[Type].append(i)
            self.Warning("SKIPPED: " + Type + "::" + i)
    
    def AllKeysFound(self, fname):
        if len(self._missed) == 0: 
            self.Success("!!!All requested keys were found for " + fname)
            return None
        return self.Failure("Missing keys detected in: " + fname)
        
