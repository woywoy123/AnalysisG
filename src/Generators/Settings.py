
class Settings:

    def __init__(self):
        self.EventCache = False
        self.DataCache = False

        self.VerboseLevel = 3
        self.Threads = 12
        self.chnk = 12
        
        self.EventStart = 0
        self.EventStop = None 

        self.DumpHDF5 = False
        self.DumpPickle = False

        self.OutputDirectory = False

        self.ProjectName = "UNTITLED"
        self.Tracer = None
        self.Tree = False
        self._PullCode = False
        self.Device = "cpu"

        self.InputDirectory = {}
 
    def DumpSettings(self, other):
        
        Unique = {}
        for i in self.__dict__:
            if i not in other.__dict__:
                continue
            if self.__dict__[i] == other.__dict__[i]:
                continue

            Unique[i] = other.__dict__[i]
        return Unique
