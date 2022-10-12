from .Hashing import Hashing

class Event(Hashing):

    def __init__(self):
        self.Trees = {}
        self.EventIndex = 0
        self.Filename = None

    def MakeEvent(self, ClearVal):
        for i in self.Trees:
            self.Trees[i]._Compile(ClearVal)
            self.Filename = self.MD5(self.Filename + "/" + str(self.Trees[i]._SampleIndex))
    
