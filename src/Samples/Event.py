from .Hashing import Hashing

class EventContainer(Hashing):

    def __init__(self):
        self.Trees = {}
        self.EventIndex = 0
        self.Filename = None
        self.Compiled = False
        self.Train = None

    def MakeEvent(self, ClearVal):
        for i in self.Trees:
            self.Trees[i]._Compile(ClearVal)
            self.Filename = self.MD5(self.Filename + "/" + str(self.Trees[i]._SampleIndex))
    
    def MakeGraph(self):
        if self.Compiled: 
            return self
        for i in self.Trees:
            self.Trees[i] = self.Trees[i].ConvertToData()
        self.Compiled = True
        return self

    def __add__(self, other):
        if other == None:
            return self
        
        if self.Filename != other.Filename:
            return [self, other]
        self.Trees |= other.Trees
        return self

    def UpdateIndex(self, index):
        self.EventIndex = index
        return self
