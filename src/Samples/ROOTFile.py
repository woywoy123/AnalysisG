from .Hashing import Hashing 

class ROOTFile(Hashing):

    def __init__(self):
        self.Trees = []
        self.Branches = []
        self.Leaves = []
        self.EventIndex = {}
        self.Filename = None
        self._HashMap = {}

    def NextEvent(self, Tree):
        self.EventIndex[Tree] +=1

    def MakeHash(self, it):
        self._HashMap[it] = self.MD5(self.Filename + "/" + str(it))

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __add__(self, other):
        if isinstance(other, list):
            for i in other:
                self.__add__(i)
            return self
        
        if self.Filename != other.Filename:
            return [self, other]

        self.Trees = list(set(self.Trees + other.Trees))
        self.Branches = list(set(self.Branches + other.Branches))
        self.Leaves = list(set(self.Leaves + other.Leaves))
        
        for i in other.EventIndex:
            if i not in self.EventIndex:
                self.EventIndex[i] = 0
            self.EventIndex[i] += other.EventIndex[i]
        
        hashmap = list(other._HashMap.values()) + list(self._HashMap.values())
        self._HashMap = {hashmap.index(i) : i for i in set(hashmap)}
        
        return [self]


