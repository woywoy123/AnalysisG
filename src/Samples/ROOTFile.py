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
