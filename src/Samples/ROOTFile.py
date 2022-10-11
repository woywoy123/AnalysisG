class ROOTFile:

    def __init__(self):
        self.Trees = []
        self.Branches = []
        self.Leaves = []
        self.EventIndex = {}
        self.FileName = None

    def NextEvent(self, Tree):
        self.EventIndex[Tree] +=1
