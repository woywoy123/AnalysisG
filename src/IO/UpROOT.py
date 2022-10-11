from AnalysisTopGNN.Notification import UpROOT
import uproot 

class File(UpROOT):
    def __init__(self, ROOTFile, Threads = 1):
        self.Caller = "FILE"
        self.Trees = []
        self.Branches = []
        self.Leaves = []
        self.ROOTFile = ROOTFile

        self._Reader =  uproot.open(self.ROOTFile, num_workers = Threads)
        self._State = None
        self.Tracer = False
   
    def _CheckKeys(self, List, Type):
        TMP = []

        t = []
        for i in [k for k in self._State.keys() if "/" not in k]:
            if ";" in i:
                i = i.split(";")[0]
            t.append(i)

        for i in List:
            if i not in t:
                continue
            TMP.append(i)
            
        return TMP
    
    def _GetKeys(self, List1, List2, Type):
        TMP = []
        for i in List1:
            self._State = self._Reader[i]
            TMP += [i + "/" + j for j in self._CheckKeys(List2, Type)]
             
        for i in TMP:
            if i.split("/")[-1] in List2:
                List2.pop(List2.index(i.split("/")[-1]))
        
        if len(List1) == 0:
            return []
        self.SkippedKey(Type, List2) 
        return TMP

    def _GetBranches(self):
        return self._GetKeys(self.Trees, self.Branches, "BRANCH")
    
    def _GetLeaves(self):
        leaves = []
        leaves += self._GetKeys(self.Branches, self.Leaves, "LEAF")
        leaves += self._GetKeys(self.Trees, self.Leaves, "LEAF")
        return leaves 
    
    def ValidateKeys(self):
        self.Leaves = list(set(self.Leaves))
        self.Branches = list(set(self.Branches))
        self.Trees = list(set(self.Trees))
        self._State = self._Reader 
        self.Trees = self._CheckKeys(self.Trees, "TREE")
        self.Branches = self._GetBranches()
        self.Leaves = self._GetLeaves() 

        if self.Tracer:
            self.Tracer.AddROOTFile(self)
        
    def GetTreeValues(self, Tree):
        All = []
        All += [b.split("/")[-1] for b in self.Branches]
        All += [l.split("/")[-1] for l in self.Leaves]
        self._All = All
        self._Tree = Tree 
 
    def __iter__(self):
        self._C = []
        self._iter = self._Reader[self._Tree].iterate(self._All, library = "ak", step_size = 10000)
        return self

    def __next__(self):
        if len(self._C) == 0:
            self.Iter = next(self._iter)
            self._C += self.Iter.to_list()
        
        if self.Tracer:
            self.Tracer.ROOTInfo[self.ROOTFile].NextEvent(self._Tree)

        return self._C.pop(0)


