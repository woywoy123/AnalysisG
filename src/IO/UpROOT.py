from AnalysisTopGNN.Notification.UpROOT import UpROOT
from AnalysisTopGNN.Generators.Settings import Settings
from AnalysisTopGNN.Tools import Threading
import uproot 

class File(UpROOT, Settings):
    def __init__(self, ROOTFile, Threads = 1):
        self.Caller = "FILE"
        Settings.__init__(self)
        self.Trees = []
        self.Branches = []
        self.Leaves = []
        self.ROOTFile = ROOTFile
        self.Threads = Threads 
        self._Reader =  uproot.open(self.ROOTFile)
        self._State = None
   
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

    def __iter__(self):
        def function(lst):
            dic = {}
            for k in lst:
                if k[0] not in dic:
                    dic[k[0]] = []
                dic[k[0]].append(k[1])
           
            out = {tr + "-" + br : [] for tr in dic for br in dic[tr]}
            it_r = {tr : self._Reader[tr].iterate(dic[tr], library = "ak", step_size = self.StepSize) for tr in dic}
            it = {tr : [] for tr in it_r}
            while True:
                if sum([len(it[tr]) for tr in it]) == 0:
                    try:
                        it = {tr : next(it_r[tr]).to_list() for tr in it_r}
                    except:
                        break
                for tr in dic:
                    tmp = it[tr].pop(0)
                    for br in dic[tr]:
                        out[tr + "-" + br].append(tmp[br])

            for i in range(len(lst)):
                lst[i].append(out[lst[i][0] + "-" + lst[i][1]])
            return lst

        All = [b.split("/")[-1] for b in self.Branches] + [l.split("/")[-1] for l in self.Leaves]
        lsts = [[self.Trees[tr], All[i]] for tr in range(len(self.Trees)) for i in range(len(All))]
        self.Iter = {tree : {} for tree in self.Trees} 

        th = Threading(lsts, function, self.Threads)
        th.VerboseLevel = 1
        th.Start()

        for i in th._lists:
            self.Iter[i[0]] |= {i[1] : i[2]}
        return self

    def __next__(self):
        try: 
            return {Tree : {br : self.Iter[Tree][br].pop(0) for br in self.Iter[Tree]} for Tree in self.Iter}
        except:
            raise StopIteration



