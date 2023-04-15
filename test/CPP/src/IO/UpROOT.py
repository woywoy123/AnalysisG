import uproot 
from AnalysisG.Settings import Settings
from AnalysisG.Notification import _UpROOT

class UpROOT(_UpROOT, Settings):
    
    def __init__(self, ROOTFile):
        self.Caller = "Up-ROOT"
        Settings.__init__(self)
        self.ROOTFile = {}
        if isinstance(ROOTFile, str):
            self.ROOTFile = { ROOTFile : uproot.open(ROOTFile)}
        elif isinstance(ROOTFile, list):
            self.ROOTFile = {i : uproot.open(i) for i in ROOTFile}
        
        if len(self.ROOTFile) == 0:
            self.InvalidROOTFileInput
            self.ROOTFile = False
            return 
        self.Keys = {}
        self._it = False 

    @property
    def _StartIter(self):
        if self._it: return
        self._it = iter(list(self.ROOTFile))

    @property
    def ScanKeys(self):
        if not self.ROOTFile:
            return False
        
        def Recursion(inpt, k_, keys):
            for i in keys:
                k__ = k_ + "/" + i
                try:
                    k_n = inpt[k__].keys()
                    self._struct[k__] = None
                except AttributeError:
                    continue
                Recursion(inpt, k__, k_n)
        
        self._StartIter
        try:
            fname = next(self._it)
        except StopIteration:
            self._it = False
            return 

        f = self.ROOTFile[fname]
        self._struct = {}
        self._missed = {"TREE" : [], "BRANCH" : [], "LEAF" : []}
        
        Recursion(f, "", f.keys())
        
        found = {}
        for i in self.Trees:
            found |= {k : self._struct[k] for k in self._struct if i in k}
        self.CheckValidKeys(self.Trees, found, "TREE")
        found = {}

        for i in self.Branches:
            found |= {k : self._struct[k] for k in self._struct if i in k.split("/")}
        self.CheckValidKeys(self.Branches, found, "BRANCH")
        
        for i in self.Leaves:
            found |= {k : self._struct[k] for k in self._struct if i in k.split("/")}
        self.CheckValidKeys(self.Leaves, found, "LEAF")
        
        self.Keys[fname] = {"found" : found, "missed" : self._missed}
        self.AllKeysFound(fname)        
        
        self.ScanKeys

    def __iter__(self):
        self.ScanKeys
        keys = self.Keys[list(self.ROOTFile)[0]]["found"]
        
        t = { T : [r for r in self.ROOTFile if T not in self.Keys[r]["missed"]["TREE"]] for T in self.Trees }
        get = {tr : [i.split("/")[-1] for i in keys if tr in i] for tr in t}
        self._root = {tr : uproot.concatenate([r + ":" + tr for r in t[tr]], get[tr], library = "np", step_size = self.StepSize) for tr in get}
        self._root = {tr + "/" + l : self._root[tr][l] for tr in self._root for l in self._root[tr]} 
        return self 
    
    def __len__(self):
        self.__iter__()
        return len(self._root[list(self._root)[-1]])

    def __next__(self):
        try:
            r = {k : self._root[k][-1].tolist() for k in self._root}
        except IndexError:
            raise StopIteration
        self._root = {k : self._root[k][:-1] for k in self._root}
        return r
