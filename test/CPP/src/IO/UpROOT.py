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
        if len(self.Keys) != len(self.ROOTFile): self.ScanKeys
        keys = self.Keys[list(self.ROOTFile)[0]]["found"]
        
        self._t = { T : [r for r in self.ROOTFile if T not in self.Keys[r]["missed"]["TREE"]] for T in self.Trees }
        self._get = {tr : [i.split("/")[-1] for i in keys if tr in i] for tr in self._t}
        dct = {
                tr : {
                    "files" : {r : tr for r in self._t[tr]}, 
                    "library" : "np", 
                    "step_size" : self.StepSize, 
                    "report" : True, 
                    "how" : dict, 
                    "expressions" : self._get[tr], 
                } 
                for tr in self._get}
        self._root = {tr : uproot.iterate(**dct[tr]) for tr in dct}
        self._r = None
        self._cur_r = None
        self._EventIndex = 0
        return self 
    
    def __len__(self):
        self.__iter__()
        v = {tr : sum([uproot.open(r + ":" + tr).num_entries for r in self._t[tr]]) for tr in self._get}
        return list(v.values())[0]

    def __next__(self):
        try:
            r = {key : self._r[key][0].pop() for key in self._r}
            _r = self._r[list(r)[0]][1]
            fname = _r.file_path
            self._EventIndex = 0 if self._cur_r != fname else self._EventIndex+1
            self._cur_r = fname

            r |= {"ROOT" : fname, "EventIndex" : self._EventIndex}
            return r
        except:
            r = {tr : next(self._root[tr]) for tr in self._root}
            self._r = {tr + "/" + l : [r[tr][0][l].tolist(), r[tr][1]] for tr in r for l in r[tr][0]}            
            return self.__next__()
