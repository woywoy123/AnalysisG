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
                    self._struct[k__] = inpt[k__].iterate(library = "np", step_size = self.StepSize)
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
        self._missed = {}
        
        Recursion(f, "", f.keys())
        
        found = {}
        for i in self.Trees:
            found |= {k : self._struct[k] for k in self._struct if i in k}
        self.CheckValidKeys(self.Trees, found, "TREE")

        for i in self.Branches:
            found |= {k : self._struct[k] for k in self._struct if i in k}
        self.CheckValidKeys(self.Branches, found, "BRANCH")
        
        for i in self.Leaves:
            found |= {k : self._struct[k] for k in self._struct if i in k}
        self.CheckValidKeys(self.Leaves, found, "LEAF")
        
        self.ROOTFile[fname] = {"found" : found, "missed" : self._missed}
        self.AllKeysFound(fname)        
        
        self.ScanKeys
