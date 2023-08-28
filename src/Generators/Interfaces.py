from AnalysisG.Tools import Tools
from typing import Union, Dict, List
from tqdm import tqdm


class _Interface(Tools):
    def __init__(self):
        pass

    def InputSelection(self, val: Union[str]):
        self.Files = self.ListFilesInDir(val, ".hdf5")

    def InputSamples(self, val: Union[Dict, List, str, None]):
        self.Files = self.ListFilesInDir(val, ".root")

    def InputSample(
        self,
        Name: Union[str] = None,
        SampleDirectory: Union[str, Dict, List, None] = None,
    ):
        self.Files = {}
        self.InputSamples(SampleDirectory)
        if Name is None: Name = ""
        self.SampleMap[Name] = self.Files

    def _StartStop(self, it: Union[int]):
        if self.EventStart > it and self.EventStart != -1: return False
        if self.EventStop != None and self.EventStop - 1 < it: return None
        return True

    def _MakeBar(
        self, inpt: Union[int], CustTitle: Union[None, str] = None, Leave=False
    ):
        _dct = {}
        _dct["desc"] = f"Progress {self.Caller}" if CustTitle is None else CustTitle
        _dct["leave"] = Leave
        _dct["colour"] = "GREEN"
        _dct["dynamic_ncols"] = True
        _dct["total"] = inpt
        return (None, tqdm(**_dct))

    # Define the observable features
    def AddGraphFeature(self, fx, name=""):
        c_name, dupl = self.SetAttribute(name, fx, "G")
        if dupl: self.Warning("Found Duplicate Graph " + c_name + " Attribute")

    def AddNodeFeature(self, fx, name=""):
        self.SetAttribute(name, fx, self.NodeAttribute)

    def AddEdgeFeature(self, fx, name=""):
        self.SetAttribute(name, fx, self.EdgeAttribute)

    # Define the truth features used for supervised learning
    def AddGraphTruth(self, fx, name=""):
        self.SetAttribute("T_" + name, fx, self.GraphAttribute)

    def AddNodeTruth(self, fx, name=""):
        self.SetAttribute("T_" + name, fx, self.NodeAttribute)

    def AddEdgeTruth(self, fx, name=""):
        self.SetAttribute("T_" + name, fx, self.EdgeAttribute)

    # Define any last minute changes to attributes before adding to graph
    def AddGraphPreprocessing(self, name, fx):
        self.SetAttribute("P_" + name, fx, self.GraphAttribute)

    def AddNodePreprocessing(self, name, fx):
        self.SetAttribute("P_" + name, fx, self.NodeAttribute)

    def AddEdgePreprocessing(self, name, fx):
        self.SetAttribute("P_" + name, fx, self.EdgeAttribute)

    # Selection generator
    def AddSelection(self, name: Union[str], function):
        self.Selections[name] = function

    def MergeSelection(self, name: Union[str]):
        self.Merge[name] = []

    def This(self, path: Union[str], tree: Union[str]):
        if tree not in self._DumpThis: self._DumpThis[tree] = []
        if path in self._DumpThis[tree]: return self.Warning("'" + path + "' is already in list")
        self._DumpThis[tree].append(path)
