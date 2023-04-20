from AnalysisG.Tools import Tools
from typing import Union
from tqdm import tqdm 

class _Interface(Tools):
    
    def __init__(self):
        pass

    def InputSamples(self, val: Union[dict[str], list[str], str, None]):
        if isinstance(val, dict): 
            for i in val:
                if len(val[i]) != 0: 
                    self.Files |= {i : [v for v in val[i] if v.endswith(".root")]}
                    continue
                self.Files |= self.ListFilesInDir(i, ".root")

        elif isinstance(val, list): 
            for i in val:
                if i.endswith(".root"): 
                    _dir = "/".join(i.split("/")[:-1])
                    if _dir not in self.Files: self.Files[_dir] = []
                    self.Files[_dir].append(i.split("/")[-1])
                    continue
                self.Files |= self.ListFilesInDir(i, ".root")

        elif isinstance(val, str): 
            if val.endswith(".root"): self.Files |= {"/".join(val.split("/")[:-1]) : [val.split("/")[-1]]}
            else: self.Files |= {val : self.ListFilesInDir(val, ".root")}

    def _StartStop(self, it: Union[int]):
        if self.EventStart > it and self.EventStart != -1: return False
        if self.EventStop != None and self.EventStop-1 < it: return None
        return True

    def _MakeBar(self, inpt: Union[int], CustTitle: Union[None, str] = None):
        _dct = {}
        _dct["desc"] = CustTitle if CustTitle == None else f'Progress {self.Caller}'
        _dct["leave"] = False
        _dct["colour"] = "GREEN"
        _dct["dynamic_ncols"] = True
        _dct["total"] = inpt
        return (None, tqdm(**_dct))

    def SetAttribute(self, c_name, fx, container):
        if c_name == "P_" or c_name == "T_": c_name += fx.__name__ 
        elif c_name == "": c_name += fx.__name__ 
        if c_name not in container: container[c_name] = fx
        else: self.Warning("Found Duplicate " + c_name + " Attribute")
 
    # Define the observable features
    def AddGraphFeature(self, fx, name = ""):
        self.SetAttribute(name, fx, self.GraphAttribute)

    def AddNodeFeature(self, fx, name = ""):
        self.SetAttribute(name, fx, self.NodeAttribute)

    def AddEdgeFeature(self, fx, name = ""):
        self.SetAttribute(name, fx, self.EdgeAttribute)

    
    # Define the truth features used for supervised learning 
    def AddGraphTruth(self, fx, name = ""):
        self.SetAttribute("T_" + name, fx, self.GraphAttribute)

    def AddNodeTruth(self, fx, name = ""):
        self.SetAttribute("T_" + name, fx, self.NodeAttribute)

    def AddEdgeTruth(self, fx, name = ""):
        self.SetAttribute("T_" + name, fx, self.EdgeAttribute)

    
    # Define any last minute changes to attributes before adding to graph
    def AddGraphPreprocessing(self, name, fx):
        self.SetAttribute("P_" + name, fx, self.GraphAttribute)

    def AddNodePreprocessing(self, name, fx):
        self.SetAttribute("P_" + name, fx, self.NodeAttribute)

    def AddEdgePreprocessing(self, name, fx):
        self.SetAttribute("P_" + name, fx, self.EdgeAttribute)
    
    def AddSelection(self, name: Union[str], function):
        self.Selections[name] = function

    def MergeSelection(self, name: Union[str]):
        self.Merge[name] = True 
