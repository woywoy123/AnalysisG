from AnalysisG.Tools import Tools
from typing import Union

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
                    self.Files |= {"/".join(i.split("/")[:-1]) : [i]}
                    continue
                self.Files |= self.ListFilesInDir(i, ".root")

        elif isinstance(val, str): 
            if val.endswith(".root"): self.Files |= {"/".join(val.split("/")[:-1]) : [val]}
            else: self.Files |= {val : self.ListFilesInDir(val, ".root")}


