from AnalysisG.Tracer import SampleTracer
from AnalysisG.Notification import _EventGenerator
from AnalysisG.Settings import Settings
from AnalysisG.Tools import Code, Threading
from AnalysisG.IO import UpROOT
from .Interfaces import _Interface
from typing import Union

class EventGenerator(_EventGenerator, Settings, SampleTracer, _Interface):
    
    def __init__(self,  val = None):
        self.Caller = "EVENTGENERATOR"
        Settings.__init__(self)
        SampleTracer.__init__(self)
        self.InputSamples(val) 

    @property
    def MakeEvents(self):
        if not self.CheckEventImplementation: return self
        if "Event" not in self._Code: self._Code["Event"] = []
        self._Code["Event"].append(Code(self.Event))
        try: 
            dc = self.Event.Objects
            if not isinstance(dc, dict): raise AttributeError
        except AttributeError: self.Event = self.Event()
        except TypeError: self.Event = self.Event()
        except: return self.ObjectCollectFailure
        
        self._Code["Objects"] = {i : Code(self.Event.Objects[i]) for i in self.Event.Objects}
        
        if not self.CheckROOTFiles: return self
        
        ev = self.Event
        ev.__interpret__

        io = UpROOT([i + "/" + k for i in self.Files for k in self.Files[i]])
        io.Verbose = self.Verbose
        io.Trees = ev.Trees
        io.Leaves = ev.Leaves 
        
        from tqdm import tqdm 
        for val in tqdm(io):
            root, index = val["ROOT"], val["EventIndex"]
            res = ev.__compiler__(val)
            for j in res:
                j.CompileEvent()
            self.AddEvent(res, root, index)

