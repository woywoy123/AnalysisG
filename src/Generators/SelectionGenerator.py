from AnalysisG.Notification import _SelectionGenerator
from AnalysisG.Tools import Code, Threading 
from .EventGenerator import EventGenerator
from AnalysisG.Tracer import SampleTracer
from AnalysisG.Settings import Settings 
from .Interfaces import _Interface
from typing import Union 
from time import sleep

class SelectionGenerator(_SelectionGenerator, Settings, SampleTracer, _Interface):
    
    def __init__(self, inpt: Union[EventGenerator, None] = None):
        self.Caller = "SELECTIONGENERATOR"
        Settings.__init__(self)
        SampleTracer.__init__(self)
        _Interface.__init__(self)
        _SelectionGenerator.__init__(self, inpt)
  
    @staticmethod 
    def __compile__(inpt, _prgbar):
        lock, bar = _prgbar
        output = {}
        for i in range(len(inpt)):
            name, sel, event, hash_, ROOT = inpt[i] 
            sel.hash = hash_ 
            sel.ROOTName = ROOT
            sel._EventPreprocessing(event)

            if name not in output: output[name] = sel
            else: output[name]+=sel

            if lock == None: bar.update(1)
            else:
                with lock: bar.update(1)
                sleep(0.001) # This actually improves speed!!!
        if lock == None: del bar
        return [output]

    def __collect__(self, inpt, key):
        x = {c_name : Code(inpt[c_name]) for c_name in inpt}
        if len(x) != 0: self._Code[key] = x 
    
    @property
    def __merge__(self):
        self.result = {i : self.Merge[i].pop(0) for i in self.Merge}
        for name in self.Merge: self.result[name] += sum(self.Merge[name])
    
    @property
    def MakeSelection(self):
        self.__collect__(self.Selections, "Selections") 
        if self._condor: return self._Code
        if len(self.Merge) != 0: pass
        elif self.CheckSettings: return False
        
        inpt = []
        for name in self.Selections:
            if name not in self.Merge: self.Merge[name] = []
            for ev, i in zip(self, range(len(self))):
                if self._StartStop(i) == False: continue
                if self._StartStop(i) == None: break
                sel = self._Code["Selections"][name].clone
                inpt.append([name, sel, ev, ev.hash, ev.ROOT])
        if len(inpt) == 0 and len(self.Merge) != 0: return self.__merge__
        if self.Threads > 1:
            th = Threading(inpt, self.__compile__, self.Threads, 2)
            th.Title = self.Caller
            th.Start
            out = th._lists
        else: out = self.__compile__(inpt, self._MakeBar(len(inpt)))
        
        for i in out: 
            if i is None: continue
            for name in i: self.Merge[name].append(i[name])
        self.__merge__
