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
    def _CompileSelection(inpt, _prgbar):
        lock, bar = _prgbar
        output = {}
        for i in range(len(inpt)):
            name, sel, event, hash, ROOT = inpt[i] 
            sel.hash = hash 
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

    @property
    def MakeSelection(self):
        if self.CheckSettings: return self
        if "Selections" not in self._Code: self._Code["Selections"] = []
        code = { name : Code(self.Selections[name]) for name in self.Selections}
        self._Code["Selections"].append(code)
        
        inpt = []
        for name in self.Selections:
            for ev, i in zip(self, range(len(self))):
                if self._StartStop(i) == False: continue
                if self._StartStop(i) == None: break
                sel = self._Code["Selections"][-1][name].clone
                inpt.append([name, sel, ev, ev.hash, ev.ROOT])
        
        if self.Threads > 1:
            th = Threading(inpt, self._CompileSelection, self.Threads, 2)
            th.Title = self.Caller
            th.Start
        out = th._lists if self.Threads > 1 else self._CompileSelection(inpt, self._MakeBar(len(inpt)))
        
        result = {}
        for i in out: 
            if i is None: continue
            for name in i:
                if name not in result: result[name] = i[name]
                else: result[name] += i[name]
        self.result = result
