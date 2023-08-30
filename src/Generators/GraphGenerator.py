from AnalysisG.Generators.EventGenerator import EventGenerator
from AnalysisG.Notification import _GraphGenerator
from AnalysisG.SampleTracer import SampleTracer
from AnalysisG.Tools import Threading
from .Interfaces import _Interface
from typing import Union
import pickle
from time import sleep

class GraphGenerator(_GraphGenerator, SampleTracer, _Interface):
    def __init__(self, inpt: Union[EventGenerator, None] = None):
        SampleTracer.__init__(self)
        _GraphGenerator.__init__(self, inpt)
        self.Caller = "GRAPHGENERATOR"
        _Interface.__init__(self)

    @staticmethod
    def _CompileGraph(inpt, _prgbar):
        lock, bar = _prgbar
        tr_ = None
        gr_ = None
        for i in range(len(inpt)):
            ev, gr, sett = inpt[i]
            if tr_ is None:
                tr = SampleTracer()
                tr.ImportSettings(sett)
            if gr_ is None: gr_ = gr.clone()
            print(ev, gr, sett)
            sleep(1)
            if lock is None:
                if bar is None: continue
                bar.update(1)
                continue
            with lock: bar.update(1)
        return [tr]

    def MakeGraphs(self):
        #if not self.CheckGraphImplementation(): return False
        #if not self.CheckSettings(): return False

        inpt = []
        s = len(self)
        _, bar = self._MakeBar(s, "PREPARING GRAPH COMPILER")
        sett = self.ExportSettings()
        print(sett)
        print(self.ShowTrees)
        print(self.ShowEvents)
        for i in self: print("here")
        for ev, i in zip(self, range(s)):
            print(ev)
            if self._StartStop(i) == False: continue
            if self._StartStop(i) == None: break
            bar.update(1)
            if ev.Graph: continue
            inpt.append([ev, self.Graph, sett])

        if len(inpt) == 0: return True
        if self.Threads > 1:
            th = Threading(inpt, self._CompileGraph, self.Threads, self.chnk)
            th.Title = self.Caller
            out = th.Start()
        else:
            out = self._CompileGraph(inpt, (None, None)) #self._MakeBar(len(inpt)))

        self += sum([i for i in out if i is not None])
        return True
