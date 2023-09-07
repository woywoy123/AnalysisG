from AnalysisG.Notification import _GraphGenerator
from AnalysisG._cmodules.SampleTracer import Event
from AnalysisG.Templates import GraphTemplate
from AnalysisG.SampleTracer import SampleTracer
from AnalysisG.Templates import GraphTemplate
from AnalysisG.Tools import Threading
from .Interfaces import _Interface
from typing import Union
import pickle
from time import sleep

class GraphGenerator(_GraphGenerator, SampleTracer, _Interface):
    def __init__(self, inpt = None):
        SampleTracer.__init__(self)
        _GraphGenerator.__init__(self, inpt)
        self.Caller = "GRAPHGENERATOR"
        _Interface.__init__(self)

    @staticmethod
    def _CompileGraph(inpt, _prgbar):
        lock, bar = _prgbar
        if not len(inpt): return [None]
        for i in range(len(inpt)):
            ev = inpt[i][0]
            gr_exp = inpt[i][2]
            code, graph = inpt[i][1]
            code  = pickle.loads(code)
            grx = graph(ev)
            gr = graph()
            gr.Import(gr_exp)
            gr.ImportCode(code)
            gr.Event = ev
            gr.Particles = [p.get() for p in grx.Particles]
            gr.Build()
            inpt[i] = gr
            if bar is None: continue
            elif lock is None: bar.update(1)
            else:
                with lock: bar.update(1)
        return inpt

    def MakeGraphs(self):
        if not self.CheckGraphImplementation(): return False
        if not self.CheckSettings(): return False

        itx = 1
        inpt = []
        chnks = self.Threads * self.Chunks
        step = chnks
        code = pickle.dumps(self.Graph.code)
        graph_exp = self.Graph.Export
        graph = self.Graph.clone()
        for ev, i in zip(self, range(len(self))):
            if self._StartStop(i) == False: continue
            if self._StartStop(i) == None: break
            if ev.Graph: continue
            inpt.append([ev, (code, graph), graph_exp])
            if not i >= step: continue
            itx += 1
            step = itx*chnks
            th = Threading(inpt, self._CompileGraph, self.Threads, self.Chunks)
            th.Start()
            for x in th._lists: self.AddGraph(x)
            inpt = []
        th = Threading(inpt, self._CompileGraph, self.Threads, self.Chunks)
        th.Start()
        for i in th._lists: self.AddGraph(i)
        return True

    def preiteration(self):
        if not len(self.ShowLength): return True
        if not len(self.ShowTrees): return True

        if not len(self.Tree):
            try: self.Tree = self.ShowTrees[0]
            except IndexError: return True

        if not len(self.EventName):
            try: self.EventName = self.ShowEvents[0]
            except IndexError: self.EventName = None
            self.GetEvent = True

        if not len(self.GraphName):
            try: self.GraphName = self.ShowGraphs[0]
            except IndexError: self.GraphName = None
            self.GetGraph = True

        return False
