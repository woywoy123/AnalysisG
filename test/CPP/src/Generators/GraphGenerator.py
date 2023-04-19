from AnalysisG.Generators.EventGenerator import EventGenerator
from AnalysisG.Notification import _GraphGenerator
from AnalysisG.Tools import Code, Threading
from AnalysisG.Tracer import SampleTracer
from AnalysisG.Settings import Settings
from .Interfaces import _Interface
from typing import Union

class GraphGenerator(_GraphGenerator, Settings, SampleTracer, _Interface):
    
    def __init__(self, inpt: Union[EventGenerator, None] = None):
        
        self.Caller = "GRAPHGENERATOR"
        Settings.__init__(self)
        SampleTracer.__init__(self)
        _Interface.__init__(self)
        _GraphGenerator.__init__(self, inpt)
 
    @staticmethod 
    def _CompileGraph(inpt, _prgbar):
        lock, bar = _prgbar
        for i in range(len(inpt)):
            try: 
                inpt[i].ConvertToData()
                inpt[i] = inpt[i].purge
            except AttributeError: pass
            if lock == None: bar.update(1)
            else:
                with lock: bar.update(1)
                sleep(0.001) # This actually improves speed!!!
        if lock == None: del bar
        return inpt

    @property 
    def MakeGraphs(self):
        if not self.CheckGraphImplementation: return self
        if not self.CheckSettings: return self

        if "EventGraph" not in self._Code: self._Code["EventGraph"] = []
        self._Code["EventGraph"].append(Code(self.EventGraph))
        
        Features = {}
        Features |= {c_name : Code(self.GraphAttribute[c_name]) for c_name in self.GraphAttribute}
        Features |= {c_name : Code(self.NodeAttribute[c_name]) for c_name in self.NodeAttribute}   
        Features |= {c_name : Code(self.EdgeAttribute[c_name]) for c_name in self.EdgeAttribute}
       

        inpt = []
        for ev, i in zip(self, range(len(self))):
            if self._StartStop(i) == False: continue
            if self._StartStop(i) == None: break

            gr = self._Code["EventGraph"][-1].clone
            try: gr = gr(ev)
            except AttributeError:
                gr = gr.Escape(gr)
                gr.Event = ev
                gr.Particles = []
            gr.GraphAttr |= self.GraphAttribute     
            gr.NodeAttr |= self.NodeAttribute     
            gr.EdgeAttr |= self.EdgeAttribute    
            gr.index = ev.index
            gr.SelfLoop = self.SelfLoop
            gr.FullyConnect = self.FullyConnect
            inpt.append(gr)
             
        if self.Threads > 1:
            th = Threading(inpt, self._CompileGraph, self.Threads, self.chnk)
            th.Title = self.Caller
            th.Start
        out = th._lists if self.Threads > 1 else self._CompileGraph(inpt, self._MakeBar(len(inpt)))
        
        print(out)
        #for i in out: self.AddEvent(i[0], i[1], i[2])
 
