from AnalysisG.Notification import _EventGenerator
from AnalysisG.Tools import Code, Threading
from AnalysisG.SampleTracer import SampleTracer
from .Interfaces import _Interface
from AnalysisG.IO import UpROOT
from typing import Union
from time import sleep
import pickle


class EventGenerator(_EventGenerator, _Interface, SampleTracer):
    def __init__(self, val=None):
        SampleTracer.__init__(self)
        self.Caller = "EVENTGENERATOR"
        self.InputSamples(val)

    @staticmethod
    def _CompileEvent(inpt, _prgbar):
        lock, bar = _prgbar
        ev = pickle.loads(inpt[0][0]).clone()
        tracer = SampleTracer()
        for i in range(len(inpt)):
            _, vals = inpt[i]
            res = ev.__compiler__(vals)
            for k in res:
                k.CompileEvent()
                tracer.AddEvent(k, vals["MetaData"])

            if lock is None:
                if bar is None: continue
                bar.update(1)
                continue
            with lock: bar.update(1)
        return [tracer]

    def MakeEvents(self):
        if not self.CheckEventImplementation(): return False
        self.CheckSettings()

        if len(self.ShowEvents) > 0: pass
        else: return self.ObjectCollectFailure()
        if not self.CheckROOTFiles(): return False
        if not self.CheckVariableNames(): return False

        ev = self.Event
        ev.__getleaves__()

        io = UpROOT(self.Files)
        io.Verbose = self.Verbose
        io.Trees = ev.Trees
        io.Leaves = ev.Leaves
        io.EnablePyAMI = self.EnablePyAMI

        inpt = []
        ev = pickle.dumps(ev)
        i = -1
        for v in io:
            i += 1
            if self._StartStop(i) == False: continue
            if self._StartStop(i) == None: break
            inpt.append([ev, v])

        if self.Threads > 1:
            th = Threading(inpt, self._CompileEvent, self.Threads, self.Chunks)
            th.Start()
            out = th._lists
        else: out = self._CompileEvent(inpt, self._MakeBar(len(inpt)))

        for i in out:
            if i is None: pass
            else: self += i
        return self.CheckSpawnedEvents()
