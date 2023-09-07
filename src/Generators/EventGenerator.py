from AnalysisG.Notification import _EventGenerator
from AnalysisG.SampleTracer import SampleTracer
from AnalysisG.IO.UpROOT import UpROOT
from AnalysisG.Tools import Threading
from .Interfaces import _Interface
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
            inpt[i] = []
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
        if len(self.ShowEvents) > 0: pass
        else: return self.ObjectCollectFailure()
        if not self.CheckROOTFiles(): return False
        if not self.CheckVariableNames(): return False

        ev = self.Event
        ev.__getleaves__()

        io = UpROOT(self.Files, True)
        io.Verbose = self.Verbose
        io.Trees = ev.Trees
        io.Leaves = ev.Leaves
        io.EnablePyAMI = self.EnablePyAMI

        i = -1
        ev = pickle.dumps(ev)
        inpt = []
        chnks = self.Threads * self.Chunks*2
        step = chnks
        itx = 1
        for v in io:
            i += 1
            if self._StartStop(i) == False: continue
            if self._StartStop(i) == None: break

            inpt.append([ev,v])
            if not i >= step: continue
            itx += 1
            step = itx*chnks
            th = Threading(inpt, self._CompileEvent, self.Threads, self.Chunks)
            th.Start()
            for x in th._lists:
                if x is None: continue
                self += x
            inpt = []

        th = Threading(inpt, self._CompileEvent, self.Threads, self.Chunks)
        th.Start()
        for i in th._lists:
            if i is None: continue
            self += i
        return self.CheckSpawnedEvents()

    def preiteration(self):
        self.GetEvent = True
        self.GetGraph = False
        if not len(self.EventName):
            try: self.EventName = self.ShowEvents[0]
            except IndexError: return True
        if not len(self.Tree):
            try: self.Tree = self.ShowTrees[0]
            except IndexError: return True
        return False
