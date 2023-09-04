from AnalysisG.Notification import _SelectionGenerator
from AnalysisG.SampleTracer import SampleTracer
from AnalysisG.Tools import Threading
from .Interfaces import _Interface

class SelectionGenerator(_SelectionGenerator, SampleTracer, _Interface):
    def __init__(self, inpt = None):
        SampleTracer.__init__(self)
        self.Caller = "SELECTIONGENERATOR"
        _Interface.__init__(self)
        _SelectionGenerator.__init__(self, inpt)

    @staticmethod
    def __compile__(inpt, _prgbar):
        lock, bar = _prgbar
        for i in range(len(inpt)):
            ev, sel = inpt[i]
            sel = sel.clone()
            sel.__processing__(ev)
            inpt[i] = sel
            if bar is None: continue
            elif lock is None: bar.update(1)
            else:
                with lock: bar.update(1)
        return inpt

    def MakeSelections(self):
        if self.CheckSettings(): return False
        chnks = self.Threads * self.Chunks*2

        for name in self.Selections:

            itx = 1
            inpt = []
            step = chnks
            sel = self.Selections[name]
            title = self.Caller + "::" + name
            for ev, i in zip(self, range(len(self))):
                if self._StartStop(i) == False: continue
                if self._StartStop(i) == None: break
                inpt.append([ev, sel])
                if not i >= step: continue
                itx += 1
                step = itx*chnks
                th = Threading(inpt, self.__compile__, self.Threads, self.Chunks)
                th.Title = self.Caller + "::" + name
                th.Start()
                for x in th._lists: self.AddSelections(x)
                inpt = []

            th = Threading(inpt, self.__compile__, self.Threads, self.Chunks)
            th.Start()
            for i in th._lists: self.AddSelections(i)


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

        if not len(self.SelectionName):
            try: self.SelectionName = self.ShowSelections[0]
            except IndexError: self.SelectionName = None
            self.GetSelection = True

        return False
