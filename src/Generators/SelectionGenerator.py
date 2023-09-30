from AnalysisG.Notification import _SelectionGenerator
from AnalysisG.SampleTracer import SampleTracer
from AnalysisG.Tools import Threading
from .Interfaces import _Interface
import pickle

class SelectionGenerator(_SelectionGenerator, SampleTracer, _Interface):
    def __init__(self, inpt = None):
        SampleTracer.__init__(self)
        self.Caller = "SELECTION-GENERATOR"
        _Interface.__init__(self)
        _SelectionGenerator.__init__(self, inpt)

    @staticmethod
    def _CompileSelection(inpt, _prgbar):
        lock, bar = _prgbar
        for i in range(len(inpt)):
            ev, co, sel = pickle.loads(inpt[i])
            ev, meta = ev
            ev.ImportMetaData(meta)
            se = co.InstantiateObject
            se.__setstate__(sel)
            se.__processing__(ev)
            inpt[i] = se.__getstate__()
            del sel
            del co
            if bar is None: continue
            elif lock is None: bar.update(1)
            else:
                with lock: bar.update(1)
        return inpt

    def MakeSelections(self, sample = None):
        if sample is not None: pass
        else: sample = self
        self.preiteration(sample)

        if self.CheckSettings(sample): return False

        chnks = self.Threads * self.Chunks
        command = [[], self._CompileSelection, self.Threads, self.Chunks]

        path = sample.Tree + "/" + sample.EventName
        try: itr = sample.ShowLength[path]
        except KeyError: itr = 0
        if not itr: return False

        code = {i.class_name : i for i in self.rebuild_code(None)}
        for name in sample.Selections:
            itx = 1
            step = chnks
            co = code[name]
            sel = sample.Selections[name].__getstate__()
            title = self.Caller + "::" + name
            _, bar = self._makebar(itr, self.Caller + "::Preparing...")
            for ev, i in zip(sample, range(itr)):
                if self._StartStop(i) == False: continue
                if self._StartStop(i) == None: break

                meta = ev.meta()
                ev = ev.release_event()
                if not ev: continue
                command[0] += [pickle.dumps(((ev, meta), co, sel))]
                bar.update(1)

                if not i >= step: continue
                itx += 1
                step = itx*chnks
                th = Threading(*command)
                th.Title = self.Caller + "::" + name
                th.Verbose = self.Verbose
                th.Start()
                for x in th._lists: sample.AddSelections(x)
                command[0] = []
                del th

            if not len(command[0]): continue
            th = Threading(*command)
            th.Verbose = self.Verbose
            th.Start()
            for i in th._lists: sample.AddSelections(i)
            command[0] = []
            del th
        return True

    def preiteration(self, inpt = None):
        if inpt is not None: pass
        else: inpt = self

        if not len(inpt.ShowLength): return True
        if not len(inpt.ShowTrees): return True

        if not len(inpt.Tree):
            try: inpt.Tree = inpt.ShowTrees[0]
            except IndexError: return True

        if not len(inpt.EventName):
            try: inpt.EventName = inpt.ShowEvents[0]
            except IndexError: inpt.EventName = None
            self.GetEvent = True

        if not len(inpt.SelectionName):
            try: inpt.SelectionName = inpt.ShowSelections[0]
            except IndexError: inpt.SelectionName = None
            self.GetSelection = True
        return False
