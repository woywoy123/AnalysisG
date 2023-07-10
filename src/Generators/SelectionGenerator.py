from AnalysisG.Notification import _SelectionGenerator
from AnalysisG.Templates import SelectionTemplate
from AnalysisG.IO import PickleObject, UnpickleObject
from AnalysisG.Tools import Code, Threading, Hash
from .EventGenerator import EventGenerator
from AnalysisG.Tracer import SampleTracer
from AnalysisG.Settings import Settings
from .Interfaces import _Interface
from typing import Union
import h5py
import sys

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
        fname = ""
        for i in range(len(inpt)):
            name, sel, event, pth = inpt[i]
            sel = SampleTracer._decoder(sel)
            sel.hash = event.hash
            sel.ROOTName = event.ROOT
            sel.index = event.index
            sel.Tree = event.Tree
            sel._EventPreprocessing(event)
            fname = Hash(fname + sel.hash)
            if name not in output: output[name] = []
            output[name].append(sel)

            if lock == None: bar.update(1)
            else:
                with lock: bar.update(1)
        if lock == None: del bar
        for name in output:
            f = h5py.File(pth + name + "/" + fname + ".hdf5", "w")
            for sel in output[name]:
                ref = f.create_dataset(sel.hash, (1), dtype = h5py.ref_dtype)
                ref.attrs[name] = SampleTracer._encoder(sel)
            f.close()
        return []

    def __collect__(self, inpt, key):
        x = {c_name: Code(inpt[c_name]) for c_name in inpt}
        if len(x) != 0: self._Code[key] = x

    @property
    def __merge__(self):
        for name in self.Merge:
            if len(self.Merge[name]) == 0: continue
            self.mkdir(self.OutputDirectory + "/Selections/Merged/")
            fo = h5py.File(self.OutputDirectory + "/Selections/Merged/" + name + ".hdf5", "w")
            ref = fo.create_dataset(name, (1), dtype = h5py.ref_dtype)
            for h in self.Merge[name]:
                f = h5py.File(h, "r")
                for key in f.keys(): ref.attrs[key] = f[key].attrs[name]
                self.rm(h)
            self.Merge[name] = []
            self.rm(self.OutputDirectory + "/Selections/" + name)

    @property
    def MakeSelection(self):
        self.__collect__(self.Selections, "Selections")
        if self._condor: return self._Code
        if len(self.Merge) != 0: pass
        elif self.CheckSettings: return False

        self.pth = self.OutputDirectory + "/Selections/"
        for name in self.Selections:
            self.mkdir(self.pth + name)
            f = h5py.File(self.pth + name + "/" + Hash(name) + ".hdf5", "w")
            ref = f.create_dataset("code", (1), dtype = h5py.ref_dtype)
            sel = self._encoder(self._Code["Selections"][name].clone)
            try: ref.attrs[name] = sel
            except ValueError: self._rebuild_code(f["code"][name], self.pth + name)
            inpt = []
            for ev, i in zip(self, range(len(self))):
                if self._StartStop(i) == False: continue
                if self._StartStop(i) == None: break
                if ev.Graph: continue
                inpt.append([name, sel, ev, self.pth])

            if len(inpt) == 0: return self.__merge__
            if self.Threads > 1:
                th = Threading(inpt, self.__compile__, self.Threads, self.chnk)
                th.Title = self.Caller + "::" + name
                th.Start
            else: self.__compile__(inpt, self._MakeBar(len(inpt)))

        if len(self.Merge) == 0: return
        self._Code["Selections"] = {}
        for name in self.Merge:
            self.Merge[name] = [
                self.pth + name + "/" + i for i in self.ls(self.pth + name + "/")
            ]
            self.__merge__
