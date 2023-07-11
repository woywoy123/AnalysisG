from AnalysisG.Generators.Interfaces import _Interface
from AnalysisG.Notification.nTupler import _nTupler
from AnalysisG.Tracer import SampleTracer
from typing import Union
import h5py
import uproot

class container(SampleTracer):

    def __init__(self):
        self.Tree = None
        self.SelectionName = None
        self.Variable = None
        self.FilePath = None
        self._h5 = []

    def _strip(self, inpt):
        msg = "Syntax Error! Format is '<selection name> -> <variable> -> <key>'"
        if "->" not in inpt: return msg
        self.SelectionName = inpt.split("->")[0].strip(" ")
        self.Variable = inpt.split("->")[1].strip(" ")

    @property
    def hdf5(self):
        try:
            self._h5 = h5py.File(self.FilePath, "r")
            self._rebuild_code("/".join(self.FilePath.split("/")[:-1]) + "/code/", self._h5)
            self._h5 = self._h5[self.SelectionName]
        except KeyError: return True

    def __iter__(self):
        self._it = iter(self._h5.attrs)
        return self

    def __next__(self):
        return self._h5.attrs[next(self._it)]

    def __len__(self):
        return len(self._h5.attrs)

class nTupler(_Interface, _nTupler):

    def __init__(self, inpt = None):
        self.Caller = "nTupler"
        self.Verbose = 3
        self.Files = []
        self.__it = None
        self._it = None
        self._DumpThis = {}
        self._Container = []
        if inpt is None: return
        self.InputSelection(inpt)

    def __make_list__(self, inpt, key = None):
        if isinstance(inpt, dict): pass
        elif isinstance(inpt, list): return {key : inpt}
        else: return {key : [inpt]}

        x = {}
        for i in inpt:
            k = key + "/" + i if key is not None else i
            x.update(self.__make_list__(inpt[i], k))
        return x
    
    def __uproot__(self, out, sel):
        tree = self.i.Tree
        _o = out if out is not None else "/".join(self.__it.FilePath.split("/")[:-1])
        _o = self.AddTrailing(_o, "/") + sel + ".root"
        self.output = self.__make_list__(self.output)
        Len = {}
        for i in self.output:
            l = len(self.output[i])
            if l not in Len: Len[l] = []
            Len[l] += [{i : self.output[i]}]

        self.output = {}
        for i in Len:
            for k in Len[i]:
                try: x = uproot.create(_o)
                except OSError: x = uproot.update(_o)
                wr = tree 
                wr += "" if len(Len) == 1 else "_" + next(iter(k))
                x[wr] = k
        x.close()

    def Write(self, OutDir: Union[str, None] = None):
        self.output = {}
        sel = None
        self.i = None
        for i in self:
            if i.Tree != self.__it.Tree: continue
            if sel is None: sel = self.__it.SelectionName
            if sel != self.__it.SelectionName: 
                self.__uproot__(OutDir, sel)

            sel = self.__it.SelectionName
            pth = self.__it.Variable
            var = i.__dict__[pth]
            self.i = i
            if isinstance(var, dict):
                if pth not in self.output: self.output[pth] = {}
                self.output[pth] = self.MergeData(self.output[pth], var)
            else: 
                if pth not in self.output: 
                    self.output[pth] = var
                    continue
                self.output[pth] = var
        self.__uproot__(OutDir, sel)

    def __scrape__(self, file):
        for tree in self._DumpThis:
            for entry in self._DumpThis[tree]:
                self._Container.append(container())
                self._Container[-1].Tree = tree
                self._Container[-1].FilePath = file
                res = self._Container[-1]._strip(entry)
                skip = self._Container[-1].hdf5
                if skip is True:
                    del self._Container[-1]
                    continue
                if res is None: continue
                else: self.Warning(res)

    @property
    def __Dumps__(self):
        self._Container = []
        files = self.DictToList(self.Files)
        for file in files: self.__scrape__(file)

    def __iter__(self):
        self.__Dumps__
        self._it = iter(self._Container)
        return self

    def __next__(self):
        try:
            if self.__it is None: raise StopIteration
            self._br.update(1)
            return SampleTracer._decoder(next(self.__it))
        except StopIteration:
            self.__it = next(self._it)
            self._br = None
            msg = "nTupler::" + self.__it.SelectionName + " -> " + self.__it.Variable
            _, self._br = self._MakeBar(len(self.__it), msg)
            self.__it = iter(self.__it)
            return self.__next__()
