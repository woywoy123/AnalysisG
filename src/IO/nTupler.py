from AnalysisG.Generators.Interfaces import _Interface
from AnalysisG.Notification.nTupler import _nTupler
from AnalysisG.SampleTracer import SampleTracer
from AnalysisG.Tools import Threading
from typing import Union
import awkward
import pickle
import uproot
import h5py

class nTupler(_Interface, _nTupler, SampleTracer):

    def __init__(self):
        SampleTracer.__init__(self)
        self.Caller = "n-Tupler"
        _Interface.__init__(self)
        _nTupler.__init__(self)
        self._DumpThis = {}
        self._iterator = {}
        self._variables = {}
        self._events = {}
        self._loaded = False
        self.root = None


    @staticmethod
    def __thmerge__(inpt, _prgbar):
        lock, bar = _prgbar
        out = {}
        for i in range(len(inpt)):
            key, val = inpt[i]
            val = pickle.loads(val)
            if key not in out: out[key] = val
            else: out[key] += val
            inpt[i] = None

            if lock is None:
                if bar is None: continue
                bar.update(1)
                continue
            with lock: bar.update(1)
        return [out]

    @staticmethod
    def __throot__(inpt, _prgbar = None):
        def _triggers_(inpt):
            if isinstance(inpt, float): pass
            elif isinstance(inpt, int): pass
            elif isinstance(inpt, str): pass
            else: return False
            return True

        def _merge_(inpt, key, start):
            if isinstance(inpt, dict):
                for x in inpt:
                    if len(key): k = key + "/" + x
                    else: k = x
                    if not _triggers_(inpt[x]):
                        start = _merge_(inpt[x], k, start)
                        continue
                    if k not in start: start[k] = [[inpt[x]]]
                    else: start[k].append([inpt[x]])

            elif isinstance(inpt, list):
                for i in inpt:
                    if not _triggers_(i):
                        start = _merge_(i, key, start)
                        continue
                    if key not in start: start[key] = [inpt]
                    else: start[key].append(inpt)
                    return start
            return start

        output = {}
        for i in range(len(inpt)):
            evnt_key, path, evnt = inpt[i]
            evnt = pickle.loads(evnt)
            evnt_key = evnt_key.replace(".", "/")
            if evnt_key not in output: output[evnt_key] = {}
            for k in path:
                x = evnt
                spl = k.split("->")[1:-1]
                for t in spl:
                    if isinstance(x, dict): x = x[t]
                    else: x = getattr(x, t)
                k = "/".join(spl)
                try: output[evnt_key][k].append(x)
                except KeyError: output[evnt_key][k] = [x]

            indx = evnt_key + "/event_index"
            try: output[indx].append([evnt.index])
            except KeyError: output[indx] = [evnt.index]
            output = _merge_(output, "", {})
        return [output]



    def __uproot__(self, inpt, OutDir):
        if OutDir.endswith("/"): OutDir += "UNTITLED.root"
        if not OutDir.endswith(".root"): OutDir += ".root"
        if self.root is not None: pass
        else:
            try: self.root = uproot.create(OutDir)
            except OSError: self.root = uproot.recreate(OutDir)

        data_t = {}
        data_c = {}
        for tr_n in inpt:
            spl = tr_n.split("/")
            tree_name = spl[0] + "_" + spl[1]
            var_name = "_".join(tr_n.split("/")[2:])
            if var_name.startswith("CutFlow"):
                tree_name = tree_name + "_CutFlow"
                var_name = var_name[len("CutFlow_"):]
            if tree_name not in data_c:
                data_c[tree_name] = {}
            if var_name not in data_c[tree_name]:
                data_c[tree_name][var_name] = {}
            data_c[tree_name][var_name] = awkward.Array(inpt[tr_n])

        for tr_n in data_c:
            if tr_n not in self.root: self.root[tr_n] = data_c[tr_n]
            else: self.root[tr_n].extend(data_c[tr_n])


    def MakeROOT(self, OutDir: Union[str, None] = None):
        if OutDir is None: OutDir = self.WorkingPath
        else: OutDir = self.abs(OutDir)

        chnks = self.Threads * self.Chunks * 10
        commands = [[], self.__throot__, self.Threads, self.Chunks]
        for i in self:
            for t, x in i.items():
                evn = pickle.dumps(x.release_selection())
                var = self._variables[t.replace(".", "/")]
                commands[0] += [(t, var, evn)]

            if len(commands[0]) < chnks: continue
            th = Threading(*commands)
            th.Start()
            for k in th._lists:
                if k is None: continue
                self.__uproot__(k, OutDir)
            commands[0] = []
            del th
        th = Threading(*commands)
        th.Start()
        for k in th._lists:
            if k is None: continue
            self.__uproot__(k, OutDir)
        commands[0] = []
        del th
        self.root.close()

    def merged(self):
        tmp = []
        chnks = self.Threads * self.Chunks * 10
        commands = [[], self.__thmerge__, self.Threads, self.Chunks]
        for i in self:
            commands[0] += [(t, pickle.dumps(x.release_selection())) for t, x in i.items()]
            if len(commands[0]) < chnks: continue
            th = Threading(*commands)
            th.Start()
            tmp += [k for k in th._lists if k is not None]
            commands[0] = []
            del th

        th = Threading(*commands)
        th.Start()
        tmp += [k for k in th._lists if k is not None]
        commands[0] = []
        del th
        out = {}
        for t in [(key, val) for t in tmp for key, val in t.items()]:
            key, val = t
            if key not in out: out[key] = val
            else: out[key] += val
        return out

    def __start__(self):
        if not self._loaded: self.RestoreTracer()
        variables = []
        for i, x in self._DumpThis.items():
            x = [(i + "/" + j).replace(" ","") for j in x]
            x = [i if i.endswith("->") else i+"->" for i in x]
            variables += x

        tree_selection = {}
        for i in variables:
            cont = i.split("->")[0]
            name = cont.split("/")[-1]
            tree = cont.split("/")[0]
            var = tree + i.lstrip(cont)
            if name in self.ShowSelections: self._FoundSelectionName(name)
            else: self._MissingSelectionName(name)
            try: tree_selection[cont].append(var)
            except KeyError: tree_selection[cont] = [var]
        self._variables = tree_selection
        self._loaded = True

    def __iter__(self):
        if not self._loaded: self.__start__()
        self.GetAll = True
        self._events = self.makehashes()["selection"]
        self.GetAll = False

        self.GetSelection = True
        self._cache_paths = iter(self._events)
        self._restored = {}
        self._tmpl = {}
        for i in self._variables:
            i = i.replace("/", ".")
            self._restored[i] = []
            self._tmpl[i] = 0
        return self

    def __next__(self):
        if sum(self._tmpl.values()):
            out = {}
            for i, j in self._tmpl.items():
                if not j: continue
                out[i] = self._restored[i].pop(0)
                self._tmpl[i] -= 1
            return out
        if self._cache_paths is None: raise StopIteration
        if not len(self._events):
            hashes = None
            self._cache_paths = None
        else:
            hashes = self._events[next(self._cache_paths)]
            self.RestoreSelections(hashes)

        for i in self._restored:
            self.Tree, self.SelectionName = i.split(".")

            if hashes is None: these = self.makelist()
            else: these = self[hashes]
            if these is not False:
                self._restored[i] = these
                self._tmpl[i] = len(self._restored[i])
            else:
                self._MissingSelectionTreeSample(i)
                self._restored[i] = []
                self._tmpl[i] = 0
        return self.__next__()
