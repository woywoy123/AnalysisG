from AnalysisG.Generators.Interfaces import _Interface
from AnalysisG.Notification.nTupler import _nTupler
from AnalysisG._cmodules.SampleTracer import Event
from AnalysisG.SampleTracer import SampleTracer
from AnalysisG.Tools import Threading, Code
from typing import Union
import awkward
import pickle
import uproot
import h5py

def _triggers_(inpt):
    if isinstance(inpt, float): pass
    elif isinstance(inpt, int): pass
    elif isinstance(inpt, str): pass
    elif inpt is None: pass
    else: return False
    return True

def _merge_(inpt, key, start):
    if isinstance(inpt, dict):
        for x in inpt:
            if len(key): k = key + "/" + str(x)
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

def __throot__(inpt, _prgbar = None):
    output = {}
    for i in range(len(inpt)):
        evnt_key, path, code, evnt = inpt[i]
        evnt_key = evnt_key.replace(".", "/")
        if evnt_key not in output: output[evnt_key] = {}
        code = code[evnt_key.split("/")[1]]
        ev = code.InstantiateObject
        ev.__setstate__(evnt)
        for k in path:
            spl, x = k.split("->")[1:-1], ev
            for t in spl:
                if isinstance(x, dict): x = x[t]
                else: x = getattr(x, t)
            k = "/".join(spl).replace("//", "/")
            try: output[evnt_key][k].append(x)
            except KeyError: output[evnt_key][k] = [x]

        indx = evnt_key + "/event_index"
        try: output[indx].append([ev.index])
        except KeyError: output[indx] = [ev.index]
        output = _merge_(output, "", {})
        del evnt
        ev = None
    return [output]

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
        self._clones = {}
        self._loaded = False
        self.root = None
        self._tracer = None

    def __uproot__(self, inpt, OutDir):
        if OutDir.endswith("/"): OutDir += "UNTITLED.root"
        if not OutDir.endswith(".root"): OutDir += ".root"
        if self.root is None:
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

            if tree_name not in data_c: data_c[tree_name] = {}
            if var_name  not in data_c[tree_name]: data_c[tree_name][var_name] = {}
            arr = awkward.highlevel.Array(inpt[tr_n])
            data_c[tree_name][var_name] = awkward.fill_none(arr, [])

        for tr_n in data_c:
            if tr_n not in self.root: self.root[tr_n] = data_c[tr_n]
            else: self.root[tr_n].extend(data_c[tr_n])


    def MakeROOT(self, OutDir: Union[str, None] = None):
        if OutDir is None: OutDir = self.WorkingPath
        else: OutDir = self.abs(OutDir)
        chnks = self.Threads * self.Chunks * self.Threads
        commands = [[], __throot__, self.Threads, self.Chunks]
        for i in self:
            for t, x in i.items():
                var = self._variables[t.replace(".", "/")]
                evnt = x.__getstate__()
                commands[0].append((t, var, self._clones, evnt))
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
        out = {}
        for i in self:
            for k in i:
                if k not in out: out[k] = []
                out[k] += [i[k]]

        output = {k : None for k in out}
        for k in out:
            tmp = {k : []}
            while True:

                try: i  = out[k].pop(0)
                except IndexError: pass

                try: i += out[k].pop(0)
                except IndexError: pass

                tmp[k] += [i]
                if len(out[k]): continue
                if not len(out[k]) and len(tmp[k]) == 1: break
                out[k] += tmp[k]
                tmp[k] = []
            output[k] = tmp[k].pop(0)
        return output

    def __start__(self):
        variables = []
        for i, x in self.DumpThis.items():
            x = [(i + "/" + j).replace(" ","") for j in x]
            x = [i if i.endswith("->") else i+"->" for i in x]
            variables += x

        tree_selection = {}
        if not self._loaded and self._tracer is None:
            self.RestoreTracer()
            self._tracer = self

        for i in variables:
            cont = i.split("->")[0]
            name = cont.split("/")[-1]
            tree = cont.split("/")[0]
            var = tree + i.lstrip(cont)
            if name in self._tracer.ShowSelections:
                if cont in tree_selection: pass
                else: self._FoundSelectionName(name)
            else: self._MissingSelectionName(name)
            try: tree_selection[cont].append(var)
            except KeyError: tree_selection[cont] = [var]
        self._variables = tree_selection
        self._loaded = True

    def preiteration(self):
        return False

    def __iter__(self):
        if not self._loaded: self.__start__()
        self._tracer.GetSelection = True
        self._tracer.GetAll = True
        self._tmpl = {}
        self._restored = {}
        tmp = {i.class_name : i for i in self.rebuild_code(None)}
        lst = [i.hash for i in self._tracer.makelist()]
        self._events = self.Quantize(list(set(lst)), self.Chunks*self.Threads*self.Threads)
        self._events = {k : l for k, l in enumerate(self._events)}
        self._cache_paths = iter(self._events)
        self._clones = {}
        for i in self._variables:
            i = i.replace("/", ".")
            name = i.split(".")[-1]
            if name not in tmp: continue
            self._clones |= {name : tmp[name]}
            self._restored[i] = []
            self._tmpl[i] = 0

        for i in self._variables:
            name = i.split("/")[-1]
            if name in self._clones: continue
            self._MissingSelectionTreeSample(i.replace("/", "."))
        self._itr = len(lst)
        self._bar = self._makebar(self._itr, "Running n-Tupler")[1]
        return self

    def __next__(self):
        if sum(self._tmpl.values()):
            out = {}
            for i, j in self._tmpl.items():
                if not j: continue
                out[i] = self._restored[i].pop(0)
                self._tmpl[i] -= 1
            self._bar.update(1)
            return out

        if self._cache_paths is None: raise StopIteration
        if not len(self._events):
            hashes = None
            self._cache_paths = None
        else: hashes = self._events[next(self._cache_paths)]

        for i in self._restored:
            self._tracer.Tree, self._tracer.SelectionName = i.split(".")
            if hashes is None: continue

            self._tracer.RestoreSelections(hashes)
            if hashes is None: these = self._tracer.makelist(hashes)
            elif len(hashes) > 1: these = self._tracer.makelist(hashes)
            else: these = self._tracer.makelist(hashes)
            self._restored[i] = [k.release_selection() for k in these]
            self._tmpl[i] = len(these)
        self._tracer.FlushSelections(hashes)
        return self.__next__()
