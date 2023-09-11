# distuils: language = c++
# cython: language_level = 3

from cysampletracer cimport CySampleTracer, CyBatch

from cyevent cimport CyEventTemplate
from cygraph cimport CyGraphTemplate
from cyselection cimport CySelectionTemplate

from cytypes cimport event_t, graph_t, selection_t, code_t
from cytypes cimport tracer_t, batch_t, meta_t, settings_t, export_t

from AnalysisG.Tools import Code
from cycode cimport CyCode

from AnalysisG._cmodules.SelectionTemplate import SelectionTemplate
from AnalysisG._cmodules.EventTemplate import EventTemplate
from AnalysisG._cmodules.GraphTemplate import GraphTemplate
from AnalysisG._cmodules.MetaData import MetaData

from cython.operator cimport dereference
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map, pair
from libcpp cimport bool

from codecs import decode, encode
from typing import Union
from tqdm import tqdm
import numpy as np
import pickle
import torch
import h5py
import os

cdef string enc(str val): return val.encode("UTF-8")
cdef str env(string val): return val.decode("UTF-8")
cdef str _encoder(inpt): return encode(pickle.dumps(inpt), "base64").decode()
cdef dict _decoder(inpt): return pickle.loads(decode(enc(inpt), "base64"))
def _check_h5(f, str key):
    try: return f.create_dataset(key, (1), dtype = h5py.ref_dtype)
    except ValueError: return f[key]

def _check_sub(f, str key):
    try: return f.create_group(key)
    except ValueError: return f[key]


cdef _event_build(ref, event_t* ev):
    ev.event_name    = enc(ref.attrs["event_name"])
    ev.commit_hash   = enc(ref.attrs["commit_hash"])
    ev.code_hash     = enc(ref.attrs["code_hash"])

    ev.event_hash    = enc(ref.attrs["event_hash"])
    ev.event_tagging = enc(ref.attrs["event_tagging"])
    ev.event_tree    = enc(ref.attrs["event_tree"])
    ev.event_root    = enc(ref.attrs["event_root"])
    ev.pickled_data  = decode(enc(ref.attrs["pickled_data"]), "base64")

    ev.deprecated    = ref.attrs["deprecated"]
    ev.cached        = ref.attrs["cached"]
    ev.event         = ref.attrs["event"]

    ev.event_index   = ref.attrs["event_index"]
    ev.weight        = ref.attrs["weight"]


cdef _graph_save(ref, graph_t* gr):
    ref.attrs["event_name"]      = gr.event_name
    ref.attrs["code_hash"]       = gr.code_hash
    ref.attrs["cached"]          = gr.cached
    ref.attrs["event_index"]     = gr.event_index
    ref.attrs["weight"]          = gr.weight
    ref.attrs["event_hash"]      = gr.event_hash
    ref.attrs["event_tagging"]   = gr.event_tagging

    ref.attrs["event_tree"]      = gr.event_tree
    ref.attrs["event_root"]      = gr.event_root
    ref.attrs["pickled_data"]    = gr.pickled_data

    ref.attrs["train"]           = gr.train
    ref.attrs["evaluation"]      = gr.evaluation
    ref.attrs["validation"]      = gr.validation

    ref.attrs["empty_graph"]     = gr.empty_graph
    ref.attrs["skip_graph"]      = gr.skip_graph
    ref.attrs["self_loops"]      = gr.self_loops

    ref.attrs["errors"]          = _encoder(gr.errors)
    ref.attrs["presel"]          = _encoder(gr.presel)
    ref.attrs["src_dst"]         = _encoder(gr.src_dst)
    ref.attrs["hash_particle"]   = _encoder(gr.hash_particle)
    ref.attrs["graph_feature"]   = _encoder(gr.graph_feature)
    ref.attrs["node_feature"]    = _encoder(gr.node_feature)
    ref.attrs["edge_feature"]    = _encoder(gr.edge_feature)
    ref.attrs["pre_sel_feature"] = _encoder(gr.pre_sel_feature)

    ref.attrs["topo_hash"]       = gr.topo_hash
    ref.attrs["graph"]           = gr.graph

cdef _graph_build(ref, graph_t* gr):
    gr.event_name       = enc(ref.attrs["event_name"])
    gr.code_hash        = enc(ref.attrs["code_hash"])
    gr.cached           = ref.attrs["cached"]
    gr.event_index      = ref.attrs["event_index"]
    gr.weight           = ref.attrs["weight"]
    gr.event_hash       = enc(ref.attrs["event_hash"])
    gr.event_tagging    = enc(ref.attrs["event_tagging"])

    gr.event_tree       = enc(ref.attrs["event_tree"])
    gr.event_root       = enc(ref.attrs["event_root"])
    gr.pickled_data     = decode(enc(ref.attrs["pickled_data"]), "base64")

    gr.train            = ref.attrs["train"]
    gr.evaluation       = ref.attrs["evaluation"]
    gr.validation       = ref.attrs["validation"]

    gr.empty_graph      = ref.attrs["empty_graph"]
    gr.skip_graph       = ref.attrs["skip_graph"]
    gr.self_loops       = ref.attrs["self_loops"]

    gr.errors           = _decoder(ref.attrs["errors"])
    gr.presel           = _decoder(ref.attrs["presel"])
    gr.src_dst          = _decoder(ref.attrs["src_dst"])
    gr.hash_particle    = _decoder(ref.attrs["hash_particle"])
    gr.graph_feature    = _decoder(ref.attrs["graph_feature"])
    gr.node_feature     = _decoder(ref.attrs["node_feature"])
    gr.edge_feature     = _decoder(ref.attrs["edge_feature"])
    gr.pre_sel_feature  = _decoder(ref.attrs["pre_sel_feature"])

    gr.topo_hash        = enc(ref.attrs["topo_hash"])
    gr.graph            = ref.attrs["graph"]

cdef _sel_save(ref, selection_t* sel):
    ref.attrs["event_name"]            = sel.event_name
    ref.attrs["code_hash"]             = sel.code_hash
    ref.attrs["event_hash"]            = sel.event_hash
    ref.attrs["event_index"]           = sel.event_index
    ref.attrs["weight"]                = sel.weight

    ref.attrs["errors"]                = _encoder(sel.errors)
    ref.attrs["event_tagging"]         = sel.event_tagging
    ref.attrs["event_tree"]            = sel.event_tree
    ref.attrs["event_root"]            = sel.event_root
    ref.attrs["pickled_data"]          = sel.pickled_data
    ref.attrs["pickled_strategy_data"] = sel.pickled_strategy_data

    ref.attrs["cutflow"]               = _encoder(sel.cutflow)
    ref.attrs["timestats"]             = sel.timestats
    ref.attrs["all_weights"]           = sel.all_weights
    ref.attrs["selection_weights"]     = sel.selection_weights

    ref.attrs["allow_failure"]         = sel.allow_failure
    ref.attrs["_params_"]              = sel._params_
    ref.attrs["selection"]             = sel.selection

cdef _sel_build(ref, selection_t* sel):
    sel.event_name            = enc(ref.attrs["event_name"])
    sel.code_hash             = enc(ref.attrs["code_hash"])
    sel.event_hash            = enc(ref.attrs["event_hash"])
    sel.event_index           = ref.attrs["event_index"]
    sel.weight                = ref.attrs["weight"]

    sel.errors                = _decoder(ref.attrs["errors"])
    sel.event_tagging         = enc(ref.attrs["event_tagging"])
    sel.event_tree            = enc(ref.attrs["event_tree"])
    sel.event_root            = enc(ref.attrs["event_root"])
    sel.pickled_data          = decode(enc(ref.attrs["pickled_data"]), "base64")
    sel.pickled_strategy_data = decode(enc(ref.attrs["pickled_strategy_data"]), "base64")

    sel.cutflow               = _decoder(ref.attrs["cutflow"])
    sel.timestats             = ref.attrs["timestats"]
    sel.all_weights           = ref.attrs["all_weights"]
    sel.selection_weights     = ref.attrs["selection_weights"]

    sel.allow_failure         = ref.attrs["allow_failure"]
    sel._params_              = enc(ref.attrs["_params_"])
    sel.selection             = ref.attrs["selection"]







cdef class Event:

    cdef CyBatch* ptr
    cdef meta_t m_meta

    cdef _event
    cdef _graph
    cdef _selection
    cdef _meta
    cdef _owner

    def __cinit__(self):
        self.ptr = NULL
        self._event = None
        self._selection = None
        self._graph = None
        self._meta = None
        self._owner = False

    def __init__(self): pass

    def __eq__(self, other):
        try: return self.hash == other.hash
        except: return False

    def __hash__(self):
        return int(self.hash[:8], 0)

    def __dealloc__(self):
        if not self._owner: return
        else: del self.ptr

    def __getattr__(self, req):
        self.__getevent__()
        try: return getattr(self._event, req)
        except AttributeError: pass

        self.__getgraph__()
        try: return getattr(self._graph, req)
        except AttributeError: pass

        self.__getselection__()
        try: return getattr(self._selection, req)
        except AttributeError: pass

        self.__getmeta__()
        try: return getattr(self._meta, req)
        except AttributeError: pass

    def meta(self):
        if self._meta is not None: return self._meta
        else: return dereference(self.ptr.meta)

    def __getmeta__(self):
        if self._meta is not None: return
        if not self.ptr.lock_meta: return
        self._meta = MetaData()
        self._meta.__setstate__(self.ptr.meta[0])

    def __getevent__(self):
        if self._event is not None: return
        cdef CyEventTemplate* ev_ = self.ptr.this_ev
        if ev_ == NULL: self._event = False; return
        cdef CyCode* co_ = ev_.code_link
        if co_ == NULL: print("EVENT -> MISSING CODE!"); return

        cdef event_t ev = ev_.Export()
        cdef code_t  co = co_.ExportCode()
        c = Code()
        c.__setstate__(co)

        cdef pair[string, CyCode*] itr
        for itr in co_.dependency:
            co = itr.second.ExportCode()
            c.AddDependency([co])
        event = c.InstantiateObject

        cdef dict pkl = pickle.loads(ev.pickled_data)
        ev.pickled_data = b""
        event.__setstate__((pkl, ev))
        self._event = event

    def __getgraph__(self):
        if self._graph is not None: return
        cdef CyGraphTemplate* gr_ = self.ptr.this_gr
        if gr_ == NULL: self._graph = False; return
        cdef CyCode* co_ = gr_.code_link
        if co_ == NULL: print("GRAPH -> MISSING CODE!"); return
        c = Code()
        c.__setstate__(co_.ExportCode())
        gr = c.InstantiateObject
        if self._event is None: self._graph = gr()
        elif not self._event: self._graph = gr()
        else: self._graph = gr(self._event)
        self._graph.Import(gr_.Export())


    def __getselection__(self):
        if self._selection is not None: return
        cdef CySelectionTemplate* sel_ = self.ptr.this_sel
        if sel_ == NULL: self._selection = False; return
        cdef CyCode* co_ = sel_.code_link
        if co_ == NULL: print("SELECTION -> MISSING CODE!"); return
        c = Code()
        c.__setstate__(co_.ExportCode())
        sel = c.InstantiateObject
        sel.__setstate__(sel_.Export())
        self._selection = sel

    def __getstate__(self) -> tuple:
        return (self.ptr.meta[0], self.ptr.ExportPickled())

    def __setstate__(self, tuple inpt):
        cdef batch_t b = inpt[1]
        self.m_meta = inpt[0]
        self.ptr = new CyBatch(b.hash)
        self.ptr.Import(&self.m_meta)
        self.ptr.ImportPickled(&b)
        self._owner = True


cdef class SampleTracer:

    cdef CySampleTracer* ptr
    cdef _Event
    cdef _Graph
    cdef dict _Selections
    cdef dict _graph_codes
    cdef int b_end
    cdef int b_start
    cdef int _nhashes
    cdef settings_t* _set
    cdef vector[CyBatch*] batches
    cdef export_t* _state

    def __cinit__(self):
        self.ptr = new CySampleTracer()
        self._set = &self.ptr.settings
        self._state = &self.ptr.state
        self._Event = None
        self._Graph = None
        self._Selections = {}
        self._graph_codes = {}
        self._nhashes = 0
        self.b_start = 0
        self.b_end = 0

    def __dealloc__(self):
        del self.ptr

    def __init__(self):
        pass

    def __getstate__(self) -> tracer_t:
        return self.ptr.Export()

    def __setstate__(self, tracer_t inpt):
        self.ptr.Import(inpt)

    def __getitem__(self, key: Union[list, str]):
        cdef vector[string] inpt;
        cdef str it
        self.preiteration()
        if isinstance(key, str): inpt = [enc(key)]
        else: inpt = [enc(it) for it in key]
        self._set.search = inpt
        cdef list output = self.makelist()
        self._set.search.clear()
        if not len(output): return False
        return output[0] if len(output) == 1 else output

    def __contains__(self, str val) -> bool:
        return self.__getitem__(val) != False

    def __len__(self) -> int:
        cdef map[string, int] f = self.ptr.length()
        cdef pair[string, int] it
        cdef int entries = 0
        cdef str name = ""
        if len(self.Tree): name += self.Tree
        if len(self.EventName): name += "/" + self.EventName
        if not len(name):
            dc = {env(it.first) : it.second for it in f}
            self._nhashes = dc["n_hashes"]
            del dc["n_hashes"]
            return sum([entries for entries in dc.values()])

        for it in f:
            if name not in env(it.first): continue
            entries += it.second
        return entries

    def __add__(self, SampleTracer other) -> SampleTracer:
        cdef SampleTracer tr = self.clone()
        tr.ptr.iadd(self.ptr)
        tr.ptr.iadd(other.ptr)
        return tr

    def __radd__(self, other) -> SampleTracer:
        if other == 0: return self
        return self.__add__(other)

    def __iadd__(self, other) -> SampleTracer:
        if not self.is_self(other): return self
        cdef SampleTracer tr = other
        self.ptr.iadd(tr.ptr)
        return self

    def __iter__(self):
        if self.preiteration(): return self
        self.batches = self.ptr.MakeIterable()
        self.b_end = self.batches.size()
        self.b_start = 0
        return self

    def __next__(self) -> Event:
        if self.b_end == self.b_start: raise StopIteration
        cdef Event event = Event()
        event.ptr = self.batches[self.b_start]
        event.ptr.Import(self.batches[self.b_start].meta)
        self.b_start += 1
        return event


    # ------------------ CUSTOM FUNCTIONS ------------------ #
    def preiteration(self) -> bool:
        if not len(self.EventName):
            try:
                self.EventName = self.ShowEvents[0]
                self.GetEvent = True
            except IndexError: self.GetEvent = False

        if not len(self.GraphName):
            try:
                self.GraphName = self.ShowGraphs[0]
                self.GetGraph = True
            except IndexError: self.GetGraph = False

        if not len(self.SelectionName):
            try:
                self.SelectionName = self.ShowSelections[0]
                self.GetSelection = True
            except IndexError: self.GetSelection = False
        if not len(self.Tree):
            try: self.Tree = self.ShowTrees[0]
            except IndexError: return True

        return False

    def DumpTracer(self, retag = None):
        cdef pair[string, meta_t] itr
        cdef pair[string, code_t] itc
        cdef pair[string, string] its
        cdef str entry, s_name
        cdef string root_n, h_
        self.ptr.DumpTracer()
        for itr in self._state.root_meta:
            root_n = itr.first
            entry = self.WorkingPath + "Tracer/"
            entry += env(root_n).replace(".root.1", "")
            try: os.makedirs("/".join(entry.split("/")[:-1]))
            except FileExistsError: pass

            f = h5py.File(entry + ".hdf5", "a")
            ref = _check_h5(f, "meta")
            s_name = env(itr.second.sample_name)
            if retag is None: pass
            elif retag in s_name.split("|"): pass
            elif not len(s_name): itr.second.sample_name += enc(retag)
            else: itr.second.sample_name += enc("|" + retag)
            self.ptr.AddMeta(itr.second, root_n)
            ref.attrs.update({root_n : _encoder(itr.second)})

            ref = _check_h5(f, "code")
            for itc in self._state.hashed_code:
                ref.attrs[itc.first] = _encoder(itc.second)

            ref = _check_h5(f, "link_event_code")
            for its in self._state.link_event_code:
                ref.attrs[its.first] = its.second

            ref = _check_h5(f, "link_graph_code")
            for its in self._state.link_graph_code:
                ref.attrs[its.first] = its.second

            ref = _check_h5(f, "link_selection_code")
            for its in self._state.link_selection_code:
                ref.attrs[its.first] = its.second

            ref = _check_h5(f, "event_dir")
            if not self._state.event_dir.count(root_n): pass
            else: ref.attrs[root_n] = self._state.event_dir[root_n]

            ref = _check_h5(f, "graph_dir")
            if not self._state.graph_dir.count(root_n): pass
            else: ref.attrs[root_n] = self._state.graph_dir[root_n]

            ref = _check_h5(f, "selection_dir")
            if not self._state.selection_dir.count(root_n): pass
            else: ref.attrs[root_n] = self._state.selection_dir[root_n]


            ref = _check_h5(f, "event_name_hash")
            for h_ in self._state.event_name_hash[root_n]:
                ref.attrs[h_] = root_n

            ref = _check_h5(f, "graph_name_hash")
            for h_ in self._state.graph_name_hash[root_n]:
                ref.attrs[h_] = root_n

            ref = _check_h5(f, "selection_name_hash")
            for h_ in self._state.selection_name_hash[root_n]:
                ref.attrs[h_] = root_n

            f.close()
        self.ptr.state = export_t()
        self._state = &self.ptr.state

    def RestoreTracer(self, dict tracers = {}, sample_name = None):
        cdef str root, f, root_path
        cdef list files_ = []
        cdef list files

        if len(tracers):
            for root, files in tracers.items():
                files_ += [root + "/" + f for f in files if f.endswith(".hdf5")]
        else:
            root_path = self.WorkingPath + "Tracer/"
            for root, _, files in os.walk(root_path):
                files_ += [root + "/" + f for f in files if f.endswith(".hdf5")]

        cdef meta_t meta
        cdef str key, val, path
        cdef CyBatch* batch
        cdef string event_root
        for f in files_:
            f5 = h5py.File(f, "r")
            print("TRACER::RESTORE -> " + f.split("/")[-1])
            key = list(f5["meta"].attrs)[0]
            event_root = enc(key)
            meta = _decoder(f5["meta"].attrs[key])
            if sample_name is None: pass
            elif sample_name not in env(meta.sample_name).split("|"): continue
            self.ptr.AddMeta(meta, event_root)
            for i in f5["code"].attrs.values():
                self.ptr.AddCode(_decoder(i))
                self._set.hashed_code[enc(i)] = _decoder(i)

            for key, val in f5["event_name_hash"].attrs.items():
                path = f5["event_dir"].attrs[val]
                batch = self.ptr.RegisterHash(enc(key), event_root)
                batch.event_dir[enc(path)] = enc(val)

            for key, val in f5["graph_name_hash"].attrs.items():
                path = f5["graph_dir"].attrs[val]
                batch = self.ptr.RegisterHash(enc(key), event_root)
                batch.graph_dir[enc(path)] = enc(val)

            for key, val in f5["selection_name_hash"].attrs.items():
                path = f5["selection_dir"].attrs[val]
                batch = self.ptr.RegisterHash(enc(key), event_root)
                batch.selection_dir[enc(path)] = enc(val)

            for key, val in f5["link_event_code"].attrs.items():
                self.ptr.link_event_code[enc(key)] = enc(val)

            for key, val in f5["link_graph_code"].attrs.items():
                self.ptr.link_graph_code[enc(key)] = enc(val)

            for key, val in f5["link_selection_code"].attrs.items():
                self.ptr.link_selection_code[enc(key)] = enc(val)

            f5.close()

    def DumpEvents(self):
        cdef map[string, vector[event_t*]] events = self.ptr.DumpEvents()
        cdef pair[string, vector[event_t*]] itr
        cdef str entry, bar_
        cdef event_t* ev
        for itr in events:
            entry = self.WorkingPath + "EventCache/" + env(itr.first)
            try: os.makedirs("/".join(entry.split("/")[:-1]))
            except FileExistsError: pass
            f = h5py.File(entry + ".hdf5", "a")
            bar_ = "TRACER::EVENT-DUMPING: " + entry.split("/")[-1]
            print(bar_)
            _, bar = self._makebar(itr.second.size(), "")
            for ev in itr.second:
                ev.cached = True
                dt  = _check_sub(f, env(ev.event_hash))
                ref = _check_h5(dt, env(ev.event_tree + b'.' + ev.event_name))
                ref.attrs.update(dereference(ev))
                self._state.event_name_hash[itr.first].push_back(ev.event_hash)
                self._state.event_dir[itr.first] = enc(entry)
                ev.pickled_data = decode(ev.pickled_data, "base64")
                bar.update(1)
            f.close()
            del bar

    def DumpGraphs(self):
        cdef map[string, vector[graph_t]] graphs = self.ptr.DumpGraphs()
        cdef pair[string, vector[graph_t]] itr
        cdef str entry, bar_
        cdef graph_t gr
        for itr in graphs:
            entry = self.WorkingPath + "GraphCache/" + env(itr.first)
            try: os.makedirs("/".join(entry.split("/")[:-1]))
            except FileExistsError: pass
            f = h5py.File(entry + ".hdf5", "a")
            bar_ = "TRACER::GRAPH-DUMPING: " + entry.split("/")[-1]
            print(bar_)
            _, bar = self._makebar(itr.second.size(), "")
            for gr in itr.second:
                gr.cached = True
                dt  = _check_sub(f, env(gr.event_hash))
                ref = _check_h5(dt, env(gr.event_tree + b'.' + gr.event_name))
                _graph_save(ref, &gr)
                self._state.graph_name_hash[itr.first].push_back(gr.event_hash)
                self._state.graph_dir[itr.first] = enc(entry)
                bar.update(1)
            f.close()
            del bar

    def DumpSelections(self):
        cdef map[string, vector[selection_t]] sel = self.ptr.DumpSelections()
        cdef pair[string, vector[selection_t]] itr
        cdef str entry, bar_
        cdef selection_t se
        for itr in sel:
            entry = self.WorkingPath + "SelectionCache/" + env(itr.first)
            try: os.makedirs("/".join(entry.split("/")[:-1]))
            except FileExistsError: pass
            f = h5py.File(entry + ".hdf5", "a")
            bar_ = "TRACER::SELECTION-DUMPING: " + entry.split("/")[-1]
            print(bar_)
            _, bar = self._makebar(itr.second.size(), "")
            for se in itr.second:
                dt = _check_sub(f, env(se.event_hash))
                ref = _check_h5(dt, env(se.event_tree + b'.' + se.event_name))
                _sel_save(ref, &se)
                self._state.selection_name_hash[itr.first].push_back(se.event_hash)
                self._state.selection_dir[itr.first] = enc(entry)
                bar.update(1)
            f.close()
            del bar

    def RestoreEvents(self, list these_hashes = []):
        if len(these_hashes): self._set.get_all = False
        else: self._set.get_all = True

        cdef str i, file
        self._set.search = [enc(i) for i in these_hashes]

        cdef CyBatch* batch
        cdef pair[string, string] its
        cdef pair[string, vector[CyBatch*]] itc
        cdef map[string, vector[CyBatch*]] cache_map
        for batch in self.ptr.MakeIterable():
            for its in batch.event_dir: cache_map[its.first].push_back(batch)

        cdef event_t event
        for itc in cache_map:
            file = env(itc.first)
            f = h5py.File(file + ".hdf5", "r")
            bar_ = "EVENT-READING (EventCache): " + file.split("/")[-1]
            print(bar_)
            _, bar = self._makebar(itc.second.size(), "")
            for batch in itc.second:
                batch.event_dir.erase(itc.first)
                dt = f[batch.hash]
                for i in dt.keys():
                    event = event_t()
                    _event_build(dt[i], &event)
                    batch.Import(&event)
                batch.ApplyCodeHash(&self.ptr.code_hashes)
                bar.update(1)
            del bar
            f.close()
        self._set.search.clear()
        self._set.get_all = False
        self.ptr.length()


    def RestoreGraphs(self, list these_hashes = []):
        if len(these_hashes): self._set.get_all = False
        else: self._set.get_all = True

        cdef str i, file
        self._set.search = [enc(i) for i in these_hashes]

        cdef CyBatch* batch
        cdef pair[string, string] its
        cdef pair[string, vector[CyBatch*]] itc
        cdef map[string, vector[CyBatch*]] cache_map
        for batch in self.ptr.MakeIterable():
            for its in batch.graph_dir: cache_map[its.first].push_back(batch)

        cdef graph_t graph
        for itc in cache_map:
            file = env(itc.first)
            f = h5py.File(file + ".hdf5", "r")
            bar_ = "GRAPH-READING: (GraphCache)" + file.split("/")[-1]
            print(bar_)
            _, bar = self._makebar(itc.second.size(), "")
            for batch in itc.second:
                batch.graph_dir.erase(itc.first)
                dt = f[batch.hash]
                for i in dt.keys():
                    graph = graph_t()
                    _graph_build(dt[i], &graph)
                    batch.Import(&graph)
                batch.ApplyCodeHash(&self.ptr.code_hashes)
                bar.update(1)
            del bar
            f.close()
        self._set.search.clear()
        self._set.get_all = False
        self.ptr.length()

    def RestoreSelections(self, list these_hashes = []):
        if len(these_hashes): self._set.get_all = False
        else: self._set.get_all = True

        cdef str i, file
        self._set.search = [enc(i) for i in these_hashes]

        cdef CyBatch* batch
        cdef pair[string, string] its
        cdef pair[string, vector[CyBatch*]] itc
        cdef map[string, vector[CyBatch*]] cache_map
        for batch in self.ptr.MakeIterable():
            for its in batch.selection_dir: cache_map[its.first].push_back(batch)

        cdef selection_t sel
        for itc in cache_map:
            file = env(itc.first)
            f = h5py.File(file + ".hdf5", "r")
            bar_ = "SELECTION-READING: (SelectionCache) " + file.split("/")[-1]
            _, bar = self._makebar(itc.second.size(), "")
            for batch in itc.second:
                batch.selection_dir.erase(itc.first)
                dt = f[batch.hash]
                for i in dt.keys():
                    sel = selection_t()
                    _sel_build(dt[i], &sel)
                    batch.Import(&sel)
                batch.ApplyCodeHash(&self.ptr.code_hashes)
                bar.update(1)
            del bar
            f.close()
        self._set.search.clear()
        self._set.get_all = False
        self.ptr.length()

    def _makebar(self, inpt: Union[int], CustTitle: Union[None, str] = None):
        _dct = {}
        _dct["desc"] = f'Progress {self.Caller}' if CustTitle is None else CustTitle
        _dct["leave"] = True
        _dct["colour"] = "GREEN"
        _dct["dynamic_ncols"] = True
        _dct["total"] = inpt
        return (None, tqdm(**_dct))

    def trace_code(self, obj) -> code_t:
        cdef code_t co = Code(obj).__getstate__()
        self.ptr.AddCode(co)
        return co

    def rebuild_code(self, val: Union[list, str]):
        cdef CyCode* c
        cdef string name
        cdef str name_s
        cdef output = []
        if isinstance(val, str):
            name = enc(val)
            if not self.ptr.code_hashes.count(name):
                return output
            c = self.ptr.code_hashes[name]
            co = Code()
            co.__setstate__(c.ExportCode())
            output.append(co)
            return output
        elif isinstance(val, list):
            for name_s in val:
                output += self.rebuild_code(name_s)
            return output

    def ImportSettings(self, settings_t inpt):
        self.ptr.ImportSettings(inpt)

    def ExportSettings(self) -> settings_t:
        return self.ptr.ExportSettings()

    def clone(self):
        return self.__class__()

    def is_self(self, inpt, obj = SampleTracer) -> bool:
        return issubclass(inpt.__class__, obj)

    def makelist(self) -> list:
        cdef vector[CyBatch*] evnt = self.ptr.MakeIterable()
        cdef CyBatch* batch
        cdef Event event
        cdef list output = []
        for batch in evnt:
            event = Event()
            event.__setstate__((batch.meta[0], batch.ExportPickled()))
            output.append(event)
        return output

    def AddEvent(self, event_inpt, meta_inpt = None):
        cdef event_t event
        cdef code_t co
        cdef string name
        cdef tuple get

        if meta_inpt is not None:
            get = event_inpt.__getstate__()
            event = get[1]
            event.pickled_data = pickle.dumps(get[0])
            name = event.event_name

            self.ptr.event_trees[event.event_tree] += 1
            if not self.ptr.link_event_code.count(name): self.Event = event_inpt
            self.ptr.AddEvent(event, meta_inpt.__getstate__())
            return

        cdef str g
        cdef dict ef
        cdef list evnts = [ef for g in event_inpt for ef in event_inpt[g].values()]
        for ef in evnts: self.AddEvent(ef["Event"], ef["MetaData"])


    def AddGraph(self, graph_inpt, meta_inpt = None):
        if graph_inpt is None: return
        if isinstance(graph_inpt, dict): self.ptr.AddGraph(graph_inpt, meta_t())
        elif meta_inpt is None: self.ptr.AddGraph(graph_inpt.__getstate__(), meta_t())
        else: self.ptr.AddGraph(graph_inpt.__getstate__(), meta_inpt.__getstate__())

    def AddSelections(self, selection_inpt, meta_inpt = None):
        if selection_inpt is None: return
        if meta_inpt is None: self.ptr.AddSelection(selection_inpt.__getstate__(), meta_t())
        else: self.ptr.AddSelection(selection_inpt.__getstate__(), meta_inpt.__getstate__())

    def SetAttribute(self, fx, str name) -> bool:
        if name in self._graph_codes: return False
        self._graph_codes[name] = fx
        if self._Graph is None: return True
        self.Graph = self._Graph
        self._graph_codes = {}
        return True

    @property
    def Event(self):
        if self._Event is not None: return self._Event
        cdef CyCode* code
        cdef pair[string, string] its
        for its in self.ptr.link_event_code:
            co = self.rebuild_code(env(its.second))
            if not len(co): return None
            return co[0].InstantiateObject
        return None

    @Event.setter
    def Event(self, event):
        try: event = event()
        except: pass
        cdef code_t co
        cdef string name = enc(event.__name__())
        if type(event).__module__.endswith("cmodules.code"):
            co = event.code

            if self.ptr.code_hashes.count(co.hash): return
            self.ptr.link_event_code[name] = co.hash
            self.ptr.code_hashes[co.hash] = new CyCode()
            self.ptr.code_hashes[co.hash].ImportCode(co)
            return

        if not self.is_self(event, EventTemplate):
            return

        cdef map[string, code_t] deps = {}
        for o in event.Objects.values():
            co = self.trace_code(o)
            deps[co.hash] = co

        co = self.trace_code(event)
        self.ptr.link_event_code[name] = co.hash
        self.ptr.code_hashes[co.hash].AddDependency(deps)
        self._Event = event

    @property
    def Graph(self):
        if self._Graph is not None: return self._Graph
        cdef pair[string, string] its
        cdef CyCode* code
        cdef dict features
        for its in self.ptr.link_graph_code:
            co = self.rebuild_code(env(its.second))
            if not len(co): return None
            code = self.ptr.code_hashes[its.second]
            features = {}
            for its in code.container.param_space:
                if its.first == code.hash: continue
                if its.first == b'__state__':
                    features["__state__"] = pickle.loads(its.second)
                    continue
                c = self.rebuild_code(env(its.first))
                if not len(c):continue
                features[env(its.second)] = c[0]
            co = co[0].InstantiateObject
            setattr(co, "code", features)
            return co
        return None

    @Graph.setter
    def Graph(self, graph):
        try: graph = graph()
        except: pass
        if not self.is_self(graph, GraphTemplate): return
        cdef code_t co
        cdef graph_t gr
        cdef string name = enc(graph.__name__())
        cdef str name_
        if type(graph).__module__.endswith("cmodules.code"):
            co = graph.code

            if self.ptr.code_hashes.count(co.hash): return
            self.ptr.link_graph_code[name] = co.hash
            self.ptr.code_hashes[co.hash] = new CyCode()
            self.ptr.code_hashes[co.hash].ImportCode(co)
            return

        for name_, c_ in self._graph_codes.items():
            graph.__scrapecode__(c_, name_)

        co = self.trace_code(graph)
        self.ptr.link_graph_code[name] = co.hash
        cdef CyCode* c = self.ptr.code_hashes[co.hash]
        c.container.param_space[b'__state__'] = pickle.dumps(graph.__getstate__())
        for name_, o in graph.code.items():
            co = o.__getstate__()
            c.container.param_space[co.hash] = enc(name_)
            self.ptr.AddCode(co)
        self._Graph = graph

    @property
    def Selections(self):
        return self._Selections

    @Selections.setter
    def Selections(self, selection):
        try: selection = selection()
        except: pass
        if not self.is_self(selection, SelectionTemplate): return
        cdef code_t co = self.trace_code(selection)
        self.ptr.link_selection_code[co.class_name] = co.hash
        self._Selections[env(co.class_name)] = selection

    @property
    def ShowEvents(self) -> list:
        cdef list out = []
        cdef pair[string, string] it
        cdef map[string, string] ev = self.ptr.link_event_code
        for it in ev: out.append(env(it.first))
        return out

    @property
    def ShowGraphs(self) -> list:
        cdef list out = []
        cdef pair[string, string] it
        cdef map[string, string] ev = self.ptr.link_graph_code
        for it in ev: out.append(env(it.first))
        return out

    @property
    def ShowSelections(self) -> list:
        cdef list out = []
        cdef pair[string, string] it
        cdef map[string, string] ev = self.ptr.link_selection_code
        for it in ev: out.append(env(it.first))
        return out

    @property
    def ShowLength(self) -> dict:
        cdef pair[string, int] it
        cdef dict output = {}
        for it in self.ptr.length():
            if env(it.first) == "n_hashes": self._nhashes = it.second
            else: output[env(it.first)] = it.second
        return output

    @property
    def ShowTrees(self) -> list:
        cdef pair[string, int] it
        cdef map[string, int] ev = self.ptr.event_trees
        cdef list out = []
        for it in ev: out.append(env(it.first))
        return out

    @property
    def Files(self) -> dict:
        cdef dict output = {}
        cdef string k
        cdef pair[string, vector[string]] it
        for it in self._set.files: output[env(it.first)] = [env(k) for k in  it.second]
        return output

    @Files.setter
    def Files(self, val: Union[str, list, dict, None]):
        cdef dict Files = {}
        cdef str key, k
        if val is None: self._set.files.clear()
        elif isinstance(val, str): Files[""] = [val]
        elif isinstance(val, list): Files[""] = val
        elif isinstance(val, dict): Files.update(val)
        for key in Files:
            for k in Files[key]: self._set.files[enc(key)].push_back(enc(k))

    @property
    def Threads(self) -> int:
        return self._set.threads

    @Threads.setter
    def Threads(self, int val):
        self._set.threads = val

    @property
    def GetAll(self):
        return self._set.get_all

    @GetAll.setter
    def GetAll(self, bool val):
        self._set.get_all = val

    @property
    def GetEvent(self) -> bool:
        return self._set.getevent

    @GetEvent.setter
    def GetEvent(self, bool val):
        self._set.getevent = val

    @property
    def EventCache(self):
        return self._set.eventcache

    @EventCache.setter
    def EventCache(self, bool val):
        if val: self._set.eventcache = val
        self._set.getevent = val

    @property
    def GetGraph(self) -> bool:
        return self._set.getgraph

    @GetGraph.setter
    def GetGraph(self, bool val):
        self._set.getgraph = val

    @property
    def DataCache(self):
        return self._set.graphcache

    @DataCache.setter
    def DataCache(self, bool val):
        if val: self._set.graphcache = val
        self._set.getgraph = val

    @property
    def GetSelection(self) -> bool:
        return self._set.getselection

    @GetSelection.setter
    def GetSelection(self, bool val):
        self._set.getselection = val

    @property
    def ProjectName(self) -> str:
        return env(self._set.projectname)

    @ProjectName.setter
    def ProjectName(self, str val):
        self._set.projectname = enc(val)

    @property
    def WorkingPath(self):
        return os.path.abspath(self.OutputDirectory + self.ProjectName) + "/"

    @property
    def SampleMap(self):
        cdef dict output = {}
        cdef string i
        cdef pair[string, vector[string]] itr
        for itr in self._set.samplemap:
            output[env(itr.first)] = [env(i) for i in itr.second]
        return output

    @SampleMap.setter
    def SampleMap(self, val: Union[str, list, dict]):
        cdef dict state = self.SampleMap
        cdef str i, f
        if isinstance(val, str): state[val] = []
        elif isinstance(val, list):
            if "" in state: state[""] += val
            else: state[""] = val
        elif isinstance(val, dict):
            for i in val:
                if i not in state: state[i] = []
                state[i] += [k for k in val[i]]

        for i in state:
            state[i] = list(set(state[i]))
            self._set.samplemap[enc(i)] = [enc(f) for f in state[i]]


    @property
    def EventName(self) -> str:
        return env(self._set.eventname)

    @EventName.setter
    def EventName(self, val: Union[str, None]):
        if val is None: val = "NULL"
        self._set.eventname = enc(val)

    @property
    def GraphName(self) -> str:
        return env(self._set.graphname)

    @GraphName.setter
    def GraphName(self, val: Union[str, None]):
        if val is None: val = "NULL"
        self._set.graphname = enc(val)

    @property
    def SelectionName(self) -> str:
        return env(self._set.selectionname)

    @SelectionName.setter
    def SelectionName(self, val: Union[str, None]):
        if val is None: val = "NULL"
        self._set.selectionname = enc(val)

    @property
    def Tree(self) -> str:
        return env(self._set.tree)

    @Tree.setter
    def Tree(self, str val):
        self._set.tree = enc(val)

    @property
    def ProjectName(self) -> str:
        return env(self._set.projectname)

    @ProjectName.setter
    def ProjectName(self, str val):
        self._set.projectname = enc(val)

    @property
    def OutputDirectory(self) -> str:
        return env(self._set.outputdirectory)

    @OutputDirectory.setter
    def OutputDirectory(self, str val):
        if not val.endswith("/"): val += "/"
        val = os.path.abspath(val)
        self._set.outputdirectory = enc(val)

    @property
    def Caller(self) -> str:
        return env(self.ptr.caller)

    @Caller.setter
    def Caller(self, str val):
        self.ptr.caller = enc(val.upper())

    @property
    def EventStart(self):
        return self._set.event_start

    @EventStart.setter
    def EventStart(self, x: Union[int, None]):
        if x is None: x = 0
        self._set.event_start = x

    @property
    def EventStop(self):
        if self._set.event_stop == 0: return None
        return self._set.event_stop

    @EventStop.setter
    def EventStop(self, x: Union[int, None]):
        if x is None: x = 0
        self._set.event_stop = x

    @property
    def Verbose(self):
        return self._set.verbose

    @Verbose.setter
    def Verbose(self, int val):
        self._set.verbose = val

    @property
    def Chunks(self):
        return self._set.chunks

    @Chunks.setter
    def Chunks(self, int val):
        self._set.chunks = val

    @property
    def EnablePyAMI(self):
        return self._set.enable_pyami

    @EnablePyAMI.setter
    def EnablePyAMI(self, bool val):
        self._set.enable_pyami = val

    @property
    def Device(self):
        return env(self._set.device)

    @Device.setter
    def Device(self, str val):
        self._set.device = enc(val)

    @property
    def nHashes(self) -> int:
        return self._nhashes
