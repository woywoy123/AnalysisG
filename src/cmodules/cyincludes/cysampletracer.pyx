# distuils: language = c++
# cython: language_level = 3

from cysampletracer cimport CySampleTracer, CyBatch
from cytools cimport *

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

from typing import Union
from tqdm import tqdm
import torch
import pickle
import h5py
import os

def _check_h5(f, str key):
    try: return f.create_dataset(key, (1), dtype = h5py.ref_dtype)
    except ValueError: return f[key]

def _check_sub(f, str key):
    try: return f.create_group(key)
    except ValueError: return f[key]

cdef class Event:
    cdef _event
    cdef _graph
    cdef _selection
    cdef _meta
    cdef CyBatch* ptr
    cdef meta_t m_meta
    cdef dict _internal
    cdef bool _owner

    def __cinit__(self):
        self.ptr = NULL
        self._event = None
        self._selection = None
        self._graph = None
        self._meta = None
        self._owner = False
        self._internal = {}

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
        return self._internal[req]

    def __setattr__(self, key, val):
        self.__getevent__()
        try: setattr(self._event, key, val)
        except AttributeError: pass

        self.__getgraph__()
        try: setattr(self._graph, key, val)
        except AttributeError: pass

        self.__getselection__()
        try: setattr(self._selection, key, val)
        except AttributeError: pass
        self._internal[key] = val

    def release_selection(self):
        self.__getselection__()
        return self._selection

    def release_graph(self):
        self.__getgraph__()
        return self._graph

    def release_event(self):
        self.__getevent__()
        return self._event

    def event_cache_dir(self):
        return map_to_dict(self.ptr.event_dir)

    def graph_cache_dir(self):
        return map_to_dict(self.ptr.graph_dir)

    def selection_cache_dir(self):
        return map_to_dict(self.ptr.selection_dir)

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

        c = Code()
        c.__setstate__(co_.ExportCode())
        cdef code_t co
        cdef pair[string, CyCode*] itr
        for itr in co_.dependency:
            co = itr.second.ExportCode()
            c.AddDependency([co])
        self._event = c.InstantiateObject

        cdef event_t ev = ev_.Export()
        cdef dict pkl = pickle.loads(ev.pickled_data)
        ev.pickled_data = b""
        self._event.__setstate__((pkl, ev))

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
        self._selection = c.InstantiateObject
        self._selection.__setstate__(sel_.Export())

    def __getstate__(self) -> tuple:
        return (self.ptr.meta[0], self.ptr.ExportPickled())

    def __setstate__(self, tuple inpt):
        cdef batch_t b = inpt[1]
        self.m_meta = inpt[0]
        self.ptr = new CyBatch(b.hash)
        self.ptr.Import(&self.m_meta)
        self.ptr.ImportPickled(&b)
        self._owner = True

    @property
    def hash(self): return env(self.ptr.hash)

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
        cdef int entries = 0
        cdef string name = b""
        cdef string name_e = b""
        cdef string name_g = b""

        name_e = enc(self.Tree + "/" + self.EventName)
        if f.count(name_e): entries += f[name_e]
        else: name_e = b""

        name_g = enc(self.Tree + "/" + self.GraphName)
        if f.count(name_g): entries += f[name_g]
        else: name_g = b""

        if name_e.size(): name = name_e
        elif name_g.size(): name = name_g
        else: name = b""

        if not name.size():
            dc = {env(it.first) : it.second for it in f}
            self._nhashes = dc["n_hashes"]
            del dc["n_hashes"]
            return sum([entries for entries in dc.values()])
        return entries

    def __add__(self, SampleTracer other) -> SampleTracer:
        cdef SampleTracer out = self.clone()
        out.ptr.iadd(self.ptr)
        out.ptr.iadd(other.ptr)
        return out

    def __radd__(self, other) -> SampleTracer:
        if other == 0: return self
        return self.__add__(other)

    def __iadd__(self, SampleTracer other) -> SampleTracer:
        self.ptr.iadd(other.ptr)
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
        if not len(self.Tree):
            try: self.Tree = self.ShowTrees[0]
            except IndexError: return True

        if not len(self.EventName):
            try: self.EventName = self.ShowEvents[0]
            except IndexError: pass

        if not len(self.GraphName):
            try: self.GraphName = self.ShowGraphs[0]
            except IndexError: pass

        if not len(self.SelectionName):
            try: self.SelectionName = self.ShowSelections[0]
            except IndexError: pass

        if not len(self.EventName): pass
        else: self.GetEvent = True
        if not len(self.GraphName): pass
        else: self.GetGraph = True
        if not len(self.SelectionName): pass
        else: self.GetSelection = True
        return False

    def DumpTracer(self, retag = None):
        cdef pair[string, meta_t] itr
        cdef pair[string, code_t] itc
        cdef str entry, s_name
        cdef string root_n, h_
        self.ptr.DumpTracer()
        for itr in self._state.root_meta:
            root_n = itr.first
            entry = self.WorkingPath + "Tracer/"
            entry += env(root_n).replace(".root.1", "")
            try: os.makedirs("/".join(entry.split("/")[:-1]))
            except FileExistsError: pass
            entry += ".hdf5"
            f = h5py.File(entry, "a")

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
            dump_this(ref, &self._state.link_event_code)

            ref = _check_h5(f, "link_graph_code")
            dump_this(ref, &self._state.link_graph_code)

            ref = _check_h5(f, "link_selection_code")
            dump_this(ref, &self._state.link_selection_code)

            ref = _check_h5(f, "event_dir")
            count_this(ref, &self._state.event_dir, root_n)

            ref = _check_h5(f, "graph_dir")
            count_this(ref, &self._state.graph_dir, root_n)

            ref = _check_h5(f, "selection_dir")
            count_this(ref, &self._state.selection_dir, root_n)

            ref = _check_h5(f, "event_name_hash")
            dump_hash(ref, &self._state.event_name_hash, root_n)

            ref = _check_h5(f, "graph_name_hash")
            dump_hash(ref, &self._state.graph_name_hash, root_n)

            ref = _check_h5(f, "selection_name_hash")
            dump_hash(ref, &self._state.selection_name_hash, root_n)

            f.close()
        self.ptr.state = export_t()
        self._state = &self.ptr.state

    cdef void _register_hash(self, ref, str key, string root_name):
        cdef str hash_, val, path
        cdef string path_, val_
        cdef CyBatch* batch

        for hash_, val in ref[key + "_name_hash"].attrs.items():
            path = ref[key + "_dir"].attrs[val]
            batch = self.ptr.RegisterHash(enc(hash_), root_name)
            path_, val_ = enc(path), enc(val)
            if key == "event": batch.event_dir[path_] = val_
            elif key == "graph": batch.graph_dir[path_] = val_
            elif key == "selection": batch.selection_dir[path_] = val_

        for hash_, val in ref["link_" + key + "_code"].attrs.items():
            path_, val_ = enc(hash_), enc(val)
            if key == "event": self.ptr.link_event_code[path_] = val_
            elif key == "graph": self.ptr.link_graph_code[path_] = val_
            elif key == "selection": self.ptr.link_selection_code[path_] = val_


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
            for key, i in f5["code"].attrs.items():
                self._set.hashed_code[enc(key)] = _decoder(i)
                self.ptr.AddCode(self._set.hashed_code[enc(key)])
            self._register_hash(f5, "event", event_root)
            self._register_hash(f5, "graph", event_root)
            self._register_hash(f5, "selection", event_root)
            f5.close()

    cdef void _store_objects(self, map[string, vector[obj_t*]] cont, str _type):
        cdef str _path = self.WorkingPath + _type + "Cache/"
        cdef str _short, _daod
        cdef str out_path
        cdef str title = "TRACER::" + _type.upper() + "-SAVE: "
        cdef string _hash

        cdef obj_t* ev
        cdef map[string, string]* daod_dir
        cdef pair[string, vector[obj_t*]] itr
        cdef map[string, vector[string]]* daod_hash
        for itr in cont:
            _daod = env(itr.first).split(":")[0]
            _short = env(itr.first).split(":")[1]
            out_path = _path + _daod
            try: os.makedirs("/".join(out_path.split("/")[:-1]))
            except FileExistsError: pass
            out_path = out_path.rstrip(".root.1") + ".hdf5"
            f = h5py.File(out_path, "a")
            _, bar = self._makebar(itr.second.size(), title + _daod.split("/")[-1] + " (" + _short + ")")
            for obj_t in itr.second:
                _hash = obj_t.Hash()
                dt  = _check_sub(f, env(_hash))
                ref = _check_h5(dt, _short)
                if obj_t.is_event:
                    save_event(ref, &(<CyEventTemplate*>(obj_t)).event)
                    daod_hash = &self._state.event_name_hash
                    daod_dir = &self._state.event_dir

                elif obj_t.is_graph:
                    (<CyGraphTemplate*>(obj_t)).graph.cached = True
                    save_graph(ref, (<CyGraphTemplate*>(obj_t)).Export())
                    daod_hash = &self._state.graph_name_hash
                    daod_dir = &self._state.graph_dir

                elif obj_t.is_selection:
                    save_selection(ref, &(<CySelectionTemplate*>(obj_t)).selection)
                    daod_hash = &self._state.selection_name_hash
                    daod_dir = &self._state.selection_dir
                else: continue

                bar.update(1)
                dump_dir(daod_hash, daod_dir, _daod, _hash, out_path)
            f.close()
            del bar


    cdef _restore_objects(self, str type_, common_t getter, list these_hashes = []):
        cdef str i, file, key
        cdef CyBatch* batch
        cdef pair[string, vector[CyBatch*]] itc
        cdef map[string, vector[CyBatch*]] cache_map

        cdef pair[string, string] its
        cdef map[string, string] dir_map

        self._set.get_all = True
        self._set.search = [enc(i) for i in these_hashes]

        cdef int max_indx = self._set.event_stop
        cdef int idx = 0
        for batch in self.ptr.MakeIterable():
            if type_ == "Event": dir_map = batch.event_dir
            elif type_ == "Graph": dir_map = batch.graph_dir
            elif type_ == "Selection": dir_map = batch.selection_dir
            else: continue
            for its in dir_map: cache_map[its.first].push_back(batch)

            if not max_indx: pass
            elif max_indx > idx: idx += 1
            else: break

        i = type_.upper() + "-READING (" + type_.upper() + "): "
        for itc in cache_map:
            file = env(itc.first)
            try: f = h5py.File(file, "r")
            except FileNotFoundError: continue
            _, bar = self._makebar(itc.second.size(), i + file.split("/")[-1])
            for batch in itc.second:
                if type_ == "Event": batch.event_dir.erase(itc.first)
                elif type_ == "Graph": batch.graph_dir.erase(itc.first)
                elif type_ == "Selection": batch.selection_dir.erase(itc.first)
                else: continue
                dt = f[batch.hash]
                for key in dt.keys():
                    if type_ == "Event": restore_event(dt[key], <event_t*>(&getter))
                    elif type_ == "Graph": restore_graph(dt[key], <graph_t*>(&getter))
                    elif type_ == "Selection": restore_selection(dt[key], <selection_t*>(&getter))
                    else: continue
                    batch.Import(&getter)
                batch.ApplyCodeHash(&self.ptr.code_hashes)
                bar.update(1)
            del bar
            f.close()
        self._set.search.clear()
        self._set.get_all = False
        self.ptr.length()

    def DumpEvents(self):
        self._store_objects(self.ptr.DumpEvents(), "Event")

    def DumpGraphs(self):
        self._store_objects(self.ptr.DumpGraphs(), "Graph")

    def DumpSelections(self):
        self._store_objects(self.ptr.DumpSelections(), "Selection")

    def RestoreEvents(self, list these_hashes = []):
        self._restore_objects("Event", event_t(), these_hashes)

    def RestoreGraphs(self, list these_hashes = []):
        self._restore_objects("Graph", graph_t(), these_hashes)

    def RestoreSelections(self, list these_hashes = []):
        self._restore_objects("Selection", selection_t(), these_hashes)


    def FlushEvents(self, list these_hashes = []):
        cdef str i
        self.ptr.FlushEvents(<vector[string]>[enc(i) for i in these_hashes])

    def FlushGraphs(self, list these_hashes = []):
        cdef str i
        self.ptr.FlushGraphs(<vector[string]>[enc(i) for i in these_hashes])

    def FlushSelections(self, list these_hashes = []):
        cdef str i
        self.ptr.FlushSelections(<vector[string]>[enc(i) for i in these_hashes])

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

    def rebuild_code(self, val: Union[list, str, None]):
        cdef CyCode* c
        cdef string name
        cdef str name_s
        cdef output = []
        cdef pair[string, CyCode*] itc

        if isinstance(val, str):
            name = enc(val)
            if self.ptr.code_hashes.count(name): pass
            else: return output
            c = self.ptr.code_hashes[name]
            co = Code()
            co.__setstate__(c.ExportCode())
            output.append(co)
            return output
        elif isinstance(val, list):
            for name_s in val:
                output += self.rebuild_code(name_s)
            return output
        for itc in self.ptr.code_hashes:
            output += self.rebuild_code(env(itc.first))
        return output

    def ImportSettings(self, settings_t inpt):
        self.ptr.ImportSettings(inpt)

    def ExportSettings(self) -> settings_t:
        return self.ptr.ExportSettings()

    def clone(self):
        return self.__class__()

    def is_self(self, inpt, obj = SampleTracer) -> bool:
        return issubclass(inpt.__class__, obj)

    def makehashes(self) -> dict:
        cdef dict out = {}
        cdef CyBatch* batch
        cdef map[string, vector[string]] ev, gr, sel
        for batch in self.ptr.MakeIterable():
            merge(&ev, &batch.event_dir, batch.hash)
            merge(&gr, &batch.graph_dir, batch.hash)
            merge(&sel, &batch.selection_dir, batch.hash)
        out["event"] = map_vector_to_dict(&ev)
        out["graph"] = map_vector_to_dict(&gr)
        out["selection"] = map_vector_to_dict(&sel)
        return out

    def makelist(self) -> list:
        cdef Event event
        cdef CyBatch* batch
        cdef list output = []
        for batch in self.ptr.MakeIterable():
            event = Event()
            event.ptr = batch
            event.ptr.Import(batch.meta)
            if self._set.getevent: event.__getevent__()
            if self._set.getgraph: event.__getgraph__()
            if self._set.getselection: event.__getselection__()
            output.append(event)
        return output

    def AddEvent(self, event_inpt, meta_inpt = None):
        cdef event_t event
        cdef code_t co
        cdef string name
        cdef dict pkl

        if meta_inpt is not None:
            pkl, event = event_inpt.__getstate__()
            event.pickled_data = pickle.dumps(pkl)
            name = event.event_name

            self.ptr.event_trees[event.event_tree] += 1
            if self.ptr.link_event_code.count(name): pass
            else: self.Event = event_inpt

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
        if isinstance(selection_inpt, dict): self.ptr.AddSelection(selection_inpt, meta_t())
        elif meta_inpt is None: self.ptr.AddSelection(selection_inpt.__getstate__(), meta_t())
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
        if self.ptr.link_event_code.size(): pass
        elif self._Event is None: return
        else: return self._Event
        cdef pair[string, string] its
        for its in self.ptr.link_event_code:
            co = self.rebuild_code(env(its.second))
            if not len(co): return None
            return co[0].InstantiateObject
        return None

    @Event.setter
    def Event(self, event):
        if event is not None: pass
        else: self._Event = None; return
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
        cdef CyCode* code
        cdef dict features
        cdef pair[string, string] its
        cdef pair[string, string] its_

        for its in self.ptr.link_graph_code:
            co = self.rebuild_code(env(its.second))
            if not len(co): return None
            code = self.ptr.code_hashes[its.second]
            features = {}
            for its_ in code.container.param_space:
                if its_.first == code.hash: continue
                if its_.first == b'__state__':
                    features["__state__"] = pickle.loads(its_.second)
                    continue
                c = self.rebuild_code(env(its_.first))
                if not len(c):continue
                features[env(its_.second)] = c[0]
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
        cdef CyCode* code
        cdef pair[string, string] its
        cdef string name
        for its in self.ptr.link_selection_code:
            co = self.rebuild_code(env(its.second))
            if not len(co): continue
            code = self.ptr.code_hashes[its.second]
            name = code.container.class_name
            if env(name) not in self._Selections: pass
            else:
                self.Selections = self._Selections[env(name)]
                continue
            features = {}
            co = co[0].InstantiateObject
            self._Selections[env(code.container.class_name)] = co
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
    def Model(self):
        c = Code()
        try:
            c.__setstate__(self._set.model)
            return c.InstantiateObject
        except: return None

    @Model.setter
    def Model(self, val):
        if val is None: return
        c = Code(val)
        self._set.model = c.__getstate__()

    @property
    def ShowEvents(self) -> list:
        return map_to_list(self.ptr.link_event_code)

    @property
    def ShowGraphs(self) -> list:
        return map_to_list(self.ptr.link_graph_code)

    @property
    def ShowSelections(self) -> list:
        return map_to_list(self.ptr.link_selection_code)

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
        return map_to_list(self.ptr.event_trees)

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
    def WorkingPath(self):
        self.OutputDirectory = self.OutputDirectory
        cdef str path = ""
        path += self.OutputDirectory
        path += self.ProjectName
        return os.path.abspath(path) + "/"

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
    def EventCache(self):
        return self._set.eventcache

    @EventCache.setter
    def EventCache(self, bool val):
        self._set.eventcache = val

    @property
    def GetEvent(self):
        return self._set.getevent

    @GetEvent.setter
    def GetEvent(self, bool val):
        self._set.getevent = val

    @property
    def DataCache(self):
        return self._set.graphcache

    @DataCache.setter
    def DataCache(self, bool val):
        self._set.graphcache = val

    @property
    def GetGraph(self):
        return self._set.getgraph

    @GetGraph.setter
    def GetGraph(self, bool val):
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
    def EventName(self) -> str:
        return env(self._set.eventname)

    @EventName.setter
    def EventName(self, val: Union[str, None]):
        if val is None: val = ""
        self._set.eventname = enc(val)

    @property
    def GraphName(self) -> str:
        return env(self._set.graphname)

    @GraphName.setter
    def GraphName(self, val: Union[str, None]):
        if val is None: val = ""
        self._set.graphname = enc(val)

    @property
    def SelectionName(self) -> str:
        return env(self._set.selectionname)

    @SelectionName.setter
    def SelectionName(self, val: Union[str, None]):
        if val is None: val = ""
        self._set.selectionname = enc(val)

    @property
    def Tree(self) -> str:
        return env(self._set.tree)

    @Tree.setter
    def Tree(self, str val):
        self._set.tree = enc(val)

    @property
    def OutputDirectory(self) -> str:
        return env(self._set.outputdirectory)

    @OutputDirectory.setter
    def OutputDirectory(self, str val):
        if not val.endswith("/"): val += "/"
        val = os.path.abspath(val)
        self._set.outputdirectory = enc(val +"/")

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
        try: torch.cuda.mem_get_info()
        except AssertionError: self._set.device = b"cpu"
        return env(self._set.device)

    @Device.setter
    def Device(self, str val):
        self._set.device = enc(val)
        try: torch.cuda.mem_get_info()
        except AssertionError: self._set.device = b"cpu"

    @property
    def nHashes(self) -> int:
        return self._nhashes

    @property
    def TrainingName(self) -> str:
        return env(self._set.training_name)

    @TrainingName.setter
    def TrainingName(self, str val):
        self._set.training_name = enc(val)

    @property
    def RunName(self) -> str:
        return env(self._set.run_name)

    @RunName.setter
    def RunName(self, str val):
        self._set.run_name = enc(val)

    @property
    def Optimizer(self) -> str:
        return env(self._set.optimizer_name)

    @Optimizer.setter
    def Optimizer(self, str val):
        self._set.optimizer_name = enc(val)

    @property
    def OptimizerParams(self) -> dict:
        cdef pair[string, string] itr
        cdef dict output = {}
        for itr in self._set.optimizer_params:
            output[env(itr.first)] = pickle.loads(itr.second)
        return output

    @OptimizerParams.setter
    def OptimizerParams(self, dict val):
        cdef str key
        cdef string pkl
        self._set.optimizer_params.clear()
        for key in val:
            pkl = pickle.dumps(val[key])
            self._set.optimizer_params[enc(key)] = pkl

    @property
    def Scheduler(self) -> str:
        return env(self._set.scheduler_name)

    @Scheduler.setter
    def Scheduler(self, str val):
        self._set.scheduler_name = enc(val)

    @property
    def SchedulerParams(self) -> dict:
        cdef pair[string, string] itr
        cdef dict output = {}
        for itr in self._set.scheduler_params:
            output[env(itr.first)] = pickle.loads(itr.second)
        return output

    @SchedulerParams.setter
    def SchedulerParams(self, dict val):
        cdef str key
        cdef string pkl
        self._set.scheduler_params.clear()
        for key in val:
            pkl = pickle.dumps(val[key])
            self._set.scheduler_params[enc(key)] = pkl

    @property
    def kFolds(self) -> int:
        return self._set.kfolds

    @kFolds.setter
    def kFolds(self, int val):
        self._set.kfolds = val

    @property
    def kFold(self):
        if not self._set.kfold.size(): return None
        if self._set.kfold.size() != 1: pass
        else: return self._set.kfold[0]
        return self._set.kfold

    @kFold.setter
    def kFold(self, val: Union[int, list, None]):
        cdef vector[int] val_
        if val is None: val_ = []
        elif isinstance(val, int): val_ = [val]
        else: val_ = val
        self._set.kfold = val_

    @property
    def Epoch(self):
        if not self._set.epoch.size(): return None
        else: return self._set.epoch

    @Epoch.setter
    def Epoch(self, val: Union[int, list, dict]):
        cdef int k, i
        self._set.epoch.clear()
        if isinstance(val, int):
            for k in self._set.kfold:
                self._set.epoch[k] = val
        if isinstance(val, list):
            for i, k in zip(val, self.kFold):
                self._set.epoch[k] = i
        if isinstance(val, dict):
            for k, i in val.items():
                self._set.epoch[k] = i

    @property
    def Epochs(self):
        if self._set.epochs == -1: return None
        return self._set.epochs

    @Epochs.setter
    def Epochs(self, int val):
        self._set.epochs = val

    @property
    def KinematicMap(self) -> dict:
        cdef dict output = {}
        cdef pair[string, string] itr
        for itr in self._set.kinematic_map:
            output[env(itr.first)] = env(itr.second)
        return output

    @KinematicMap.setter
    def KinematicMap(self, dict val):
        cdef str key
        cdef string pkl
        self._set.kinematic_map.clear()
        for key in val:
            self._set.kinematic_map[enc(key)] = env(val[key])

    @property
    def ModelParams(self) -> dict:
        cdef pair[string, string] itr
        cdef dict output = {}
        for itr in self._set.model_params:
            output[env(itr.first)] = pickle.loads(itr.second)
        return output

    @ModelParams.setter
    def ModelParams(self, dict val):
        cdef str key
        cdef string pkl
        self._set.model_params.clear()
        for key in val:
            pkl = pickle.dumps(val[key])
            self._set.model_params[enc(key)] = pkl

    @property
    def DebugMode(self) -> bool:
        return self._set.debug_mode

    @DebugMode.setter
    def DebugMode(self, bool val):
        self._set.debug_mode = val

    @property
    def ContinueTraining(self) -> bool:
        return self._set.continue_training

    @ContinueTraining.setter
    def ContinueTraining(self, bool val):
        self._set.continue_training = val

    @property
    def SortByNodes(self) -> bool:
        return self._set.sort_by_nodes

    @SortByNodes.setter
    def SortByNodes(self, bool val):
        self._set.sort_by_nodes = val

    @property
    def EnableReconstruction(self) -> bool:
        return self._set.enable_reconstruction

    @EnableReconstruction.setter
    def EnableReconstruction(self, bool val):
        self._set.enable_reconstruction = val

    @property
    def BatchSize(self):
        return self._set.batch_size

    @BatchSize.setter
    def BatchSize(self, int val):
        self._set.batch_size = val

    @property
    def PlotLearningMetrics(self):
        return self._set.runplotting

    @PlotLearningMetrics.setter
    def PlotLearningMetrics(self, bool val):
        self._set.runplotting = val


