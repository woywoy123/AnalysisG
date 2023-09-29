from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map, pair
from libcpp cimport bool

from cython.operator cimport dereference
from cytypes cimport code_t, event_t, graph_t, selection_t

from cyevent cimport CyEventTemplate
from cygraph cimport CyGraphTemplate
from cyselection cimport CySelectionTemplate

import pickle

ctypedef fused gen_t:
    int
    bool
    string

ctypedef fused common_t:
    event_t
    graph_t
    selection_t

ctypedef fused obj_t:
    CyEventTemplate
    CyGraphTemplate
    CySelectionTemplate

cdef extern from "../abstractions/abstractions.h" namespace "Tools":
    string encode64(string*) except + nogil
    string decode64(string*) except + nogil
    string Hashing(string) except + nogil

cdef inline string enc(str val): return val.encode("UTF-8")
cdef inline str env(string val): return val.decode("UTF-8")

cdef inline dict _decoder(str inpt):
    cdef string x = enc(inpt)
    return pickle.loads(decode64(&x))

cdef inline string _encoder(inpt): 
    cdef string x = pickle.dumps(inpt)
    return encode64(&x)

cdef inline list map_to_list(map[string, gen_t] inpt):
    cdef pair[string, gen_t] its
    cdef list output = []
    for its in inpt: output.append(env(its.first))
    return output

cdef inline dict map_to_dict(map[string, gen_t] inpt):
    cdef pair[string, gen_t] its
    cdef dict output = {}
    for its in inpt:
        if isinstance(its.second, int): output[env(its.first)] = its.second
        else: output[env(its.first)] = its.second.decode("UTF-8")
    return output


cdef inline dump_this(ref, map[string, string]* inpt):
    cdef pair[string, string] itr
    for itr in dereference(inpt): ref.attrs[itr.first] = itr.second

cdef inline count_this(ref, map[string, string]* inpt, string root_n):
    if not inpt.count(root_n): return
    ref.attrs[root_n] = inpt.at(root_n)

cdef inline dump_hash(ref, map[string, vector[string]]* inpt, string root_n):
    cdef string h_
    for h_ in dereference(inpt)[root_n]: ref.attrs[h_] = root_n


cdef inline dump_dir(
        map[string, vector[string]]* hash_,
        map[string, string]* dir_, str daod, 
        string h_, str path):
    dereference(hash_)[enc(daod)].push_back(h_)
    dereference(dir_)[enc(daod)] = enc(path)


cdef inline merge(map[string, vector[string]]* out, map[string, string]* get, string hash_):
    if not get.size(): return
    cdef pair[string, string] itr
    for itr in dereference(get): dereference(out)[itr.first].push_back(hash_)

cdef inline dict map_vector_to_dict(map[string, vector[string]]* inpt):
    cdef pair[string, vector[string]] itr
    cdef string h
    cdef dict output = {}
    for itr in dereference(inpt):
        output[env(itr.first)] = [env(h) for h in itr.second]
    return output

# ----------------------- cache dumpers -------------------------- #
cdef inline restore_base(ref, common_t* com):
    com.event_name    = enc(ref.attrs["event_name"])
    com.code_hash     = enc(ref.attrs["code_hash"])
    com.event_hash    = enc(ref.attrs["event_hash"])
    com.event_tagging = enc(ref.attrs["event_tagging"])
    com.event_tree    = enc(ref.attrs["event_tree"])
    com.event_root    = enc(ref.attrs["event_root"])
    com.weight        = ref.attrs["weight"]
    com.cached        = ref.attrs["cached"]
    com.pickled_data  = enc(ref.attrs["pickled_data"])
    com.pickled_data  = decode64(&com.pickled_data)
    com.event_index   = ref.attrs["event_index"]

cdef inline save_base(ref, common_t* com):
    ref.attrs["event_name"]    = com.event_name
    ref.attrs["code_hash"]     = com.code_hash
    ref.attrs["event_hash"]    = com.event_hash
    ref.attrs["event_tagging"] = com.event_tagging
    ref.attrs["event_tree"]    = com.event_tree
    ref.attrs["event_root"]    = com.event_root
    ref.attrs["weight"]        = com.weight
    ref.attrs["cached"]        = com.cached
    ref.attrs["pickled_data"]  = encode64(&com.pickled_data)
    ref.attrs["event_index"]   = com.event_index


cdef inline restore_event(ref, event_t* ev):
    restore_base(ref, ev)
    ev.commit_hash   = enc(ref.attrs["commit_hash"])
    ev.deprecated    = ref.attrs["deprecated"]
    ev.event         = ref.attrs["event"]


cdef inline save_event(ref, event_t* ev):
    ev.cached = True
    save_base(ref, ev)
    ref.attrs["commit_hash"] = ev.commit_hash
    ref.attrs["deprecated"]  = ev.deprecated
    ref.attrs["event"]       = ev.event


cdef inline restore_graph(ref, graph_t* gr):
    restore_base(ref, gr)

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


cdef inline save_graph(ref, graph_t gr):
    save_base(ref, &gr)
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


cdef inline restore_selection(ref, selection_t* sel):
    restore_base(ref, sel)

    sel.errors                = _decoder(ref.attrs["errors"])
    sel.pickled_strategy_data = enc(ref.attrs["pickled_strategy_data"])
    sel.pickled_strategy_data = decode64(&sel.pickled_strategy_data)

    sel.cutflow               = _decoder(ref.attrs["cutflow"])
    sel.timestats             = ref.attrs["timestats"]
    sel.all_weights           = ref.attrs["all_weights"]
    sel.selection_weights     = ref.attrs["selection_weights"]

    sel.allow_failure         = ref.attrs["allow_failure"]
    sel._params_              = enc(ref.attrs["_params_"])
    sel._params_              = decode64(&sel._params_)
    sel.selection             = ref.attrs["selection"]


cdef inline save_selection(ref, selection_t* sel):
    save_base(ref, sel)

    ref.attrs["errors"]                = _encoder(sel.errors)
    ref.attrs["pickled_strategy_data"] = sel.pickled_strategy_data

    ref.attrs["cutflow"]               = _encoder(sel.cutflow)
    ref.attrs["timestats"]             = sel.timestats
    ref.attrs["all_weights"]           = sel.all_weights
    ref.attrs["selection_weights"]     = sel.selection_weights

    ref.attrs["allow_failure"]         = sel.allow_failure

    ref.attrs["_params_"]              = encode64(&sel._params_)
    ref.attrs["selection"]             = sel.selection


