# distutils: language=c++
# cython: language_level = 3

from libcpp cimport string
from libcpp.map cimport map, pair
from libcpp.vector cimport vector

from AnalysisG.core.tools cimport *
from AnalysisG.core.meta cimport *
from AnalysisG.core.structs cimport *
from cython.parallel import prange
from cython.operator cimport dereference as deref

cdef class MetricTemplate:
    def __cinit__(self): 
        self.mtx = NULL 
        #new metric_template()
        self.root_leaves = {}
        self.root_fx = {}

    def __init__(self): 
        if self.mtx != NULL: return
        print("Forgot to override self.mtx in cython?")
        exit()

    def __dealloc__(self): del self.mtx
    def __name__(self): return env(self.mtx.name)

    @property
    def RunNames(self): 
        cdef map[string, string] o = self.mtx.run_names
        return as_basic_dict(&o)

    @RunNames.setter
    def RunNames(self, dict val): 
        cdef map[string, string] o
        as_map(val, &o)
        self.mtx.run_names = o

    @property
    def Variables(self): 
        cdef vector[string] o = self.mtx.variables
        return env_vec(&o)

    @Variables.setter
    def Variables(self, list val): 
        self.mtx.variables = enc_list(val)

    def Postprocessing(self): pass

    def InterpretROOT(self, str path, list epochs = [], list kfolds = []):
        if not len(self.root_leaves) or not len(self.root_fx):
            self.mtx.failure(b"Failed to interpret!")
            self.mtx.failure(b"Please set the attributes:")
            self.mtx.failure(b"-> dictionary {<tree> : [<leaves>]} : 'root_leaves'")
            self.mtx.failure(b"-> dictionary {<tree> : <fx(class, data, meta)>} or {<tree>.<leaves> : <fx(class, data, meta)>}: 'root_fx'")
            return self
        if   path.endswith("/"): path += "*"
        elif path.endswith(".root"): pass
        else: path += "*"
       
        cdef long ix, il
        cdef int kfold, epoch

        cdef str k, l, kx

        cdef string key, kl, model_name 

        cdef data_t* dt = NULL
        cdef data_t* lxk = NULL

        cdef bool unpause, keep_going

        cdef list leaves = []

        cdef map[string, long] idx_map
        cdef map[string, bool] endx, pause

        cdef vector[string] lsx
        cdef vector[vector[string]] mxf 
        cdef vector[string] epochs_, kfolds_ 

        cdef map[string, vector[string ]] mapfx
        cdef map[string, vector[data_t*]] mapdx

        cdef map[string, map[string, long]] mapx

        cdef dict itx = {}
        cdef dict meta = {}

        cdef tools tl
        epochs_ = [enc(str(epoch)) for epoch in epochs]
        kfolds_ = [enc(str(kfold)) for kfold in kfolds]

        if epochs_.size() + kfolds_.size(): lsx = self.ptr.ls(enc(path), b".root")
        for kfold in prange(lsx.size(), nogil = True, num_threads = 12):
            if not finder(&lsx[kfold], &kfolds_, &epochs_): continue
            idx_map[lsx[kfold]]

        if epochs_.size() + kfolds_.size() and idx_map.size(): pass
        elif epochs_.size() + kfolds_.size() == 0: pass
        else: self.mtx.failure(b"Files for requested epoch and kfolds not found!"); exit()
        lsx = <vector[string]>(list(idx_map))

        unpause = True
        keep_going = True
        cdef dict remap = {enc(l) : self.root_fx[l] for l in self.root_fx}

        for k in self.root_leaves:
            key = enc(k)
            for l in self.root_leaves[k]:
                kx = k + "." + l
                kl = enc(kx)
                if k in self.root_fx:    mapfx[key].push_back(enc(l)); mapdx[key].push_back(NULL)
                elif kx in self.root_fx: mapfx[kl].push_back(enc(l));  mapdx[kl].push_back(NULL)
                else: continue 
                leaves += [l]

        cdef IO iox = IO(path if not lsx.size() else env_vec(&lsx))
        iox.Trees  = list(self.root_leaves)
        iox.Leaves = list(set(leaves))
        iox.__iter__()

        cdef pair[string, data_t*] itr
        cdef map[string, data_t*] dmp = deref(iox.data_ops)
        cdef vector[string] dmi = [itr.first for itr in dmp]

        leaves = []
        kl = b""
        idx_map.clear()
        mapx = iox.ptr.tree_entries
        cdef pair[string, vector[string]] itm
        for itr in dmp: 
            pause[itr.first] = False
            for itm in mapfx:
                leaves.append([itr.second.tree_name, itr.first, itm.first])
                if self.ptr.has_string(&itr.first, itm.first + b"."): 
                    for ix in range(itm.second.size()):
                        if not self.ptr.ends_with(&itr.first, b"." + itm.second[ix]): continue
                        mapdx[itm.first][ix] = itr.second
                        break
                if itr.first == itm.first: mapdx[itm.first][0] = itr.second
                if lxk != NULL: continue
                lxk = itr.second

        mxf = <vector[vector[string]]>(leaves)
        while keep_going:
            itx = {}
            for key in remap:
                itx[key] = {}
                unpause = False
                for dt in mapdx[key]: 
                    if endx[dt.path]: break
                    if pause[dt.path]: break
                    itx[key] |= switch_board(dt)
                    unpause = True
                idx_map[key] += unpause

            keep_going = False
            for il in prange(dmp.size(), nogil = True, num_threads = dmp.size()): 
                dt = dmp[dmi[il]]; ix = dt.index
                if pause[dt.path] or endx[dt.path]: continue
                endx[dt.path] += dt.next() 
                keep_going += not endx[dt.path]

            if not ix:
                kl = deref(lxk.fname)
                leaves =  env(kl).split("/")[-3:]
                kfold = int(leaves[-1].replace(".root", "").replace("kfold-", ""))
                epoch = int(leaves[-3].replace("epoch-", ""))
                model_name = enc(leaves[-2])
                meta = {b"filename" : kl, b"epoch" : epoch, b"kfold" : kfold, b"model_name" : model_name}
                iox.prg.set_description("/".join(env(kl).split("/")[-4:]))
                iox.prg.refresh()
            
            for key in remap: 
                if not len(itx[key]): continue
                remap[key](self, itx[key], {b"index" : idx_map[key]-1} | meta)

            for ix in prange(mxf.size(), nogil = True, num_threads = mxf.size()):
                pause[mxf[ix][1]] = mapx[kl][mxf[ix][0]] == (dmp[mxf[ix][1]].index+1)
     
            unpause = True  
            for ix in prange(mxf.size(), nogil = True): unpause *= pause[mxf[ix][1]]
            if unpause: pause.clear(); idx_map.clear(); kl = b""
            iox.prg.update(1)
            if not keep_going: break

        self.Postprocessing()
        return self
