# cython: language_level = 3
from libcpp cimport bool

from AnalysisG._cmodules.code import Code
from cytypes cimport code_t

from AnalysisG.Model.LossFunctions import LossFunctions
from torch_geometric.data import Data

from torch.optim.lr_scheduler import ExponentialLR, CyclicLR
from torch.optim import Adam, SGD
import torch


try: from pyc.interface import pyc_cuda as pyx
except:
    try: from pyc.interface import pyc_tensor as pyx
    except: pyx = None

cdef dict polar_edge_mass(edge_index, prediction, pmu):
    if pyx is None: return {}
    pmc = pyx.combined.transform.PxPyPzE(pmu)
    cdef int key
    cdef dict res = pyx.graph.edge_aggregation(edge_index, prediction, pmc)
    cdef dict msk = {key : (res[key]["clusters"] > -1).sum(-1) > 1 for key in res}
    cdef dict out = {}
    for key in res:
        if not key: continue
        if not msk[key].sum(-1): continue
        out[key] = pyx.combined.physics.cartesian.M(res[key]["unique_sum"][msk[key]])
    return out

cdef dict polar_node_mass(edge_index, prediction, pmu):
    if pyx is None: return {}
    cdef int key
    pmc = pyx.combined.transform.PxPyPzE(pmu)
    cdef dict res = pyx.graph.node_aggregation(edge_index, prediction, pmc)
    cdef dict msk = {key : (res[key]["clusters"] > -1).sum(-1) > 1 for key in res}
    cdef dict out = {}
    for key in res:
        if not key: continue
        if not msk[key].sum(-1): continue
        out[key] = pyx.combined.physics.cartesian.M(res[key]["unique_sum"][msk[key]])
    return out

cdef dict cartesian_edge_mass(edge_index, prediction, pmc):
    if pyx is None: return {}
    cdef int key
    cdef dict res = pyx.graph.edge_aggregation(edge_index, prediction, pmc)
    cdef dict msk = {key : (res[key]["clusters"] > -1).sum(-1) > 1 for key in res}
    cdef dict out = {}
    for key in res:
        if not key: continue
        if not msk[key].sum(-1): continue
        out[key] = pyx.combined.physics.cartesian.M(res[key]["unique_sum"][msk[key]])
    return out

cdef dict cartesian_node_mass(edge_index, prediction, pmc):
    if pyx is None: return {}
    cdef int key
    cdef dict res = pyx.graph.node_aggregation(edge_index, prediction, pmc)
    cdef dict msk = {key : (res[key]["clusters"] > -1).sum(-1) > 1 for key in res}
    cdef dict out = {}
    for key in res:
        if not key: continue
        if not msk[key].sum(-1): continue
        out[key] = pyx.combined.physics.cartesian.M(res[key]["unique_sum"][msk[key]])
    return out


cdef class OptimizerWrapper:
    cdef _optim
    cdef _sched
    cdef _model
    cdef _script

    cdef str _path
    cdef str _outdir
    cdef str _run_name
    cdef str _optimizer_name
    cdef str _scheduler_name

    cdef int _epoch
    cdef int _kfold
    cdef bool _train
    cdef dict _state
    cdef dict _optim_params
    cdef dict _sched_params

    def __cinit__(self):
        self._optim = None
        self._sched = None
        self._model = None
        self._script = None

        self._path = ""
        self._outdir = ""
        self._run_name = "untitled"
        self._optimizer_name = ""
        self._scheduler_name = ""
        self._kfold = 0
        self._epoch = 0

        self._train = False
        self._optim_params = {}
        self._sched_params = {}

        self._state = {}

    def __init__(self, initial = None):
        cdef str i
        if initial is None: return
        for i in initial: setattr(self, i, initial[i])

    def setoptimizer(self):
        if not len(self._optim_params): return False
        if self._model is None: return False
        cdef model_param
        if self._script is None: model_param = self._model.parameters()
        else: model_param = self._script.parameters()

        if self._optimizer_name == "ADAM":
            self._optim = Adam(model_param, **self._optim_params)
        elif self._optimizer_name == "SGD":
            self._optim = SGD(model_param, **self._optim_params)
        else: return False
        return True

    def setscheduler(self):
        if self._optim is None: return False
        if not len(self._scheduler_name): return True
        self._sched_params["optimizer"] = self._optim
        if self._scheduler_name == "ExponentialLR":
            self._sched = ExponentialLR(**self._sched_params)
        elif self._scheduler_name == "CyclicLR":
            self._sched = CyclicLR(**self._sched_params)
        else: return False
        return True

    def save(self):
        cdef str _save_path = ""
        _save_path += self._path + "/"
        _save_path += self._run_name + "/"
        _save_path += "Epoch-" + str(self._epoch) + "/"
        _save_path += "kFold-" + str(self._kfold) + "/"
        _save_path += "state.pth"

        self._state["epoch"] = self._epoch
        self._state["optim"] = self._optim.state_dict()
        if self._sched is None: pass
        else: self._state["sched"] = self._optim.state_dict()
        torch.save(self._state, _save_path)

    def load(self):
        cdef str _save_path = ""
        _save_path += self._path + "/"
        _save_path += self._run_name + "/"
        _save_path += "Epoch-" + str(self._epoch) + "/"
        _save_path += "kFold-" + str(self._kfold) + "/"
        _save_path += "state.pth"

        cdef dict c = torch.load(_save_path)
        self._epoch = c["epoch"]
        if self._optim is None: self.setoptimizer()
        self._optim.load_state_dict(c["optim"])
        if self._sched is None: pass
        elif "sched" not in c: pass
        else: self._sched.load_state_dict(c["sched"])
        return _save_path

    def step(self):
        if not self._train: pass
        else: self._optim.step()

    def zero(self):
        if not self._train: pass
        else: self._optim.zero_grad()

    def stepsc(self):
        if self._sched is None: pass
        else: self._sched.step()

    @property
    def optimizer(self): return self._optim

    @property
    def scheduler(self): return self._sched

    @property
    def model(self): return self._model

    @model.setter
    def model(self, val): self._model = val

    @property
    def Path(self): return self._path

    @Path.setter
    def Path(self, str pth): self._path = pth

    @property
    def RunName(self): return self._run_name

    @RunName.setter
    def RunName(self, str val): self._run_name = val

    @property
    def Optimizer(self): return self._optimizer_name

    @Optimizer.setter
    def Optimizer(self, str val):
        if val is None: self._optimizer_name = ""
        else: self._optimizer_name = val

    @property
    def Scheduler(self): return self._scheduler_name

    @Scheduler.setter
    def Scheduler(self, str val):
        if val is None: self._scheduler_name = ""
        else: self._scheduler_name = val

    @property
    def SchedulerParams(self): return self._sched_params

    @SchedulerParams.setter
    def SchedulerParams(self, dict inpt): self._sched_params = inpt

    @property
    def OptimizerParams(self): return self._optim_params

    @OptimizerParams.setter
    def OptimizerParams(self, dict inpt): self._optim_params = inpt

    @property
    def Epoch(self): return self._epoch

    @Epoch.setter
    def Epoch(self, int val): self._epoch = val

    @property
    def Train(self): return self._train

    @Train.setter
    def Train(self, bool val): self._train = val

    @property
    def KFold(self): return self._kfold

    @KFold.setter
    def KFold(self, int val): self._kfold = val


cdef class ModelWrapper:
    cdef str  _run_name
    cdef str  _path

    cdef dict _params
    cdef dict _in_map
    cdef dict _out_map
    cdef dict _loss_map
    cdef dict _class_map
    cdef int  _epoch
    cdef int  _kfold
    cdef bool _train

    # reconstruction of mass
    cdef dict _cartesian
    cdef dict _polar
    cdef dict _fxMap

    cdef public bool failure
    cdef public str error_message

    cdef code_t _code
    cdef _model
    cdef _loss_sum

    def __cinit__(self):
        self._path = ""
        self._run_name = "untitled"
        self._model = None
        self._train = True
        self.failure = False
        self.error_message = ""
        self._params = {}

        self._cartesian = {}
        self._polar = {}
        self._fxMap = {}

        self._kfold = 0
        self._epoch = 0

    def __init__(self, model = None):
        if model is None: return
        self._model = model
        try:self._script = torch.compile(model())
        except RuntimeError: self._script = torch.jit.script(model())
        except: self._script = None
        self.__checkcode__()

    cdef void __checkcode__(self):
        if self._model is None: return
        setattr(self._model, "__params__", self.__params__)
        scriptable = self._model
        co = Code(self._model)
        if not len(self.__params__): pass
        else: co.param_space = self.__params__
        self._code = co.__getstate__()
        self._model = co.InstantiateObject

        try:
            if not len(co.co_vars): pass
            else: self._model = self._model(**self.__params__)
        except Exception as e:
            self.failure = True
            self.error_message = str(e)
            return

        self.failure = False
        self.error_message = ""

        self._out_map = {}
        self._loss_map = {}
        self._class_map = {}

        cdef str i, k
        cdef dict params = self._model.__dict__
        for i, val in params.items():
            try: k = i[:2]
            except IndexError: continue
            if   k == "O_": self._out_map[i[2:]] = val
            elif k == "L_": self._loss_map[i[2:]] = val
            elif k == "C_": self._class_map[i[2:]] = val
            else: pass
        self._in_map = {}
        co = self._model.forward.__code__
        co = co.co_varnames[1: co.co_argcount]
        for i in co: self._in_map[i] = None

        cdef bool cl = False
        for i in self._loss_map:
            try: cl = self._class_map[i]
            except KeyError: cl = False
            self._loss_map[i] = LossFunctions(self._loss_map[i], cl)

    cdef dict __debatch__(self, dict sample, data):
        cdef str i, j, key
        cdef dict out = {}
        cdef dict tmp = {}
        cdef dict loss
        cdef dict mass
        cdef bool skip_m = len(self._fxMap) < 1
        cdef int dim_i

        self._loss_sum = 0
        for i, j in self._out_map.items():
            key = j[2:]
            if self._script is None: pred = self._model.__dict__[j]
            else: pred = getattr(self._script, j)
            if pred is None: continue
            ten = data._slice_dict.get(i)
            if ten is None:
                s = pred.size(0)
                for i in ["i", "batch", "edge_index"]:
                    if sample[i].size(-1) != s: continue
                    ten = data._slice_dict.get(i)
                    break
            data._slice_dict.update({j : ten})
            data._inc_dict.update({j : data._inc_dict.get(i)})
            data.__dict__["_store"][j] = pred

            loss = self._loss_map[key](pred, sample[i])
            try: pass
            except KeyError: continue
            except ValueError: continue
            self._loss_sum += loss["loss"]
            tmp["L_" + key] = loss["loss"]
            tmp["A_" + key] = loss["acc"]
            if skip_m: continue
            if j not in self._fxMap: continue
            mass = {"edge_index" : sample["edge_index"], "prediction" : pred}
            mass.update({"pmu" : torch.cat([sample[key] for key in self._fxMap[j][1]], -1)})
            tmp["M_P_" + j[2:]] = self._fxMap[j][0](**mass)
            try:
                msk = torch.zeros_like(pred);
                for dim_i in range(pred.size(1)): msk[sample[i].view(-1) == dim_i, dim_i] = 1
                msk = msk.to(dtype = torch.long)
            except RuntimeError: continue
            mass["prediction"] = msk
            tmp["M_T_" + j[2:]] = self._fxMap[j][0](**mass)

        tmp["total"] = self._loss_sum
        out["graphs"] = data.to_data_list()
        out.update(tmp)
        return out

    def __call__(self, data):
        cdef str i
        cdef dict inpt
        try: inpt = data.to_dict()
        except AttributeError: inpt = data
        if self._script is None: self._model(**{i : inpt[i] for i in self._in_map})
        else: self._script(**{i : inpt[i] for i in self._in_map})
        self._result = self.__debatch__(inpt, data)
        return self._result

    cpdef match_reconstruction(self, dict sample):
        if pyx is None: return
        cdef str kin_err = ""
        cdef str key, val
        cdef list splits
        cdef dict inpt = self._polar
        self._polar = {}
        self._cartesian = {}
        for key, val in inpt.items():
            if key.startswith("O_"): pass
            else: key = "O_" + key

            if key in self._out_map.values(): pass
            else: kin_err += " :: " + key; continue
            splits = val.replace(" ", "").split("->")
            if len(splits) != 2:
                kin_err += " format error: <coordinate> -> <v1, v2, ...>"
                continue

            val = splits[1]
            if splits[0].lower() == "polar": self._polar[key] = val.split(",")
            else: self._cartesian[key] = val.split(",")

        if len(kin_err): self.error_message += "ERROR: Kinematics -> " + kin_err

        for key, splits in self._cartesian.items():
            val = list(self._out_map)[list(self._out_map.values()).index(key)]
            if val.startswith("N_"): self._fxMap[key] = cartesian_node_mass
            elif val.startswith("E_"): self._fxMap[key] = cartesian_edge_mass
            self._cartesian[key] = [val for val in splits if val in sample]
            self._fxMap[key] = [self._fxMap[key], self._cartesian[key]]

            if len(self._cartesian[key]) == len(splits): continue
            del self._fxMap[key]
            self._cartesian[key] = []

        for key, splits in self._polar.items():
            val = list(self._out_map)[list(self._out_map.values()).index(key)]
            if val.startswith("N_"): self._fxMap[key] = polar_node_mass
            elif val.startswith("E_"): self._fxMap[key] = polar_edge_mass
            self._polar[key] = [val for val in splits if val in sample]
            self._fxMap[key] = [self._fxMap[key], self._polar[key]]

            if len(self._polar[key]) == len(splits): continue
            del self._fxMap[key]
            self._polar[key] = []

    cpdef match_data_model_vars(self, sample):
        cdef str it, key
        cdef dict inpt = sample.to_dict()
        cdef dict mapping = {}
        for it in inpt:
            try: key = it[4:]
            except IndexError: continue
            if key in self._out_map: pass
            else: continue
            mapping[it] = "O_" + key
            del self._out_map[key]

        mapping.update({"M_T_" + key : "O_" + key for key in self._out_map})
        self._out_map = mapping

    def backward(self):
        if not self._train: return False
        try: self._loss_sum.backward()
        except RuntimeError:
            self._loss_sum = sum([k for i, k in self._result.items() if i.startswith("L_")])
            self._loss_sum.backward()
            return True
        return True

    def save(self):
        cdef str _save_path = ""
        _save_path += self._path + "/"
        _save_path += self._run_name + "/"
        _save_path += "Epoch-" + str(self._epoch) + "/"
        _save_path += "kFold-" + str(self._kfold) + "/"
        _save_path += "model_state.pth"

        cdef dict out = {}
        if self._script is None: out = {"epoch": self.Epoch, "model": self._model.state_dict()}
        else:  out = {"epoch": self.Epoch, "model": self._script.state_dict()}
        torch.save(out, _save_path)

    def load(self):
        cdef str _save_path = ""
        _save_path += self._path + "/"
        _save_path += self._run_name + "/"
        _save_path += "Epoch-" + str(self._epoch) + "/"
        _save_path += "kFold-" + str(self._kfold) + "/"
        _save_path += "model_state.pth"

        cdef dict lib = torch.load(_save_path)
        self._epoch = lib["epoch"]
        try: self._model.load_state_dict(state_dict = lib["model"])
        except ValueError: self._failed_model_load()
        self._model.eval()

        if self._script is None: return
        try: self._script.load_state_dict(state_dict = lib["model"])
        except ValueError: self._failed_model_load()
        self._script.eval()

    @property
    def train(self): return self._train

    @train.setter
    def train(self, bool val):
        self._train = val
        if val: self._model.train()
        else: self._model.eval()

        if self._script is None: return
        if val: self._script.train()
        else: self._script.eval()


    @property
    def __params__(self):
        return self._params

    @__params__.setter
    def __params__(self, dict val):
        self._params = val
        self.__checkcode__()

    @property
    def in_map(self): return self._in_map

    @property
    def out_map(self): return self._out_map

    @property
    def loss_map(self): return self._loss_map

    @property
    def class_map(self): return self._class_map

    @property
    def code(self): return Code().__setstate__(self._code)

    @property
    def Epoch(self): return self._epoch

    @Epoch.setter
    def Epoch(self, int val): self._epoch = val

    @property
    def KFold(self): return self._kfold

    @KFold.setter
    def KFold(self, int val): self._kfold = val

    @property
    def Path(self): return self._path

    @Path.setter
    def Path(self, str pth): self._path = pth

    @property
    def RunName(self): return self._run_name

    @RunName.setter
    def RunName(self, str val): self._run_name = val

    @property
    def model(self):
        if self._script is None: return self._model
        return self._script

    @model.setter
    def model(self, mod):
        self._model = mod
        try: self._script = torch.jit.script(mod())
        except: self._script = None
        self.__checkcode__()


    @property
    def device(self): return self._model.device

    @device.setter
    def device(self, val):
        self._model = self._model.to(device = val)
        if self._script is None: return
        self._script = self._script.to(device = val)

    @property
    def KinematicMap(self) -> dict:
        cdef dict output = {}
        output.update(self._polar)
        output.update(self._cartesian)
        return output

    @KinematicMap.setter
    def KinematicMap(self, dict inpt):
        self._cartesian = inpt
        self._polar = inpt
