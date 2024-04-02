import inspect
import torch
import os

CUDA_SO   = "/".join(__file__.split("/")[:-3]) + "/libpyc_cuda.so"
FLOAT_SO  = "/".join(__file__.split("/")[:-3]) + "/libpyc_float.so"
TENSOR_SO = "/".join(__file__.split("/")[:-3]) + "/libpyc_tensor.so"
torch.ops.load_library(CUDA_SO)   if os.path.isfile(CUDA_SO) else None
torch.ops.load_library(TENSOR_SO) if os.path.isfile(TENSOR_SO) else None
torch.ops.load_library(FLOAT_SO)  if os.path.isfile(FLOAT_SO) else None

mod_ = __file__.split("/")
cmb_ = mod_[-1].rstrip(".py").lower()
mod_ = mod_[mod_.index("pyc")+1].lower()

__name_c__ = mod_ + "_combined_" + cmb_ + "_"
__name_s__ = mod_ + "_separate_" + cmb_ + "_"

def __router__(
        inpt, get, trig_cmb,
        cmb = __name_c__,
        sgl = __name_s__
    ):

    inpt = [i for i in inpt if i is not None]
    comb = cmb if len(inpt) == trig_cmb else sgl
    comb += get
    fx = {}
    if len([i for i in inpt if torch.is_tensor(i)]) == 0:
        fx = torch.ops.pyc_float
    elif len([i for i in inpt if i.is_cuda]) == len(inpt):
        fx = torch.ops.pyc_cuda
    else:
        inpt = [i.to(device = "cpu") for i in inpt]
        fx = torch.ops.pyc_tensor

    try: return fx.__dict__[comb], inpt
    except KeyError: pass
    return getattr(fx, comb), inpt

class Cartesian:

    def __init__(self):
        pass

    @staticmethod
    def Nu(*args, null = 10e-10):
        null, ten = (args[-1], args[:-1]) if isinstance(args[-1], float) else (null, args)
        name = inspect.currentframe().f_code.co_name
        fn, inpt = __router__(list(ten), name, 5)
        inpt += [null]
        return fn(*inpt)

    @staticmethod
    def NuNu(*args, null = 10e-10):
        null, ten = (args[-1], args[:-1]) if isinstance(args[-1], float) else (null, args)
        name = inspect.currentframe().f_code.co_name
        fn, inpt = __router__(list(ten), name, 6)
        inpt += [null]
        return fn(*inpt)

    @staticmethod
    def Combinatorial(edge_index, batch, pmc, pid, met_xy, gev = False, null = 10e-10, top_up_down = 0.90, w_up_down = 0.90, mT = 172.62*1000, mW = 80.385*1000, mNu = 0):
        output = torch.ops.pyc_cuda.combinatorial(edge_index, batch, pmc, pid, met_xy, mT, mW, mNu, top_up_down, w_up_down, gev, null)
        return {"nu_1f" : output[0], "nu_2f" : output[1], "ms_1f" : output[2], "ms_2f" : output[3], "combi" : output[4], "min" : output[5]}

