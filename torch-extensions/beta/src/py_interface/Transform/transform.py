import inspect
import torch
import os

CUDA_SO   = "/".join(__file__.split("/")[:-2]) + "/libpyc_cuda.so"
FLOAT_SO  = "/".join(__file__.split("/")[:-2]) + "/libpyc_float.so"
TENSOR_SO = "/".join(__file__.split("/")[:-2]) + "/libpyc_tensor.so"
torch.ops.load_library(CUDA_SO)   if os.path.isfile(CUDA_SO) else None
torch.ops.load_library(TENSOR_SO) if os.path.isfile(TENSOR_SO) else None
torch.ops.load_library(FLOAT_SO)  if os.path.isfile(FLOAT_SO) else None

cmb = "transform_combined_"
sgl = "transform_separate_"

def __router__(inpt, get, trig_cmb, cmb, sgl):
    inpt = [i for i in inpt if i is not None]
    comb = cmb if len(inpt) == trig_cmb else sgl
    comb += get
    fx = {}
    if len([i for i in inpt if torch.is_tensor(i)]) == 0: 
        fx = torch.ops.pyc_float
    elif len([i for i in inpt if i.is_cuda]) == 0:
        fx = torch.ops.pyc_tensor
    else:
        fx = torch.ops.pyc_cuda

    try: return fx.__dict__[comb], inpt
    except KeyError: pass
    return getattr(fx, comb), inpt

def Px(ten1, ten2 = None):
    name = inspect.currentframe().f_code.co_name
    fn, inpt = __router__([ten1, ten2], name, 1, cmb, sgl)
    return fn(*inpt)

def Py(ten1, ten2 = None):
    name = inspect.currentframe().f_code.co_name
    fn, inpt = __router__([ten1, ten2], name, 1, cmb, sgl)
    return fn(*inpt)

def Pz(ten1, ten2 = None):
    name = inspect.currentframe().f_code.co_name
    fn, inpt = __router__([ten1, ten2], name, 1, cmb, sgl)
    return fn(*inpt)

def PxPyPz(ten1, ten2 = None, ten3 = None):
    name = inspect.currentframe().f_code.co_name
    fn, inpt = __router__([ten1, ten2, ten3], name, 1, cmb, sgl)
    return fn(*inpt)

def PxPyPzE(ten1, ten2 = None, ten3 = None, ten4 = None):
    name = inspect.currentframe().f_code.co_name
    fn, inpt = __router__([ten1, ten2, ten3, ten4], name, 1, cmb, sgl)
    return fn(*inpt)

def Pt(ten1, ten2 = None):
    name = inspect.currentframe().f_code.co_name
    fn, inpt = __router__([ten1, ten2], name, 1, cmb, sgl)
    return fn(*inpt)

def Eta(ten1, ten2 = None, ten3 = None):
    name = inspect.currentframe().f_code.co_name
    fn, inpt = __router__([ten1, ten2, ten3], name, 1, cmb, sgl)
    return fn(*inpt)

def Phi(ten1, ten2 = None):
    name = inspect.currentframe().f_code.co_name
    fn, inpt = __router__([ten1, ten2], name, 1, cmb, sgl)
    return fn(*inpt)

def PtEtaPhi(ten1, ten2 = None, ten3 = None):
    name = inspect.currentframe().f_code.co_name
    fn, inpt = __router__([ten1, ten2, ten3], name, 1, cmb, sgl)
    return fn(*inpt)

def PtEtaPhiE(ten1, ten2 = None, ten3 = None, ten4 = None):
    name = inspect.currentframe().f_code.co_name
    fn, inpt = __router__([ten1, ten2, ten3, ten4], name, 1, cmb, sgl)
    return fn(*inpt)

