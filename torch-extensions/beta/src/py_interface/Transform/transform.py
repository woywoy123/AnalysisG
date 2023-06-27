import torch
import os

CUDA_SO   = "/".join(__file__.split("/")[:-2]) + "/libpyc_cuda.so"
FLOAT_SO  = "/".join(__file__.split("/")[:-2]) + "/libpyc_float.so"
TENSOR_SO = "/".join(__file__.split("/")[:-2]) + "/libpyc_tensor.so"
CUDA_SO   = False if not os.path.isfile(CUDA_SO)   else torch.ops.load_library(CUDA_SO)
TENSOR_SO = False if not os.path.isfile(TENSOR_SO) else torch.ops.load_library(TENSOR_SO)
FLOAT_SO  = False if not os.path.isfile(FLOAT_SO)  else torch.ops.load_library(FLOAT_SO)

def __cuda_s(fn):   return getattr(torch.ops.pyc_cuda,   "transform_separate_" + fn)
def __tensor_s(fn): return getattr(torch.ops.pyc_tensor, "transform_separate_" + fn)
def __float_s(fn):  return getattr(torch.ops.pyc_float,  "transform_separate_" + fn)

def __cuda_c(fn):   return getattr(torch.ops.pyc_cuda,   "transform_combined_" + fn)
def __tensor_c(fn): return getattr(torch.ops.pyc_tensor, "transform_combined_" + fn)
def __float_c(fn):  return getattr(torch.ops.pyc_float,  "transform_combined_" + fn)

def __isten(inpt): return len([1 for i in inpt if torch.is_tensor(i)]) == len(inpt)
def __iscu(inpt):
    if CUDA_SO is not None: return False
    return len([1 for i in inpt if i.is_cuda]) == len(inpt)

def __getfn(fn, lst):
    t = __isten(lst)
    if t and __iscu(lst): return __cuda_s(fn)
    if t: return __tensor_s(fn)
    else: return __float_s(fn)

def Px(pt, phi): return __getfn("Px", [pt, phi])(pt, phi)

