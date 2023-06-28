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

def __getfn_s(fn, lst):
    t = __isten(lst)
    if t and __iscu(lst): return __cuda_s(fn)
    if t: return __tensor_s(fn)
    return __float_s(fn)

def __getfn_c(fn, lst):
    t = __isten(lst)
    if t and __iscu(lst): return __cuda_c(fn)
    if t: return __tensor_c(fn)
    return __float_c(fn)

def Px(ten1, ten2 = None):
    if ten2 is None: return __getfn_c("Px", [ten1])(ten1)
    return __getfn_s("Px", [ten1, ten2])(ten1, ten2)

def Py(ten1, ten2 = None):
    if ten2 is None: return __getfn_c("Py", [ten1])(ten1)
    return __getfn_s("Py", [ten1, ten2])(ten1, ten2)

def Pz(ten1, ten2 = None):
    if ten2 is None: return __getfn_c("Pz", [ten1])(ten1)
    return __getfn_s("Pz", [ten1, ten2])(ten1, ten2)

def PxPyPz(ten1, ten2 = None, ten3 = None):
    if ten2 is None and ten3 is None: return __getfn_c("PxPyPz", [ten1])(ten1)
    return __getfn_s("PxPyPz", [ten1, ten2, ten3])(ten1, ten2, ten3)

def PxPyPzE(ten1, ten2 = None, ten3 = None, ten4 = None):
    if ten2 is None and ten3 is None and ten4 is None: return __getfn_c("PxPyPzE", [ten1])(ten1)
    return __getfn_s("PxPyPzE", [ten1, ten2, ten3, ten4])(ten1, ten2, ten3, ten4)

def Pt(ten1, ten2 = None):
    if ten2 is None: return __getfn_c("Pt", [ten1])(ten1)
    return __getfn_s("Pt", [ten1, ten2])(ten1, ten2)

def Eta(ten1, ten2 = None, ten3 = None):
    if ten2 is None and ten3 is None: return __getfn_c("Eta", [ten1])(ten1)
    return __getfn_s("Eta", [ten1, ten2, ten3])(ten1, ten2, ten3)

def Phi(ten1, ten2 = None):
    if ten2 is None: return __getfn_c("Phi", [ten1])(ten1)
    return __getfn_s("Phi", [ten1, ten2])(ten1, ten2)

def PtEtaPhi(ten1, ten2 = None, ten3 = None):
    if ten2 is None and ten3 is None: return __getfn_c("PtEtaPhi", [ten1])(ten1)
    return __getfn_s("PtEtaPhi", [ten1, ten2, ten3])(ten1, ten2, ten3)

def PtEtaPhiE(ten1, ten2 = None, ten3 = None, ten4 = None):
    if ten2 is None and ten3 is None and ten4 is None: return __getfn_c("PtEtaPhiE", [ten1])(ten1)
    return __getfn_s("PtEtaPhiE", [ten1, ten2, ten3, ten4])(ten1, ten2, ten3, ten4)
