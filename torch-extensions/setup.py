import os 
from pathlib import Path
from setuptools import setup

cuda = False if os.path.isfile("cpu.txt") else True
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import BuildExtension
if cuda: from torch.utils.cpp_extension import CUDAExtension

def _check(inpt, ext = [".h", ".cxx", ".cu"]): 
    if isinstance(ext, str): return inpt.endswith(ext)
    return len([i for i in ext if i in inpt and not ".swp" in inpt]) > 0

def _clean(inpt):
    if isinstance(inpt, list): return [j for i in inpt for j in _clean(i)]
    if not _check(inpt, ["#include"]): return []
    if _check(inpt, ["<", ">"]): return []
    inpt = inpt.replace("\n", "")
    inpt = inpt.replace(" ", "")
    inpt = inpt.replace('"', "")
    return [inpt.split("#include")[-1]]

def _getPath(inpt, src):
    src = src.split("/")[:-1] + [inpt]
    pth = os.path.abspath("/".join(src))
    pth = pth.split(os.getcwd())[-1][1:]
    return pth 

def _reference(inpt, ref):
    try: 
        _ref = _clean(open(inpt).readlines())
        if len(_ref) == 0: return False  
    except: return False
    return ref in [_getPath(i, inpt) for i in _ref]   


def _recursive(inpt):
    if isinstance(inpt, str): inpt = [inpt] if _check(inpt) else []
    else: inpt = [i for i in inpt if _check(i)]
  
    out = []
    out += list(inpt)
    f = {}
    for i in inpt:
        try: v = open(i).readlines()
        except: continue
        f[i] = _clean(v)
    f = {_getPath(k, i) : i for i in f for k in f[i]}
    for j in list(f): out += [j] + _recursive(j)
    
    this_f = inpt.pop()
    for p in Path("/".join(this_f.split("/")[:2])).rglob("*"):
        if _check(str(p), ".cu") or _check(str(p), ".h") or _check(str(p), ".cxx"): pass
        else: continue
        if "Shared" in str(p): continue
        if "CUDA" in this_f and ".cu" in str(p): pass
        elif not _reference(str(p), this_f): continue 
        out.append(str(p))
    return list(set(out))
 


PACKAGES_CXX = {
    "PyC.Transform.Floats" : "src/Transform/Shared/Floats.cxx", 
    "PyC.Transform.Tensors" : "src/Transform/Shared/Tensors.cxx",  
    "PyC.Operators.Tensors" : "src/Operators/Shared/Tensors.cxx",  
    "PyC.Physics.Tensors.Cartesian" : "src/Physics/Shared/CartesianTensors.cxx", 
    "PyC.Physics.Tensors.Polar" : "src/Physics/Shared/PolarTensors.cxx", 
    "PyC.NuSol.Tensors" : "src/NuRecon/Shared/Tensor.cxx", 
}

PACKAGES_CUDA = {
    "PyC.Transform.CUDA" : "src/Transform/Shared/CUDA.cxx", 
    "PyC.Operators.CUDA" : "src/Operators/Shared/CUDA.cxx", 
    "PyC.Physics.CUDA.Cartesian" : "src/Physics/Shared/CartesianCUDA.cxx",   
    "PyC.Physics.CUDA.Polar" : "src/Physics/Shared/PolarCUDA.cxx",   
    "PyC.NuSol.CUDA" : "src/NuRecon/Shared/CUDA.cxx"
}

PACKAGES = {}
PACKAGES |= PACKAGES_CXX 
PACKAGES |= PACKAGES_CUDA if cuda else {}
DEPENDS = {}
REFDEPENDS = {}
INST_ = []
INST_H = {}

for pkg in PACKAGES:
    deps = _recursive(PACKAGES[pkg])
    HEADER = [k for k in deps if _check(k, ".h")]
    CXX = [k for k in deps if _check(k, ".cxx")]
    CU = [k for k in deps if _check(k, ".cu") and cuda]
   
    dic = {"name" : pkg, "sources": CXX + CU, "extra_compile_args" : ["-std=c++14"]} 
    INST_.append(CppExtension(**dic) if len(CU) == 0 else CUDAExtension(**dic))
    INST_H[pkg] = HEADER

cmd = { 
        "ext_modules"  : INST_, 
        "package_data" : INST_H, 
        "cmdclass"     : {"build_ext" : BuildExtension}, 
} 
setup(**cmd)
