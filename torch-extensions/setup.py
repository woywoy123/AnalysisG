import torch
from torch.utils.cpp_extension import BuildExtension
from setuptools import setup
import os

_dir = "src/"
TrH, TrC, TrS, TrCu  = _dir + "Transform/Headers/", _dir + "Transform/CXX/", _dir + "Transform/Shared/", _dir + "Transform/CUDA/"
OpH, OpC, OpS, OpCu  = _dir + "Operators/Headers/", _dir + "Operators/CXX/", _dir + "Operators/Shared/", _dir + "Operators/CUDA/"
PhH, PhC, PhS, PhCu  = _dir + "Physics/Headers/",   _dir + "Physics/CXX/",   _dir + "Physics/Shared/",   _dir + "Physics/CUDA/"
NuH, NuC, NuS, NuCu  = _dir + "NuRecon/Headers/",   _dir + "NuRecon/CXX/",   _dir + "NuRecon/Shared/",   _dir + "NuRecon/CUDA/"      

PkgL = [
                "PyC.Transform.Floats", "PyC.Transform.Tensors", "PyC.Transform.CUDA", 
                "PyC.Operators.Tensors", "PyC.Operators.CUDA", 
                "PyC.Physics.Tensors.Cartesian", "PyC.Physics.CUDA.Cartesian", 
                "PyC.Physics.Tensors.Polar", "PyC.Physics.CUDA.Polar", 
                "PyC.NuSol.Tensors", "PyC.NuSol.CUDA" 
]

PkgH = [
        [TrH + "ToCartesianFloats.h", TrH + "ToPolarFloats.h"], 
        [TrH + "ToCartesianTensors.h", TrH  + "ToPolarTensors.h"],
        [TrH + "ToCartesianCUDA.h", TrH + "ToPolarCUDA.h"], 
        [OpH + "Tensors.h"], 
        [OpH + "CUDA.h"], 
        [TrH + "ToPolarTensors.h", PhH + "FromCartesianTensors.h", PhH + "Tensors.h"], 
        [TrH + "ToPolarCUDA.h", PhH + "CUDA.h", PhH + "FromCartesianCUDA.h"], 
        [TrH + "ToCartesianTensors.h", PhH + "FromPolarTensors.h", PhH + "Tensors.h"], 
        [TrH + "ToCartesianCUDA.h", PhH + "CUDA.h", PhH + "FromPolarCUDA.h"], 
        [TrH + "ToCartesianTensors.h", TrH + "ToPolarTensors.h", PhH + "Tensors.h", OpH + "Tensors.h", NuH + "NuSolTensor.h"], 
        [TrH + "ToCartesianCUDA.h", TrH + "ToPolarCUDA.h", PhH+ "CUDA.h", OpH + "CUDA.h", NuH + "NuSolCUDA.h"] 
]

PkgC = [
        [TrC + "ToCartesianFloats.cxx", TrC + "ToPolarFloats.cxx", TrS + "Floats.cxx"], 
        [TrC + "ToCartesianTensors.cxx", TrC + "ToPolarTensors.cxx", TrS + "Tensors.cxx"],
        [
                TrCu + "Cartesian.cu", TrCu + "CartesianKernel.cu", 
                TrCu + "CartesianTorch.cu", TrCu + "Polar.cu", 
                TrCu + "PolarKernel.cu", TrCu + "PolarTorch.cu", TrS  + "CUDA.cxx"
        ],
        [OpC + "Tensors.cxx", OpS + "Tensors.cxx"], 
        [
                OpCu + "Operators.cu", OpCu + "OperatorsKernel.cu", 
                OpCu + "OperatorsTorch.cu", OpS  + "CUDA.cxx"
        ],
        [TrC + "ToPolarTensors.cxx", PhC + "Tensors.cxx", PhS + "CartesianTensors.cxx"], 
        [
                TrCu + "Polar.cu", TrCu + "PolarKernel.cu", 
                TrCu + "PolarTorch.cu", PhCu + "Physics.cu", 
                PhCu + "PhysicsKernel.cu", PhCu + "PhysicsTorch.cu", 
                PhS + "CartesianCUDA.cxx"
        ], 
        [TrC + "ToCartesianTensors.cxx", PhC + "Tensors.cxx", PhS + "PolarTensors.cxx"], 
        [
                TrCu + "Cartesian.cu", TrCu + "CartesianKernel.cu", 
                TrCu + "CartesianTorch.cu", PhCu + "Physics.cu", 
                PhCu + "PhysicsKernel.cu", PhCu + "PhysicsTorch.cu", PhS + "PolarCUDA.cxx"
        ], 
        [
                TrC + "ToCartesianTensors.cxx", TrC + "ToPolarTensors.cxx",  
                PhC + "Tensors.cxx", OpC + "Tensors.cxx", 
                NuC + "NuSolTensor.cxx", NuC + "SingleNuTensor.cxx", 
                NuC + "DoubleNuTensor.cxx", NuS + "Tensor.cxx"
        ], 
        [
                TrCu + "Cartesian.cu", TrCu + "CartesianKernel.cu", TrCu + "CartesianTorch.cu", 
                TrCu + "Polar.cu", TrCu + "PolarKernel.cu", TrCu + "PolarTorch.cu", 
                PhCu + "Physics.cu", PhCu + "PhysicsKernel.cu", PhCu + "PhysicsTorch.cu", 
                OpCu + "Operators.cu", OpCu + "OperatorsKernel.cu", OpCu + "OperatorsTorch.cu", 
                NuCu + "NuSol.cu", NuCu + "NuSolKernel.cu", NuCu + "NuSolTorch.cu", NuS + "CUDA.cxx"
        ]
]

_cmd = {"package_data" : {}, "ext_modules" : [], "cmdclass" : {"build_ext" : BuildExtension}}
for i in range(len(PkgL)):
        _cu = os.environ.get("CUDA_PATH")
        if (_cu == None or _cu == "") and "CUDA" in PkgL[i]:
                continue
        if "CUDA" in PkgL[i]:
                from torch.utils.cpp_extension import CUDAExtension
                _cmd["ext_modules"].append(CUDAExtension( PkgL[i], PkgC[i] ))
                _cmd["package_data"][ PkgL[i] ] = PkgH[i]
                continue
        from torch.utils.cpp_extension import CppExtension
        _cmd["package_data"][ PkgL[i] ] = PkgH[i]
        _cmd["ext_modules"].append(CppExtension( PkgL[i], PkgC[i] ))
setup(**_cmd)
