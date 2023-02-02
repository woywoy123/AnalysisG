from setuptools import setup 
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
#import os 

Transform_H  = "src/Transform/Headers/"
Transform_C  = "src/Transform/CXX/"
Transform_S  = "src/Transform/Shared/"
Transform_Cu = "src/Transform/CUDA/"

Operators_H  = "src/Operators/Headers/"
Operators_C  = "src/Operators/CXX/"
Operators_S  = "src/Operators/Shared/"
Operators_Cu = "src/Operators/CUDA/"





#os.environ["CC"] = "gcc-11"
#os.environ["CXX"] = "gcc-11"

setup(
        name = "PyTCustom-HEPP", 
        version = "1.0", 
        package_data = {
            "PyC.Transform.Floats" : [
                Transform_H + "ToCartesianFloats.h", 
                Transform_H + "ToPolarFloats.h"
            ], 
            "PyC.Transform.Tensors" : [
                Transform_H + "ToCartesianTensors.h", 
                Transform_H + "ToPolarTensors.h", 
            ],
            "PyC.Transform.CUDA" : [
                Transform_H + "ToCartesianCUDA.h", 
                Transform_H + "ToPolarCUDA.h"
            ], 
            "PyC.Operators.Tensors" : [
                Operators_H + "Tensors.h" 
            ],
            "PyC.Operators.CUDA" : [
                Operators_H + "CUDA.h"
            ], 
        }, 

        ext_modules = [
            CppExtension("PyC.Transform.Floats", [
                Transform_C + "ToCartesianFloats.cxx", 
                Transform_C + "ToPolarFloats.cxx", 
                Transform_S + "Floats.cxx"
            ]),
            CppExtension("PyC.Transform.Tensors", [
                Transform_C + "ToCartesianTensors.cxx", 
                Transform_C + "ToPolarTensors.cxx", 
                Transform_S + "Tensors.cxx"
            ]),
            CUDAExtension("PyC.Transform.CUDA", [
                Transform_Cu + "Cartesian.cu",
                Transform_Cu + "CartesianKernel.cu", 
                Transform_Cu + "CartesianTorch.cu", 
                Transform_Cu + "Polar.cu",
                Transform_Cu + "PolarKernel.cu", 
                Transform_Cu + "PolarTorch.cu", 
                Transform_S  + "CUDA.cxx", 
            ]),
            CppExtension("PyC.Operators.Tensors", [
                Operators_C + "Tensors.cxx", 
                Operators_S + "Tensors.cxx", 
            ]), 
            CUDAExtension("PyC.Operators.CUDA", [
                Operators_Cu + "Operators.cu", 
                Operators_Cu + "OperatorsKernel.cu", 
                Operators_Cu + "OperatorsTorch.cu", 
                Operators_S  + "CUDA.cxx"
            ]),

        ], 
        cmdclass = {"build_ext" : BuildExtension}
)
