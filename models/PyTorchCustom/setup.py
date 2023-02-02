from setuptools import setup 
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
#import os 

Vector_H = "src/Vectors/Headers/"
Vector_C = "src/Vectors/CXX/"
Vector_S = "src/Vectors/Shared/"
Vector_Cu = "src/Vectors/CUDA/"

#os.environ["CC"] = "gcc-11"
#os.environ["CXX"] = "gcc-11"

setup(
        name = "PyTCustom-HEPP", 
        version = "1.0", 
        package_data = {
            "PyC.Vectors.Floats" : [
                Vector_H + "ToCartesianFloats.h", 
                Vector_H + "ToPolarFloats.h"
            ], 
            "PyC.Vectors.Tensors" : [
                Vector_H + "ToCartesianTensors.h", 
                Vector_H + "ToPolarTensors.h", 
            ],
            "PyC.Vectors.CUDA" : [
                Vector_H + "ToCartesianCUDA.h", 
                Vector_H + "ToPolarCUDA.h"
            ]
        }, 

        ext_modules = [
            CppExtension("PyC.Vectors.Floats", [
                Vector_C + "ToCartesianFloats.cxx", 
                Vector_C + "ToPolarFloats.cxx", 
                Vector_S + "Floats.cxx"
            ]),
            CppExtension("PyC.Vectors.Tensors", [
                Vector_C + "ToCartesianTensors.cxx", 
                Vector_C + "ToPolarTensors.cxx", 
                Vector_S + "Tensors.cxx"
            ]), 
            CUDAExtension("PyC.Vectors.CUDA", [
                                            Vector_Cu + "Cartesian.cu",
                                            Vector_Cu + "CartesianKernel.cu", 
                                            Vector_Cu + "CartesianTorch.cu", 
                                            Vector_Cu + "Polar.cu",
                                            Vector_Cu + "PolarKernel.cu", 
                                            Vector_Cu + "PolarTorch.cu", 
                                            Vector_S + "CUDA.cxx", 
            ]), 
        ], 
        cmdclass = {"build_ext" : BuildExtension}
)
