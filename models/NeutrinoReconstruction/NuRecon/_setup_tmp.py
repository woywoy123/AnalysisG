from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import os

# Make sure the CUDA compilers are consistent with the GCC of the system 
#os.environ["CC"] = "gcc-11"
#os.environ["CXX"] = "gcc-11"

setup(
        name = "BaseFunctions", 
        package_data = {
                "Floats" : [
                    "BaseFunctions/Headers/PhysicsFloats.h", 
                    "BaseFunctions/Headers/PhysicsTensors.h"
                    ], 
                "Tensors" : [
                    "BaseFunctions/Headers/PhysicsTensors.h"
                    ], 
        }, 

        ext_modules = [
                CppExtension("Floats", 
                         [
                             "BaseFunctions/CXX/PhysicsTensors.cxx", 
                             "BaseFunctions/CXX/PhysicsFloats.cxx", 
                             "BaseFunctions/Shared/PhysicsFloats.cxx"
                        ]
                ), 
                CppExtension("Tensors", 
                        [
                            "BaseFunctions/CXX/PhysicsTensors.cxx", 
                            "BaseFunctions/Shared/PhysicsTensors.cxx"
                        ]
                ), 
        ],
        cmdclass = {"build_ext" : BuildExtension}
)

#setup(
#        name = "NuSol", 
#        package_data = {
#                "Floats" : [
#                        "BaseFunctions/Headers/PhysicsFloats.h", 
#                        "BaseFunctions/Headers/PhysicsTensors.h",
#                        "NuSolutions/Headers/NuSolFloat.h"
#                    ], 
#                "Tensors" : [
#                        "BaseFunctions/Headers/PhysicsTensors.h", 
#                        "NuSolutions/Headers/NuSolTensor.h"
#                    ], 
#        }, 
#
#        ext_modules = [
#                CppExtension("Floats", 
#                    [
#                        "BaseFunctions/CXX/PhysicsFloats.cxx",
#                        "BaseFunctions/CXX/PhysicsTensors.cxx", 
#                        "NuSolutions/CXX/NuSolFloat.cxx", 
#                        "NuSolutions/Shared/NuSolFloat.cxx"
#                    ]
#                ), 
#
#                CppExtension("Tensors", 
#                    [
#                        "BaseFunctions/CXX/PhysicsTensors.cxx", 
#                        "NuSolutions/CXX/NuSolTensor.cxx", 
#                        "NuSolutions/Shared/NuSolTensor.cxx"
#                    ]
#                ), 
#        ],
#
#        cmdclass = {"build_ext" : BuildExtension}
#)
