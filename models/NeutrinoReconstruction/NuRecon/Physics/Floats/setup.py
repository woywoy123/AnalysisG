from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
        package_data = {
            "Floats" : [
                "../Tensors/Headers/PhysicsTensors.h", 
                "Headers/PhysicsFloats.h"
            ],
        }, 
        ext_modules = [
            CppExtension("Floats", 
                            [
                                "CXX/PhysicsFloats.cxx", 
                                "Shared/PhysicsFloats.cxx", 
                                "../Tensors/CXX/PhysicsTensors.cxx", 
                            ]
            ), 
        ], 
        cmdclass = {"build_ext" : BuildExtension}
)
