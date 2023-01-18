from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
        package_data = {
            "Tensors" : [
                "Headers/PhysicsTensors.h"
            ], 
        }, 
        ext_modules = [
            CppExtension("Tensors", 
                            [
                                "CXX/PhysicsTensors.cxx", 
                                "Shared/PhysicsTensors.cxx"
                            ]
            ), 
        ], 
        cmdclass = {"build_ext" : BuildExtension}
)
