from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import os

# Make sure the CUDA compilers are consistent with the GCC of the system 
os.environ["CC"] = "gcc-11"
os.environ["CXX"] = "gcc-11"

setup(
        name = "PhysicsCPU", 
        package_data = {
            "PhysicsCPU" : ["BaseFunctions/Headers/Physics.h"]
            }, 
        ext_modules = [
            CppExtension("PhysicsCPU", ["BaseFunctions/CXX/Physics.cxx"]), 

        ],
        cmdclass = {"build_ext" : BuildExtension},
)
