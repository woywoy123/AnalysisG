from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import os

# This makes sure the CUDA compiler is consistent with the GCC of the system 
os.environ["CC"] = "gcc-11"

setup(name = "PathNetOptimizer", 
    ext_modules = [
        CppExtension("PathNetOptimizerCPU", ["Combinatorial.cxx"]),
        CUDAExtension("PathNetOptimizerCUDA", ["CombinatorialCUDA.cxx", "Combinatorial.cu"])
    ],
    cmdclass = {"build_ext" : BuildExtension}
)


