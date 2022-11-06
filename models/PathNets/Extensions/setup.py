from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension#, CUDAExtension

setup(name = "PathNetOptimizer", 
    ext_modules = [
        CppExtension("PathNetOptimizer", ["Combinatorial.cxx"]),
        #CUDAExtension("PathNetOptimizerCUDA", ["CombinatorialCUDA.cxx", "Combinatorial.cu"])
    ],
    cmdclass = {"build_ext" : BuildExtension}
)


