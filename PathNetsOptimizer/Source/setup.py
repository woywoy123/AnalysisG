from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(name = "PathNetOptimizer_cpp", 
    ext_modules = [
        CppExtension("PathNetOptimizer_cpp", ["PathNetOptimizer.cpp"]),
        CUDAExtension("PathNetOptimizerCUDA_cpp", ["PathNetOptimizerCUDA.cpp", "PathNetOptimizer.cu", ])
    ],
    cmdclass = {"build_ext" : BuildExtension}
)


