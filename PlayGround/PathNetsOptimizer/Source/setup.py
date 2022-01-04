from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name = "PathNetOptimizer_cpp", 
      ext_modules = [cpp_extension.CppExtension("PathNetOptimizer_cpp", ["PathNetOptimizer.cpp"])], 
      cmdclass = {"build_ext" : cpp_extension.BuildExtension})


