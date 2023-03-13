from setuptools import setup, Extension
import Cython.Build

def pre_install_torch():
        import pip
        pip.main(["install", "torch==1.13.1"])

def post_install_extensions():
        import torch
        from torch.utils.cpp_extension import BuildExtension, CppExtension
        version = ["Floats", "Tensors", "CUDA"]
        Base = {
                        "Transform" : [ "ToCartesian", "ToPolar"],  
                        "Operators" : [ "", "" ], 
                        "Physics"   : [ "FromCartesian", "FromPolar", "" ], 
                        "NuSol"     : [ "NuSol"], 
                }






pre_install_torch()

setup(
        ext_modules = [
                Extension(
                        "AnalysisTopGNN.Vectors", 
                        sources = ["src/Vectors/Lorentz.pyx"]
                ),
        ],
        cmdclass = {
                "build_ext" : Cython.Build.build_ext, 
        },
)
