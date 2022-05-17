from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(name = "PyTorchCustom", 
        ext_modules = [
            CppExtension("LorentzVector", ["LorentzVector.cpp"])
        ],
        cmdclass = {"build_ext" : BuildExtension}
    )
