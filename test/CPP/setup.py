from setuptools import setup, Extension
from Cython.Build import cythonize

ext_mod = [
    Extension(
                name = "AnalysisG.Templates", 
                sources = [
                    "src/Templates/Cython/Particles.pyx", 
                    "src/Templates/CXX/Particles.cxx",
                    "src/Templates/CXX/Tools.cxx", 
                ], 
    ), 

    ]
setup(
        ext_modules = cythonize(ext_mod),
)
