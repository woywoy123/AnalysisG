from setuptools import setup, Extension
from Cython.Build import cythonize

ext_mod = [
    Extension(
                name = "AnalysisG.Templates.ParticleTemplates", 
                sources = [
                    "src/Templates/Cython/Particles.pyx", 
                    "src/Templates/CXX/Particles.cxx",

                    "src/Templates/CXX/Tools.cxx", 
                ], 
    ), 
    Extension(
                name = "AnalysisG.Templates.EventTemplates", 
                sources = [
                    "src/Templates/Cython/Event.pyx", 
                    "src/Templates/CXX/Event.cxx",

                    "src/Templates/CXX/Tools.cxx", 
                ], 
    ), 

    Extension(
                name = "AnalysisG.Tracer", 
                sources = [
                    "src/Templates/Cython/ROOT.pyx", 
                    "src/Templates/CXX/ROOT.cxx",

                    "src/Templates/CXX/Tools.cxx", 
                ], 
    ),
]

setup(
        ext_modules = cythonize(ext_mod),
)
