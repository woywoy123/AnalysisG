from setuptools import setup, Extension
from Cython.Build import cythonize


ext_mod = [
    Extension(
        name = "AnalysisG.Templates.EventTemplate",
        sources = [
            "src/Templates/event/cyevent.pyx",
            "src/Templates/event/event.cxx",
            "src/Templates/tools/tools.cxx"
        ],
        include_dirs = ["src/Templates/tools/"],
        language = "c++"
    ),

    Extension(
        name = "AnalysisG.Templates.ParticleTemplate",
        sources = [
            "src/Templates/particle/cyparticle.pyx",
            "src/Templates/particle/particle.cxx",
            "src/Templates/tools/tools.cxx"
        ],
        include_dirs = ["src/Templates/tools/"],
        language = "c++"
    ),

    Extension(
        name = "AnalysisG.SampleTracer.MetaData",
        sources = [
            "src/SampleTracer/metadata/cymetadata.pyx",
            "src/SampleTracer/metadata/metadata.cxx",
            "src/Templates/tools/tools.cxx",
        ],
        inlude_dirs = ["src/Templates/tools/"],
        language = "c++"
    ),
]

setup(ext_modules = cythonize(ext_mod, nthreads = 12))



