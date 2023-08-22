from setuptools import setup, Extension
from Cython.Build import cythonize


modules = [
    {
        "name" : "AnalysisG._cmodules.code",
        "sources" : [
            "src/cmodules/cyincludes/cycode.pyx",
            "src/cmodules/code/code.cxx",
            "src/cmodules/tools/tools.cxx",
        ],
        "include_dirs": [
            "src/cmodules/code/",
            "src/cmodules/tools/",
        ],
        "language" : "c++"
    },
    {
        "name" : "AnalysisG._cmodules.EventTemplate",
        "sources" : [
            "src/cmodules/cyincludes/cyevent.pyx",
            "src/cmodules/event/event.cxx",
            "src/cmodules/tools/tools.cxx",
        ],
        "include_dirs": [
            "src/cmodules/event/",
            "src/cmodules/tools/",
        ],
        "language" : "c++"
    },
    {
        "name" : "AnalysisG._cmodules.ParticleTemplate",
        "sources" : [
            "src/cmodules/cyincludes/cyparticle.pyx",
            "src/cmodules/particle/particle.cxx",
            "src/cmodules/tools/tools.cxx",
        ],
        "include_dirs": [
            "src/cmodules/particle/",
            "src/cmodules/tools/",
        ],
        "language" : "c++"
    },
    {
        "name" : "AnalysisG._cmodules.MetaData",
        "sources" : [
            "src/cmodules/cyincludes/cymetadata.pyx",
            "src/cmodules/metadata/metadata.cxx",
            "src/cmodules/tools/tools.cxx",
        ],
        "include_dirs": [
            "src/cmodules/metadata/",
            "src/cmodules/tools/",
        ],
        "language" : "c++"
    },
    {
        "name" : "AnalysisG._cmodules.SampleTracer",
        "sources" : [
            "src/cmodules/cyincludes/cysampletracer.pyx",

            "src/cmodules/sampletracer/sampletracer.cxx",
            "src/cmodules/sampletracer/root.cxx",
            "src/cmodules/metadata/metadata.cxx",
            "src/cmodules/event/event.cxx",
            "src/cmodules/code/code.cxx",
            "src/cmodules/tools/tools.cxx",
        ],
        "include_dirs": [
            "src/cmodules/code/",
            "src/cmodules/event/",
            "src/cmodules/metadata/",
            "src/cmodules/sampletracer/",
            "src/cmodules/tools/",
        ],
        "language" : "c++"
    },


]

for i in range(len(modules)):
    modules[i] = Extension(**modules[i])

setup(ext_modules = cythonize(modules, nthreads = 12))



