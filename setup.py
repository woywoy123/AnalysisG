from setuptools import setup, Extension
from Cython.Build import cythonize

src = "src/cmodules/"
modules = [
    #{
    #    "name" : "AnalysisG._cmodules.code",
    #    "sources" : [
    #        src + "cyincludes/cycode.pyx",
    #        src + "code/code.cxx",

    #        src + "abstractions/abstractions.cxx",
    #    ],

    #    "include_dirs": [
    #        src + "code/",
    #        src + "abstractions/",
    #    ],
    #    "language" : "c++"
    #},
    {
        "name" : "AnalysisG._cmodules.EventTemplate",
        "sources" : [
            src + "cyincludes/cyevent.pyx",
            src + "event/event.cxx",

            src + "abstractions/abstractions.cxx",
            src + "metadata/metadata.cxx",

        ],
        "include_dirs": [
            src + "event/",
            src + "metadata/",
            src + "abstractions/",
        ],
        "language" : "c++"
    },
    {
        "name" : "AnalysisG._cmodules.ParticleTemplate",
        "sources" : [
            src + "cyincludes/cyparticle.pyx",
            src + "particle/particle.cxx",

            src + "abstractions/abstractions.cxx",
        ],
        "include_dirs": [
            src + "particle/",
            src + "abstractions/"
        ],
        "language" : "c++"
    },
    #{
    #    "name" : "AnalysisG._cmodules.MetaData",
    #    "sources" : [
    #        src + "cyincludes/cymetadata.pyx",
    #        src + "metadata/metadata.cxx",

    #        src + "abstractions/abstractions.cxx"
    #    ],
    #    "include_dirs": [
    #        src + "metadata/",
    #        src + "abstractions/"
    #    ],
    #    "language" : "c++"
    #},
    #{
    #    "name" : "AnalysisG._cmodules.SampleTracer",
    #    "sources" : [
    #        "src/cmodules/cyincludes/cysampletracer.pyx",

    #        "src/cmodules/sampletracer/sampletracer.cxx",
    #        "src/cmodules/sampletracer/root.cxx",
    #        "src/cmodules/metadata/metadata.cxx",
    #        "src/cmodules/event/event.cxx",
    #        "src/cmodules/code/code.cxx",
    #        "src/cmodules/tools/tools.cxx",
    #    ],
    #    "include_dirs": [
    #        "src/cmodules/code/",
    #        "src/cmodules/event/",
    #        "src/cmodules/metadata/",
    #        "src/cmodules/sampletracer/",
    #        "src/cmodules/tools/",
    #    ],
    #    "language" : "c++"
    #},


]

for i in range(len(modules)):
    modules[i] = Extension(**modules[i])

setup(ext_modules = cythonize(modules, nthreads = 12))



