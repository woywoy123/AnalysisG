from distutils.core import setup
from setuptools import Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
        name = "AnalysisTopGNN", 
        version = "1.0", 

        packages = [
            "AnalysisTopGNN", 
            "AnalysisTopGNN.Events", 
            "AnalysisTopGNN.Particles", 
            "AnalysisTopGNN.Templates", 
            "AnalysisTopGNN.Reconstruction",
            "AnalysisTopGNN.Tools", 
            "AnalysisTopGNN.IO",
            "AnalysisTopGNN.Generators",
            "AnalysisTopGNN.Plotting", 
            "AnalysisTopGNN.Plotting.Legacy", 
            "AnalysisTopGNN.Submission", 
        ],
        package_dir = {
            "AnalysisTopGNN": "src",
            "AnalysisTopGNN.Events" : "src/EventTemplates/Events", 
            "AnalysisTopGNN.Particles" : "src/EventTemplates/Particles",
            "AnalysisTopGNN.Templates" : "src/EventTemplates/Templates", 
            "AnalysisTopGNN.Reconstruction" : "src/EventTemplates/Reconstruction",
            "AnalysisTopGNN.Tools" : "src/Tools",
            "AnalysisTopGNN.IO" : "src/IO",
            "AnalysisTopGNN.Generators" : "src/Generators", 
            "AnalysisTopGNN.Plotting" : "src/Plotting",
            "AnalysisTopGNN.Plotting.Legacy" : "src/Plotting/Legacy", 
            "AnalysisTopGNN.Submission" : "src/Submission", 
        },

        long_description = open("README.md").read(), 
    )

#setup(
#        name = "PyTorchCustom", 
#        ext_modules = [
#            CppExtension("LorentzVector", ["src/PyTorchCustom/Source/LorentzVector.cpp"])
#        ],
#        cmdclass = {"build_ext" : BuildExtension}
#    )
