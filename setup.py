from distutils.core import setup
from setuptools import Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
        name = "AnalysisTopGNN", 
        version = "2.0", 

        packages = [
            "AnalysisTopGNN", 
            "AnalysisTopGNN.Notification", 
            "AnalysisTopGNN.Samples", 
            
            "AnalysisTopGNN.Events", 
            "AnalysisTopGNN.Particles", 
            "AnalysisTopGNN.Templates", 
            "AnalysisTopGNN.Reconstruction",
            "AnalysisTopGNN.Tools", 
            "AnalysisTopGNN.IO",
            "AnalysisTopGNN.Generators",
            "AnalysisTopGNN.Plotting", 
            "AnalysisTopGNN.Submission", 
        ],
        package_dir = {
            "AnalysisTopGNN": "src",
            "AnalysisTopGNN.Notification" : "src/Notification",
            "AnalysisTopGNN.Samples" : "src/Samples", 


            "AnalysisTopGNN.Events" : "src/EventTemplates/Events", 
            "AnalysisTopGNN.Particles" : "src/EventTemplates/Particles",
            "AnalysisTopGNN.Templates" : "src/EventTemplates/Templates", 
            "AnalysisTopGNN.Reconstruction" : "src/Reconstruction",

            "AnalysisTopGNN.Tools" : "src/Tools",
            
            
            "AnalysisTopGNN.IO" : "src/IO",
            "AnalysisTopGNN.Generators" : "src/Generators", 
            "AnalysisTopGNN.Plotting" : "src/Plotting",
            "AnalysisTopGNN.Submission" : "src/Submission", 
        },

        long_description = open("README.md").read(), 
    )

setup(
        name = "PyTorchCustom", 
        ext_modules = [
            CppExtension("LorentzVector", ["src/PyTorchCustom/Source/LorentzVector.cpp"])
        ],
        cmdclass = {"build_ext" : BuildExtension}
    )
