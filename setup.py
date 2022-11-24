from distutils.core import setup
from setuptools import Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
        name = "PyTorchCustom", 
        ext_modules = [
            CppExtension("LorentzVector", ["src/PyTorchCustom/Source/LorentzVector.cpp"])
        ],
        cmdclass = {"build_ext" : BuildExtension}
    )

setup(
        name = "AnalysisTopGNN", 
        version = "2.0", 

        packages = [
            "AnalysisTopGNN", 
            "AnalysisTopGNN.Notification", 
            "AnalysisTopGNN.Samples", 
            "AnalysisTopGNN.Tools", 
            
            "AnalysisTopGNN.Model", 
            "AnalysisTopGNN.Templates", 
            "AnalysisTopGNN.Particles", 
            "AnalysisTopGNN.Events",
            "AnalysisTopGNN.Deprecated",
            
            "AnalysisTopGNN.Features",  
            "AnalysisTopGNN.Features.ParticleGeneric",
            "AnalysisTopGNN.Features.TruthJet",
            "AnalysisTopGNN.Features.TruthTop",
            "AnalysisTopGNN.Features.TruthTopChildren",

            "AnalysisTopGNN.IO",
            "AnalysisTopGNN.Generators",
            "AnalysisTopGNN.Plotting", 
            "AnalysisTopGNN.Submission", 
            "AnalysisTopGNN.Statistics", 
        ],
        package_dir = {
            "AnalysisTopGNN": "src",
            "AnalysisTopGNN.Notification" : "src/Notification",
            "AnalysisTopGNN.Samples" : "src/Samples", 
            "AnalysisTopGNN.Tools" : "src/Tools",
            
            "AnalysisTopGNN.Model" : "src/Model", 
            "AnalysisTopGNN.Templates" : "src/EventTemplates/Templates", 
            "AnalysisTopGNN.Particles" : "src/EventTemplates/Particles",
            "AnalysisTopGNN.Events" : "src/EventTemplates/Events",
            "AnalysisTopGNN.Deprecated" : "src/EventTemplates/Deprecated", 
            
            "AnalysisTopGNN.Features" : "src/Features", 
            "AnalysisTopGNN.Features.ParticleGeneric" : "src/Features/ParticleGeneric",
            "AnalysisTopGNN.Features.TruthJet" : "src/Features/TruthJet",
            "AnalysisTopGNN.Features.TruthTop" : "src/Features/TruthTop",
            "AnalysisTopGNN.Features.TruthTopChildren" : "src/Features/TruthTopChildren",

            "AnalysisTopGNN.IO" : "src/IO",
            "AnalysisTopGNN.Plotting" : "src/Plotting",
            "AnalysisTopGNN.Statistics" : "src/Statistics", 
            "AnalysisTopGNN.Generators" : "src/Generators", 
            "AnalysisTopGNN.Submission" : "src/Submission", 
        },

        long_description = open("README.md").read(), 
    )


