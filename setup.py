from distutils.core import setup

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
            "AnalysisTopGNN.Models"
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
            "AnalysisTopGNN.Models" : "src/Models"
        },

        long_description = open("README.md").read(), 
    )

