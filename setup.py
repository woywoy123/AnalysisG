from distutils.core import setup

setup(
        name = "AnalysisTopGNN", 
        version = "1.0", 




        packages = [
            "AnalysisTopGNN", 
            "AnalysisTopGNN.Event", 
            "AnalysisTopGNN.Event.Implementations",
            "AnalysisTopGNN.Tools", 
            "AnalysisTopGNN.IO",
            "AnalysisTopGNN.Particles"
        ],
        package_dir = {
            "AnalysisTopGNN": "src", 
            "AnalysisTopGNN.Event" : "src/Event",
            "AnalysisTopGNN.Event.Implementations" : "src/Event/Implementations",
            "AnalysisTopGNN.Tools" : "src/Tools", 
            "AnalysisTopGNN.IO" : "src/IO",
            "AnalysisTopGNN.Particles" : "src/Particles"
        },

        long_description = open("README.md").read(), 
        #package_dir = {pkg : f"src/{pkg.replace('.', '/')}" for pkg in packages}, 
    )

