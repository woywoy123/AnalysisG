try: from AnalysisG._cmodules.EventTemplate import EventTemplate
except ModuleNotFoundError: print("ERROR: Missing EventTemplate Compiler...")

try: from AnalysisG._cmodules.ParticleTemplate import ParticleTemplate
except ModuleNotFoundError: print("ERROR: Missing ParticleTemplate Compiler...")
