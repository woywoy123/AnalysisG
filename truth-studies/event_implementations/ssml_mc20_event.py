from AnalysisG.Events import SSML_MC20
from AnalysisG import Analysis
import os

smpls = os.environ["Samples"] + "mc20_13_tttt_m1250"
ana = Analysis()
ana.InputSample(None, smpls)
ana.Threads = 1
ana.Verbose = 3
ana.DebugMode = True
ana.EventCache = False
ana.Event = SSML_MC20
ana.Launch()
