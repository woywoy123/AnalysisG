from double_leptonic import DiLeptonic
from AnalysisG.Events import Event
from AnalysisG import Analysis
import os
smpls = os.environ["Samples"]

which_analysis = "double-leptonic:DiLeptonic"

_ana = which_analysis.split(":")
if _ana[0] == "double-leptonic": ana = DiLeptonic()

Ana = Analysis()
Ana.ProjectName = _ana[1]
#Ana.InputSample("ttZ", smpls + "ttZ-1000")
#Ana.InputSample("ttbar", smpls + "ttbar")
Ana.Event = Event
Ana.EventStop = 1000
Ana.EventCache = True
Ana.Threads = 1
Ana.AddSelection(ana.__class__.__name__, ana)
Ana.Launch

