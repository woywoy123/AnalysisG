from selection import NeutrinoReconstruction
from AnalysisG import Analysis
from AnalysisG.Events import Event
import os

smpl = os.environ["Samples"]

Ana = Analysis()
Ana.OutputDirectory = "./Reconstruction"
Ana.InputSample("bsm1000", smpl+"/ttZ-1000/")
Ana.AddSelection("neutrino", NeutrinoReconstruction)
Ana.EventStop = 1000
Ana.Event = Event
Ana.Threads = 1
Ana.EventCache = True
Ana.Launch


