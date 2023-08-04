from selection import NeutrinoReconstruction
from AnalysisG import Analysis
from AnalysisG.Events import Event
import os

smpl = os.environ["Samples"]

Ana = Analysis()
Ana.OutputDirectory = "./Reconstruction"
Ana.SampleInput("bsm1000", smpl+"/ttZ-1000/")
Ana.EventStop = 1000
Ana.Event = Event
Ana.EventCache = True
Ana.Launch


