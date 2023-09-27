#from plotting.double_leptonic.truthchildren import doubleleptonic_Plotting
from double_leptonic import DiLeptonic
from AnalysisG.Events import Event
from AnalysisG.IO import nTupler
from AnalysisG import Analysis
import os


smpls = os.environ["Samples"]
which_analysis = "double-leptonic:DiLeptonic"
_ana = which_analysis.split(":")
if _ana[0] == "double-leptonic": ana = DiLeptonic()

ana.__params__ = { "btagger" : "is_b" , "truth" : "children"}
samples = {
        "ttZ"   : smpls + "ttZ-1000",
        #"other" : smpls + "other",
        #"t"     : smpls + "t"    ,
        #"tt"    : smpls + "tt"   ,
        #"ttbar" : smpls + "ttbar",
        #"ttH"   : smpls + "ttH"  ,
        #"tttt"  : smpls + "tttt" ,
        #"ttX"   : smpls + "ttX"  ,
        #"ttXll" : smpls + "ttXll",
        #"ttXqq" : smpls + "ttXqq",
        #"V"     : smpls + "V"    ,
        #"Vll"   : smpls + "Vll"  ,
        #"Vqq"   : smpls + "Vqq"
}

run = True
if run:
    Ana = Analysis()
    Ana.ProjectName = _ana[1]
    for key in samples: Ana.InputSample(key, samples[key])
    Ana.Event = Event
    Ana.EventStop = 1000
    Ana.EventCache = True
    Ana.Threads = 1
    Ana.chnk = 1000
    Ana.AddSelection(ana)
    Ana.Launch()

tupler = nTupler()
tupler.This("DiLeptonic", "nominal")
tupler.Threads = 1
tupler.ProjectName = _ana[1]
x = tupler.merged()
print(x["nominal.DiLeptonic"].Luminosity)


#doubleleptonic_Plotting(x[_ana[1]], "truthchildren")

