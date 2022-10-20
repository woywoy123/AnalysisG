from AnalysisTopGNN.Generators import Analysis
from AnalysisTopGNN.Events import Event, EventGraphTruthTopChildren
#from .Optimization import Optimizer
from Templates.EventFeatureTemplate import ApplyFeatures










def TestOptimizer(Files):
    
    Ana = Analysis()
    Ana.ProjectName = "Optimizer"
    Ana.InputSample("SingleTop", Files[1])
    Ana.InputSample("4Tops", Files[0])
    Ana.EventCache = True
    Ana.DumpPickle = True 
    Ana.VerboseLevel = 1
    Ana.Event = Event
    Ana.Launch()


    Ana = Analysis()
    Ana.ProjectName = "Optimizer"
    Ana.InputSample("SingleTop")
    Ana.InputSample("4Tops")
    Ana.DumpHDF5 = True 
    Ana.DataCache = True
    Ana.EventStop = 100
    Ana.VerboseLevel = 1
    Ana.Threads = 4
    Ana.EventGraph = EventGraphTruthTopChildren
    ApplyFeatures(Ana, "TruthChildren")
    Ana.Launch()










    return True
