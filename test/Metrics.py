from AnalysisTopGNN.IO import UnpickleObject
from AnalysisTopGNN.Tools import Metrics



def TestReadTraining(modelname):
    M = Metrics(modelname, "_Models") 
    M.PlotStats()
    return True

