from Functions.IO.IO import UnpickleObject
from Functions.GNN.Metrics import Metrics



def TestReadTraining(modelname):
    M = Metrics(modelname, "_Models") 
    M.PlotStats()


    return True

