from styles  import entry
from atomics import *
from base import Metric
import asyncio

br = "/run/media/tnom6927/1.44.1-64570/MRK-4/" 
#"/home/tnom6927/DESY/thesis-data/gnn-model/metrics/"
def make_train(a, b, c):
    am = Metric(True, False, False)
    am.InterpretROOT(br, [i+1 for i in range(200)], [i+1 for i in range(a, b)], "k-", c)

def make_valid(a, b, c):
    am = Metric(False, True, False)
    am.InterpretROOT(br, [i+1 for i in range(200)], [i+1 for i in range(a, b)], "k-", c)

#n = 3
make_train(0, 10, ["MRK-4"])
make_valid(0, 10, ["MRK-4"])
#n+=1
exit()


mdlx  = ["MRK-1"] #, "MRK-2", "MRK-3", "MRK-4", "MRK-5", "MRK-6", "MRK-7", "MRK-8"]
kfold = ["kfold-" + str(i+1) for i in range(10) ]
epoch = ["epoch-" + str(i+1) for i in range(200) ]
#base_dir = "/home/tnom6927/scratch/pckl" 
base_dir = "/run/media/tnom6927/1.44.1-64570/pckl"
entry(mdlx, kfold, epoch, base_dir)

