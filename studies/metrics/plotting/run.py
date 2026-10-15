from AnalysisG.core import Tools
from base import Metric
from atomics import *
import sys

#br = "/home/tnom6927/scratch/" 
br = "/home/tnom6927/DESY/thesis-data/gnn-model/metrics/"
#br = "/mnt/perl/"
ak = "/run/media/tnom6927/1.44.1-64570/pckl/"
def make_train(a, b, c):
    tl = Tools()
    epx, klx = a + 1, b + 1
    sx = ak + c + "/training/epoch-" + str(epx) + "/kfold-" + str(klx) + ".pkl"
    if tl.is_file(sx): return
    print(sx)
    am = Metric(True, False, False)
    am.InterpretROOT(br, [epx], [klx], "k-", [c])

def make_valid(a, b, c):
    tl = Tools()
    epx, klx = a + 1, b + 1
    sx = ak + c + "/validation/epoch-" + str(epx) + "/kfold-" + str(klx) + ".pkl"
    if tl.is_file(sx): return
    print(sx)
    am = Metric(False, True, False)
    am.InterpretROOT(br, [epx], [klx], "k-", [c])


a, b, c = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
mdl = "MRK-" + str(c + 1)
make_train(a, b, mdl)
make_valid(a, b, mdl)
