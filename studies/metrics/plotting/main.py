from styles import entry
from atomics import *

br = "/run/media/tnom6927/1.44.1-64570/"
am = Metric()
#am.InterpretROOT(br, [i+1 for i in range(1)], [i+1 for i in range(2)], "k-", ["MRK-1"], ["validation.root"])
#am = Metric()
#am.InterpretROOT(br, [i+1 for i in range(2)], [i+1 for i in range(1)], "k-", ["MRK-1"], ["training.root"])
#

mdlx  = ["MRK-1", "MRK-2", "MRK-3", "MRK-4", "MRK-5", "MRK-6", "MRK-7", "MRK-8"]
kfold = ["kfold-" + str(i+1) for i in range(10)]
epoch = ["epoch-" + str(i+1) for i in range(200)]
base_dir = "/run/media/tnom6927/1.44.1-64570/pckl/"
entry(mdlx, kfold, epoch, base_dir)


