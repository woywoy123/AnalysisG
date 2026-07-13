from AnalysisG import Analysis
from AnalysisG.selections.performance.topefficiency.topefficiency import TopEfficiency
from AnalysisG.metrics.topefficiency.metric_topefficiency import TopEfficiencyMetric
from AnalysisG.events.gnn import EventGNN
from AnalysisG.core.tools import Tools
from analysis import *

class track:

    def __init__(self, fnames = None):
        self.fname = {}
        if fnames is None: return 
        for k in fnames: self.fname[i.replace("\n", "")] = False
    
    def run_this(self, fname):
        try: return self.fname[fname]
        except: pass
        self.fname[fname] = False
        return True

    def dump(self):
        fnames = list(self.fname)
        f = open("read.txt", "w")
        f.write("\n".join(fnames))
        f.close()


data_path = "/home/tnom6927/scratch/*"


tl = Tools().ls(data_path, ".root")
for i in tl:
 
    try: tr = track(open("lines.txt", "r").readlines())
    except: tr = track()
    if not tr.run_this(i): continue

    dset = i.split("/")[-2] 
    ev = EventGNN()
    sel = TopEfficiency()
 
    ana = Analysis()
    ana.Threads = 2
    #ana.DebugMode = True
    ana.AddEvent(ev, i)
    ana.AddSamples(i, i)
    ana.AddSelection(sel)
    ana.SaveSelectionToROOT = True
    ana.Start()

    tr.dump() 


mx = TopEfficiencyMetric()
mx.InterpretROOT("/home/tnom6927/scratch/", [103], [9], "k-", "Grift-")
entry(mx)



#entry(selttbar, seltttt)



