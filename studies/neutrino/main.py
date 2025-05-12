from AnalysisG import Analysis
from AnalysisG.events.bsm_4tops import BSM4Tops
from AnalysisG.selections.neutrino.validation.validation import Validation
from AnalysisG.selections.neutrino.combinatorial.combinatorial import NuNuCombinatorial
import validation.figures as valid
from combinatorial.figures import combinatorial
from pathlib import Path
import pickle

#pax = "/home/tnom6927/Downloads/mc16/ttbar/user.tnommens.40945479._000013.output.root"
#ev = BSM4Tops()
#sl = Validation()

sl = NuNuCombinatorial()
#ana = Analysis()
##ana.DebugMode = True
#ana.Threads = 1
#ana.SaveSelectionToROOT = True
#ana.AddSamples(pax, "ttbar")
#ana.AddEvent(ev, "ttbar")
#ana.AddSelection(sl)
#ana.Start()

#sl.InterpretROOT("./ProjectName/Selections/" + sl.__name__() + "-" + ev.__name__() + "/ttbar/", "nominal")

#for i in [None]: #, 400]:
#    valid.eps = i
#    valid.validation(None) # load cache
#combinatorial(sl)

pf = Path("./pkl-combinatorial")


def default(hist, pth):
    hist.Style = "ATLAS"
    if pth is not None: hist.OutputDirectory = "figures/" + pth
    hist.DPI = 300
    hist.TitleSize = 15
    hist.AutoScaling = True
    hist.Overflow = False
    hist.yScaling = 5 #10*0.75
    hist.xScaling = 5 #15*0.6
    hist.FontSize = 10
    hist.AxisSize = 10
    hist.LegendSize = 10
    return hist

data = {}
for i in pf.rglob("*pkl"):
    i = str(i)
    prc = i.split("/")[1]
    mode = i.split("/")[2]
    name = i.split("/")[3]
    if prc not in data: data[prc] = {}
    if mode not in data[prc]: data[prc][mode] = {}
    data[prc][mode][name] = i


xl = ["aqua", "orange", "green","blue","olive","teal","gold"]
for prc in data:
    for mode in data[prc]:
        for name in data[prc][mode]:
            print(name, data[prc][mode][name])
            th = pickle.load(open(data[prc][mode][name], "rb"))
            default(th, prc + "/" + mode)
            tmp = th.Title
            tmp = tmp.replace("(TopChildren)", "")
            tmp = tmp.replace("(TruthJets)", "")
            tmp = tmp.replace("(JetsChildren)", "")
            tmp = tmp.replace("(JetsLeptons)", "")
            if "nominal" in tmp: 
                tmp = tmp.replace("(nominal)", "") + " Pre-$\\Delta R$ Minimization"
                th.yLogarithmic = True
                th.yMin = 1
                for x in range(len(th.Histograms)): th.Histograms[x].Color = xl[x]
            else: continue
            th.Title = tmp
            if type(th).__name__ == "TH2F": pass
            th.SaveFigure()
