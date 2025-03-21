from AnalysisG import Analysis

# event implementation
from AnalysisG.events.ssml_mc20 import SSML_MC20

# truth study
from AnalysisG.selections.mc20.topkinematics.topkinematics_mc20 import TopKinematics
from AnalysisG.selections.mc20.topmatching.topmatching_mc20 import TopMatching
from AnalysisG.selections.mc20.zprime.zprime_mc20 import ZPrime

# figures
import topkinematics
import topmatching
import zprime
import pickle

study = "zprime"

plotting_method = {
    "topkinematics"      : topkinematics,
    "topmatching"        : topmatching,
    "zprime"             : zprime
}

gen_data = False
figure_path = "./Output/"
smpls = "/home/tnom6927/Downloads/mc20/*"
method = plotting_method[study]
method.figures.figure_path = figure_path
method.figures.mass_point  = "NULL"

if gen_data:
    ev = SSML_MC20()

    sel = None
    if study == "topkinematics" : sel = TopKinematics()
    if study == "topmatching"   : sel = TopMatching()
    if study == "zprime"        : sel = ZPrime()

    ana = Analysis()
    ana.AddSamples(smpls, "tmp")
    ana.AddEvent(ev, "tmp")
    ana.AddSelection(sel)
    ana.Start()

    f = open(study + "-NONE.pkl", "wb")
    pickle.dump(sel, f)
    f.close()

f = open(study + "-NONE.pkl", "rb")
pres = pickle.load(f)
f.close()

print("plotting: " + study)
if study == "topkinematics" : method.figures.TopKinematics(pres, "NULL")
if study == "topmatching"   : method.figures.TopMatching(pres)
if study == "zprime"        : method.figures.ZPrime(pres)
