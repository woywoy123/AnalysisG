from AnalysisG.generators.analysis import Analysis

# event implementation
from AnalysisG.events.ssml_mc20 import SSML_MC20

# truth study
from AnalysisG.selections.mc20.topkinematics.topkinematics_mc20 import TopKinematics
#from AnalysisG.selections.mc20.topmatching.topmatching import TopMatching
#from AnalysisG.selections.mc20.childrenkinematics.childrenkinematics import ChildrenKinematics
#from AnalysisG.selections.mc20.decaymodes.decaymodes import DecayModes
#from AnalysisG.selections.mc20.toptruthjets.toptruthjets import TopTruthJets

# figures
import topkinematics
import topmatching
import childrenkinematics
import decaymodes
import toptruthjets

import pickle

study = "topkinematics"

plotting_method = {
    "topkinematics"      : topkinematics,
#    "topmatching"        : topmatching,
#    "childrenkinematics" : childrenkinematics,
#    "decaymodes"         : decaymodes,
#    "toptruthjets"       : toptruthjets
}

smpls = ""

gen_data = True
figure_path = "./Output/"
for mass in ["1000", "900", "800", "700", "600", "500", "400"]:
    mass_point = "Mass." + mass + ".GeV"
#    smpls = "/home/tnom6927/Downloads/mc16/MadGraphPythia8EvtGen_noallhad_ttH_tttt_m" + mass + "/*"

    method = plotting_method[study]
    method.figures.figure_path = figure_path
    method.figures.mass_point  = mass_point

    if gen_data:
        ev = SSML_MC20()

        sel = None
        if study == "topkinematics"      : sel = TopKinematics()
        if study == "topmatching"        : sel = TopMatching()
        if study == "childrenkinematics" : sel = ChildrenKinematics()
        if study == "decaymodes"         : sel = DecayModes()
        if study == "toptruthjets"       : sel = TopTruthJets()

        ana = Analysis()
        ana.AddSamples(smpls, "tmp")
        ana.AddEvent(ev, "tmp")
        ana.AddSelection(sel)
        ana.Start()

        f = open(study + "-" + mass_point + ".pkl", "wb")
        pickle.dump(sel, f)
        f.close()

    f = open(study + "-" + mass_point + ".pkl", "rb")
    pres = pickle.load(f)
    f.close()

    print("plotting: " + study)
    if study == "topkinematics"      : method.figures.TopKinematics(pres)
    if study == "topmatching"        : method.figures.TopMatching(pres)
    if study == "childrenkinematics" : method.figures.ChildrenKinematics(pres)
    if study == "decaymodes"         : method.figures.DecayModes(pres)
    if study == "toptruthjets"       : method.figures.TopTruthJets(pres)

    break
