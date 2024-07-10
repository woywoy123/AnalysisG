from AnalysisG.generators.analysis import Analysis
from AnalysisG.core.plotting import TH1F, TH2F, TLine

# event implementation
from AnalysisG.events.event_bsm_4tops import BSM4Tops

# truth study
from AnalysisG.selections.mc16.topkinematics.topkinematics import TopKinematics
from AnalysisG.selections.mc16.topmatching.topmatching import TopMatching
from AnalysisG.selections.mc16.childrenkinematics.childrenkinematics import ChildrenKinematics

# figures
import topkinematics
import topmatching
import childrenkinematics

import pickle

study = "childrenkinematics"

plotting_method = {
    "topkinematics"      : topkinematics,
    "topmatching"        : topmatching,
    "childrenkinematics" : childrenkinematics
}

smpls = "../../test/samples/dilepton/*"

gen_data = True
figure_path = "./Output/"
for mass in ["1000", "900", "800", "700", "600", "500", "400"]:
    mass_point = "Mass." + mass + ".GeV"
    smpls = "/home/tnom6927/Downloads/DileptonCollection/m" + mass + "/*"

    method = plotting_method[study]
    method.figures.figure_path = figure_path
    method.figures.mass_point  = mass_point

    if gen_data:
        ev = BSM4Tops()

        sel = None
        if study == "topkinematics"      : sel = TopKinematics()
        if study == "topmatching"        : sel = TopMatching()
        if study == "childrenkinematics" : sel = ChildrenKinematics()

        ana = Analysis()
        ana.AddSamples(smpls, "tmp")
        ana.AddEvent(ev, "tmp")
        ana.AddSelection(sel)
        ana.Start()

        f = open(mass_point + ".pkl", "wb")
        pickle.dump(sel, f)
        f.close()

    f = open(mass_point + ".pkl", "rb")
    pres = pickle.load(f)
    f.close()

    if study == "topkinematics"      : method.figures.TopKinematics(pres)
    if study == "topmatching"        : method.figures.TopMatching(pres)
    if study == "childrenkinematics" : method.figures.ChildrenKinematics(pres)

