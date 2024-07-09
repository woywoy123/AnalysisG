from AnalysisG.generators.analysis import Analysis
from AnalysisG.core.plotting import TH1F, TH2F, TLine

# event implementation
from AnalysisG.events.event_bsm_4tops import BSM4Tops

# truth study
from AnalysisG.selections.mc16.topkinematics.topkinematics import TopKinematics
from AnalysisG.selections.mc16.topmatching.topmatching import TopMatching

# figures
import topkinematics
import topmatching

import pickle

study = "topmatching"

plotting_method = {
    "topkinematics" : topkinematics,
    "topmatching"   : topmatching,
}

smpls = "../../test/samples/dilepton/*"

gen_data = False
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
        if study == "topkinematics": sel = TopKinematics()
        if study == "topmatching"  : sel = TopMatching()

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

    if study == "topkinematics": method.figures.TopKinematics(pres)
    if study == "topmatching"  : method.figures.TopMatching(pres)

#th = TH1F()
#th.Title = "Energy"
#th.xData = res_top_kinematics[b"energy"]
#th.Filename = "figure.x"
#th.HistFill = "step"
#th.ErrorBars = True
#th.ApplyScaling = True
#th.xMin = 0
#th.CrossSection = (1.0665e-5)*10e-6
#
#th.xMax = 1000
#th.xBins = 100
#th.SaveFigure()
#

