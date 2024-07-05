from AnalysisG.generators.analysis import Analysis
from AnalysisG.core.plotting import TH1F, TH2F, TLine

# event implementation
from AnalysisG.events.event_bsm_4tops import BSM4Tops

# truth study
from AnalysisG.selections.mc16.topkinematics.topkinematics import TopKinematics
from AnalysisG.selections.mc16.topmatching.topmatching import TopMatching

# figures
import topkinematics

import pickle

gen_data = True
figure_path = "./Output/"
for mass in ["1000", "900", "800", "700", "600", "500", "400"]:
    mass_point = "Mass." + mass + ".GeV"
    topkinematics.figures.figure_path = figure_path
    topkinematics.figures.mass_point = mass_point
    #smpls = "/home/tnom6927/Downloads/DileptonCollection/m" + mass + "/*"

    smpls = "../../test/samples/dilepton/*"

    if gen_data:
        ev = BSM4Tops()
        #sel = TopKinematics()
        sel = TopMatching()

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

    #topkinematics.figures.TopKinematics(pres)
    print(pres.truth_top)

    break

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

