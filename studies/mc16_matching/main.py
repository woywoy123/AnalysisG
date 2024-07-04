from AnalysisG.generators.analysis import Analysis
from AnalysisG.events.event_bsm_4tops import BSM4Tops
from AnalysisG.core.selection_template import SelectionTemplate
from AnalysisG.selections.mc16.topkinematics.topkinematics import TopKinematics
from AnalysisG.core.plotting import TH1F, TH2F, TLine
import pickle

#import topkinematics
#
#figure_path = "../../../docs/source/studies/truth-matching/mc16/"
#topkinematics.figures.figure_path = figure_path

smpls = "../../test/samples/dilepton/*"
gen = False

if gen:
    ev = BSM4Tops()
    sel = TopKinematics()

    ana = Analysis()
    ana.AddSamples(smpls, "tmp")
    ana.AddEvent(ev, "tmp")
    ana.AddSelection(sel)
    ana.Start()

    pres = {}
    pres["res_top_kinematics"] = sel.res_top_kinematics
    pres["spec_top_kinematics"] = sel.spec_top_kinematics
    pres["mass_combi"] = sel.mass_combi
    pres["deltaR"] = sel.deltaR

    f = open("dump.pkl", "wb")
    pickle.dump(pres, f)
    f.close()

f = open("dump.pkl", "rb")
pres = pickle.load(f)
f.close()


res_top_kinematics  = pres["res_top_kinematics"]
spec_top_kinematics = pres["spec_top_kinematics"]
mass_combi          = pres["mass_combi"]
deltaR              = pres["deltaR"]

th = TH1F()
th.Title = "Energy"
th.xData = res_top_kinematics[b"energy"]
th.Filename = "figure.x"
#th.HistFill = "step"
#th.ErrorBars = True
th.ApplyScaling = True
th.xMin = 0
th.CrossSection = (1.0665e-5)*10e-6

th.xMax = 1000
th.xBins = 100
th.SaveFigure()


