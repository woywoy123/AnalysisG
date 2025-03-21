from AnalysisG import Analysis
from AnalysisG.core.plotting import TH1F, TH2F, TLine

# event implementation
from AnalysisG.events.bsm_4tops.event_bsm_4tops import BSM4Tops

# truth study
from AnalysisG.selections.mc16.topkinematics.topkinematics import TopKinematics
from AnalysisG.selections.mc16.topmatching.topmatching import TopMatching
from AnalysisG.selections.mc16.childrenkinematics.childrenkinematics import ChildrenKinematics
from AnalysisG.selections.mc16.decaymodes.decaymodes import DecayModes
from AnalysisG.selections.mc16.toptruthjets.toptruthjets import TopTruthJets
from AnalysisG.selections.mc16.topjets.topjets import TopJets
from AnalysisG.selections.mc16.zprime.zprime import ZPrime
from AnalysisG.selections.mc16.parton.parton import Parton

# figures
import topkinematics
import topmatching
import childrenkinematics
import decaymodes
import toptruthjets
import topjets
import zprime
import parton

import pickle

def default(tl):
    tl.Style = r"ATLAS"
    tl.DPI = 250
    tl.TitleSize = 20
    tl.AutoScaling = False
    tl.Overflow = False

    if study == "zprime":
        tl.yScaling = 10*0.75
        tl.xScaling = 15*1.0
    else:
        tl.yScaling = 10*0.75
        tl.xScaling = 15*0.6
    tl.FontSize = 15
    tl.AxisSize = 14


plotting_method = {
#    "zprime"             : zprime,
    "topkinematics"      : topkinematics,
#    "topmatching"        : topmatching,
#    "childrenkinematics" : childrenkinematics,
#    "decaymodes"         : decaymodes,
#    "toptruthjets"       : toptruthjets,
#    "topjets"            : topjets,
#    "parton"             : parton
}


def ExecuteStudy(study, smpls):
    ev = BSM4Tops()

    sel = None
    if study == "topkinematics"      : sel = TopKinematics()
    if study == "topmatching"        : sel = TopMatching()
    if study == "childrenkinematics" : sel = ChildrenKinematics()
    if study == "decaymodes"         : sel = DecayModes()
    if study == "toptruthjets"       : sel = TopTruthJets()
    if study == "topjets"            : sel = TopJets()
    if study == "zprime"             : sel = ZPrime()
    if study == "parton"             : sel = Parton()

    ana = Analysis()
    ana.Threads = 12
    ana.SumOfWeightsTreeName = "sumWeights"
    ana.AddSamples(smpls, "tmp")
    ana.AddEvent(ev, "tmp")
    ana.AddSelection(sel)
    ana.Start()
    return sel

#smpls = "../../test/samples/dilepton/*"

root = "/home/tnom6927/Downloads/mc16/"
for i in plotting_method:
    study = i
    plt_data = True
    gen_data = False
    figure_path = "./Output/"
    for mass in ["1000", "900", "800", "700", "600", "500", "400"]:
        if mass == "ttbar" or mass == "tttt":
            mass_point = mass
            smpls = mass + "/*"
        else:
            mass_point = "Mass." + mass + ".GeV"
            smpls = "ttH_tttt_m" + mass + "/*"

        method = plotting_method[study]
        method.figures.figure_path = figure_path
        method.figures.mass_point  = mass_point
        method.figures.default     = default
        method.study               = i

        if gen_data:
            sel = ExecuteStudy(study, root + smpls)
            f = open("pkl-data/" + study + "-" + mass_point + ".pkl", "wb")
            pickle.dump(sel, f)
            f.close()

        if not plt_data: continue
        print("plotting: " + study)
        f = open("pkl-data/" + study + "-" + mass_point + ".pkl", "rb")
        pres = pickle.load(f)
        f.close()

        mp = mass_point.replace("GeV", "(GeV)").replace(".", " ")
        if study == "topkinematics"      : method.figures.TopKinematics(pres, mp)
        if study == "topmatching"        : method.figures.TopMatching(pres)
        if study == "childrenkinematics" : method.figures.ChildrenKinematics(pres, mp)
        if study == "decaymodes"         : method.figures.DecayModes(pres)
        if study == "toptruthjets"       : method.figures.TopTruthJets(pres)
        if study == "topjets"            : method.figures.TopJets(pres)
        if study == "zprime"             : method.figures.ZPrime(pres)
        if study == "parton"             : method.figures.Parton(pres)

