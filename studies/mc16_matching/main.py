from AnalysisG.generators.analysis import Analysis
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

# figures
import topkinematics
import topmatching
import childrenkinematics
import decaymodes
import toptruthjets
import topjets
import zprime

import pickle

study = "childrenkinematics"

plotting_method = {
    "zprime"             : zprime,
    "topmatching"        : topmatching,
    "topkinematics"      : topkinematics,
    "childrenkinematics" : childrenkinematics,
    "decaymodes"         : decaymodes,
    "toptruthjets"       : toptruthjets,
    "topjets"            : topjets
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

    ana = Analysis()
    ana.AddSamples(smpls, "tmp")
    ana.AddEvent(ev, "tmp")
    ana.AddSelection(sel)
    ana.Start()
    return sel

#smpls = "../../test/samples/dilepton/*"

plt_data = True
gen_data = False
figure_path = "./Output/"
for mass in ["1000", "900", "800", "700", "600", "500", "400"]:
    mass_point = "Mass." + mass + ".GeV"
    smpls = "/home/tnom6927/Downloads/mc16/MadGraphPythia8EvtGen_noallhad_ttH_tttt_m" + mass + "/*"

    method = plotting_method[study]
    method.figures.figure_path = figure_path
    method.figures.mass_point  = mass_point

    if gen_data:
        sel = ExecuteStudy(study, smpls)
        f = open(study + "-" + mass_point + ".pkl", "wb")
        pickle.dump(sel, f)
        f.close()


    if not plt_data: continue
    print("plotting: " + study)
    f = open(study + "-" + mass_point + ".pkl", "rb")
    pres = pickle.load(f)
    f.close()

    if study == "topkinematics"      : method.figures.TopKinematics(pres)
    if study == "topmatching"        : method.figures.TopMatching(pres)
    if study == "childrenkinematics" : method.figures.ChildrenKinematics(pres)
    if study == "decaymodes"         : method.figures.DecayModes(pres)
    if study == "toptruthjets"       : method.figures.TopTruthJets(pres)
    if study == "topjets"            : method.figures.TopJets(pres)
    if study == "zprime"             : method.figures.ZPrime(pres)


