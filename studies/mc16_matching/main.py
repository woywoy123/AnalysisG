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
    tl.UseLateX = True

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
#    "topkinematics"      : topkinematics,
#    "childrenkinematics" : childrenkinematics,
#    "topmatching"        : topmatching,
#    "decaymodes"         : decaymodes,
#    "toptruthjets"       : toptruthjets,
#    "topjets"            : topjets,
    "parton"             : parton
}


def ExecuteStudy(study, smpls, name, exe = True):
    ev = BSM4Tops()

    sel = None
    if study == "topkinematics"      : sel = TopKinematics()
    if study == "zprime"             : sel = ZPrime()
    if study == "childrenkinematics" : sel = ChildrenKinematics()
    if study == "topmatching"        : sel = TopMatching()
    if study == "decaymodes"         : sel = DecayModes()
    if study == "toptruthjets"       : sel = TopTruthJets()
    if study == "topjets"            : sel = TopJets()
    if study == "parton"             : sel = Parton()

    px  = "./ProjectName/Selections/"
    px += sel.__name__() + "-" + ev.__name__()
    px += "/" + name

    if not exe and study == "decaymodes"  : return sel.load(name = name)
    if not exe and study == "toptruthjets": return sel.load(name = name)
    if not exe and study == "topjets"     : return sel.load(name = name)
    if not exe and study == "parton"      : return sel.load(name = name)
    if not exe: return sel.InterpretROOT(px, "nominal")

    ana = Analysis()
    ana.Threads = 12
    ana.SaveSelectionToROOT = True
#    ana.DebugMode = True
    ana.SumOfWeightsTreeName = "sumWeights"
    ana.AddSamples(smpls, name)
    ana.AddEvent(ev, name)
    ana.AddSelection(sel)
    ana.Start()

    if study == "decaymodes"  : sel.dump(name = name); return sel
    if study == "toptruthjets": sel.dump(name = name); return sel
    if study == "topjets"     : sel.dump(name = name); return sel
    if study == "parton"      : sel.dump(name = name); return sel
    return sel.InterpretROOT(px, "nominal")

masses = ["1000"] #, "900", "800", "700", "600", "500", "400"]
masses.reverse()

root = "/home/tnom6927/Downloads/mc16/"
for i in plotting_method:
    study = i
    plt_data = True
    gen_data = True
    figure_path = "./Output/"
    tmp = {}
    for mass in masses: 
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

        sel = ExecuteStudy(study, root + smpls, study + "-" + mass, gen_data)
        if not plt_data: continue
        if study == "zprime": 
            tmp[mass] = sel
            if len(tmp) < len(masses): continue
            sel = tmp

        print("plotting: " + study)
        mp = mass_point.replace("GeV", "(GeV)").replace(".", " ")
        if study == "topkinematics"      : method.figures.TopKinematics(sel, mp)
        if study == "topmatching"        : method.figures.TopMatching(sel)
        if study == "childrenkinematics" : method.figures.ChildrenKinematics(sel, mp)
        if study == "decaymodes"         : method.figures.DecayModes(sel)
        if study == "toptruthjets"       : method.figures.TopTruthJets(sel)
        if study == "topjets"            : method.figures.TopJets(sel)
        if study == "zprime"             : method.figures.ZPrime(sel)
        if study == "parton"             : method.figures.Parton(sel)

