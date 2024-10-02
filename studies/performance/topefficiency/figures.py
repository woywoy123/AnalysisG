from AnalysisG.core.plotting import TH1F, TH2F, TLine
from pathlib import Path
from .algorithms import *
#import torch
import pickle

global figure_path
def path(hist, subx = ""):
    hist.Style = "ATLAS"
    hist.OutputDirectory = figure_path + "/topefficiency" + subx
    return hist


def top_kinematic_region(stacks, data = None):
    if data is not None: return top_pteta(stacks, data)

    prc_topscore = {}
    top_score_mass = {}
    ks_topscore_eta_pt = {}
    pt_topmass_prc = {"truth" : {}, "prediction" : {}}

    for kin in set(list(stacks["truth"]) + list(stacks["prediction"])):
        pt_r = kin.split(",")[0]
        tru_h = None

        if pt_r not in pt_topmass_prc["truth"]: pt_topmass_prc["truth"][pt_r] = []
        if kin in stacks["truth"]:
            tru_h = TH1F()
            tru_h.Title = "Truth"
            tru_h.xData = stacks["truth"][kin]
            tru_h.Hatch = "\\\\////"
            tru_h.Color = "black"
            pt_topmass_prc["truth"][pt_r] += stacks["truth"][kin]

        hists = []
        if kin not in stacks["prediction"]: stacks["prediction"][kin] = {}

        for prc in stacks["prediction"][kin]:
            prc_h = TH1F()
            prc_h.Title = prc.split("#")[1]
            prc_h.xData = stacks["prediction"][kin][prc]
            hists.append(prc_h)

            if pt_r not in pt_topmass_prc["prediction"]: pt_topmass_prc["prediction"][pt_r] = {}
            if prc not in pt_topmass_prc["prediction"][pt_r]: pt_topmass_prc["prediction"][pt_r][prc] = []
            pt_topmass_prc["prediction"][pt_r][prc] += stacks["prediction"][kin][prc]

            if pt_r not in top_score_mass: top_score_mass[pt_r] = {"mass" : [], "score" : []}
            top_score_mass[pt_r]["mass"] += stacks["prediction"][kin][prc]
            top_score_mass[pt_r]["score"] += stacks["top_score"][kin][prc]

            if pt_r not in prc_topscore: prc_topscore[pt_r] = {}
            if prc not in prc_topscore[pt_r]: prc_topscore[pt_r][prc] = []
            prc_topscore[pt_r][prc] += stacks["top_score"][kin][prc]

        tlt = kin.replace("_", " \\leq p^{top}_T \\leq ")
        tlt = tlt.replace("-", " \\leq | \\eta_{top} | \\leq ")

        reco = path(TH1F(), "/" + kin.split(",")[0])
        reco.Title = "Reconstructed Invariant Mass of Top Candidate within \n Kinematic Region: $" + tlt + "$"
        reco.Histograms = hists
        reco.Histogram = tru_h
        reco.Stacked = True
        reco.xStep = 20
        reco.Overflow = False
        reco.xTitle = "Invariant Mass of Candidate Top (GeV)"
        reco.yTitle = "Entries / ($1$ GeV)"
        reco.xMin = 0
        reco.xMax = 400
        reco.xBins = 400
        reco.Filename = kin.split(", ")[1]
        reco.SaveFigure()

        try: ks = float(reco.KStest(tru_h).pvalue)
        except: ks = 0

        ks_topscore_eta_pt[kin] = ks

    for kin in set(list(pt_topmass_prc["truth"]) + list(pt_topmass_prc["prediction"])):
        tru = TH1F()
        tru.Title = "Truth"
        tru.xData = pt_topmass_prc["truth"][kin]
        tru.Hatch = "\\\\////"
        tru.Color = "black"

        hists = []
        for prc in pt_topmass_prc["prediction"][kin]:
            prc_h = TH1F()
            prc_h.Title = prc.split("#")[1]
            prc_h.xData = pt_topmass_prc["prediction"][kin][prc]
            hists.append(prc_h)

        tlt = kin.replace("_", " \\leq p^{top}_T \\leq ")
        reco = path(TH1F(), "/aggregated-pt/")
        reco.Title = "Reconstructed Invariant Mass of Top Candidate with \n Transverse Momentum: $" + tlt + "$"
        reco.Histograms = hists
        reco.Histogram = tru
        reco.xStep = 20
        reco.Stacked = True
        reco.Overflow = False
        reco.xTitle = "Invariant Mass of Candidate Top (GeV)"
        reco.yTitle = "Entries / ($1$ GeV)"
        reco.xMin = 0
        reco.xMax = 400
        reco.xBins = 400
        reco.Filename = kin.split(", ")[0]
        reco.SaveFigure()

    for kin in prc_topscore:

        hists = []
        for prc in prc_topscore[kin]:
            prc_h = TH1F()
            prc_h.Title = prc.split("#")[1]
            prc_h.xData = prc_topscore[kin][prc]
            hists.append(prc_h)

        tlt = kin.replace("_", " \\leq p^{top}_T \\leq ")
        s_s = path(TH1F(), "/pt-score")
        s_s.Histograms = hists
        s_s.Title = "Reconstructed Top Candidate Score with \n Transverse Momentum $" + tlt + "$"
        s_s.xTitle = "MVA Score of Candidate Top (Arb.)"
        s_s.yTitle = "Entries / ($0.01$)"

        s_s.xMin = 0
        s_s.xMax = 1
        s_s.xBins = 100
        s_s.xStep = 0.05
        s_s.Stacked = True

        s_s.Filename = "mva-score_" + kin
        s_s.SaveFigure()


    for kin in top_score_mass:
        tlt = kin.replace("_", " \\leq p^{top}_T \\leq ")

        mass_s = path(TH2F(), "/score-mass")
        mass_s.Title = "Reconstructed Top Candidate Score as a \n function of Invariant Mass for $" + tlt + "$"
        mass_s.xTitle = "Reconstructed Top Candidate Invariant Mass / ($1$ GeV)"
        mass_s.yTitle = "MVA Score of Candidate Top / ($0.01$)"

        mass_s.xMin = 0
        mass_s.xMax = 400
        mass_s.xBins = 400
        mass_s.xStep = 20

        mass_s.yMin = 0
        mass_s.yMax = 1
        mass_s.yStep = 0.05
        mass_s.yBins = 100

        mass_s.xData = top_score_mass[kin]["mass"]
        mass_s.yData = top_score_mass[kin]["score"]
        mass_s.Filename = "pt_range_" + kin
        mass_s.SaveFigure()

    eta_pt_ks = path(TH2F(), "/statistics")
    eta_pt_ks.Title = "Kolmogorov-Smirnov Test Statistic for Candidate to Truth Top \n Distribution for Various Kinematic Regions"
    eta_pt_ks.xTitle = "Reconstructed Top Candidate $p_T$ / ($100$ GeV)"
    eta_pt_ks.yTitle = "Pseudorapidity of Top Candidate / ($0.05 \\eta$)"

    eta_pt_ks.xMin = 0
    eta_pt_ks.xMax = 1500
    eta_pt_ks.xBins = 15
    eta_pt_ks.xStep = 100

    eta_pt_ks.yMin = 0
    eta_pt_ks.yMax = 6
    eta_pt_ks.yStep = 0.5
    eta_pt_ks.yBins = 12

    eta_pt_ks.xData = [float(k.split(",")[0].split("_")[0]) for k in ks_topscore_eta_pt]
    eta_pt_ks.yData = [float(k.split(",")[1].split("-")[0]) for k in ks_topscore_eta_pt]
    eta_pt_ks.Weights = list(ks_topscore_eta_pt.values())
    eta_pt_ks.Filename = "ks_score_eta_pt"
    eta_pt_ks.SaveFigure()



def TopEfficiency(ana):
    p = Path(ana)
    files = [str(x) for x in p.glob("**/*.pkl") if str(x).endswith(".pkl")]
    files = list(set(files))
#    files = files[:10]

    stack_topkin = {}
    for i in range(len(files)):
        print(files[i], (i+1) / len(files))
        pr = pickle.load(open(files[i], "rb"))
        stack_topkin = top_kinematic_region(stack_topkin, pr)

    top_kinematic_region(stack_topkin)

