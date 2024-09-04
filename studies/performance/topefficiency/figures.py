from AnalysisG.core.plotting import TH1F, TH2F, TLine
from pathlib import Path
#import torch
import pickle

global figure_path

def mapping(name):
    if "_singletop_"  in name: return "singletop"
    if "_ttH125_"     in name: return "ttH"
    if "_ttbarHT1k_"  in name: return "$t\\bar{t}$"
    if "_SM4topsNLO_" in name: return "$t\\bar{t}t\\bar{t}"
    if "_ttbar_"      in name: return "$t\\bar{t}$"
    if "_ttbarHT1k5_" in name: return "$t\\bar{t}$"
    if "_ttbarHT6c_"  in name: return "$t\\bar{t}$"
    if "_Ztautau_"    in name: return "$Z\\ell\\ell$"
    if "_llll_"       in name: return "$\\ell\\ell\\ell\\ell$"
    if "_lllv_"       in name: return "$\\ell\\ell\\ell\\nu$"
    if "_llvv_"       in name: return "$\\ell\\ell\\nu\\nu$"
    if "_lvvv_"       in name: return "$\\ell\\nu\\nu\\nu$"
    if "_tchan_"      in name: return "tchan"
    if "_tt_"         in name: return "tt"
    if "_ttee_"       in name: return "ttll"
    if "_ttmumu_"     in name: return "ttll"
    if "_tttautau_"   in name: return "ttll"
    if "_ttW_"        in name: return "ttW"
    if "_ttZnunu_"    in name: return "ttZvv"
    if "_ttZqq_"      in name: return "ttZqq"
    if "_tW_"         in name: return "tW"
    if "_tZ_"         in name: return "tZ"
    if "_Wenu_"       in name: return "Wev"
    if "_WH125_"      in name: return "WH"
    if "_WlvZqq_"     in name: return "WlvZqq"
    if "_Wmunu_"      in name: return "Wlv"
    if "_WplvWmqq_"   in name: return "WplvWmqq"
    if "_WpqqWmlv_"   in name: return "WpqqWmlv"
    if "_WqqZll_"     in name: return "WqqZll"
    if "_WqqZvv_"     in name: return "WqqZvv"
    if "_Wt_"         in name: return "Wt"
    if "_Wtaunu_"     in name: return "Wlv"
    if "_Zee_"        in name: return "Zll"
    if "_ZH125_"      in name: return "ZH"
    if "_Zmumu_"      in name: return "Zll"
    if "_ZqqZll_"     in name: return "ZqqZll"
    if "_ZqqZvv_"     in name: return "ZqqZvv"
    return "ndef"


def path(hist):
    hist.OutputDirectory = figure_path + "/topefficiency"
    return hist

def TopMassComparison(stacks, ana = None):
    if ana is not None:
        data_p = ana.p_topmass
        for r in data_p:
            if r not in stacks: stacks[r] = {}
            for k in data_p[r]:
                if k not in stacks[r]: stacks[r][k] = {"truth" : [], "pred" : []}
                stacks[r][k]["pred"] += list(data_p[r][k])

        data_t = ana.t_topmass
        for r in data_t:
            if r not in stacks: stacks[r] = {}
            for k in data_t[r]:
                if k not in stacks[r]: stacks[r][k] = {"truth" : [], "pred" : []}
                stacks[r][k]["truth"] += list(data_t[r][k])
        return stacks

    for r in stacks:
        hists = {"truth" : []}
        for k in stacks[r]:
            prc = mapping(k)
            if prc not in hists: hists[prc] = []
            hists[prc] += stacks[r][k]["pred"]
            hists["truth"] += stacks[r][k]["truth"]

        hist_t = TH1F()
        hist_t.Title = "Truth Tops"
        hist_t.xData = hists["truth"]
        del hists["truth"]

        for k in hists:
            tk = TH1F()
            tk.Title = k
            tk.xData = hists[k]
            hists[k] = tk

        hist = TH1F()
        hist.Histogram = hist_t
        hist.Histograms = list(hists.values())
        hist.Title = "Kinematic Region " + r.replace("eta", "\eta")
        hist.xTitle = "Invariant Top Mass (GeV)"
        hist.yTitle = "Entries / 2 GeV"
        hist.Style = "ATLAS"
        hist.xMin = 0
        hist.xMax = 400
        hist.xBins = 200
        hist.xStep = 20
        hist.Overflow = False

        kins = r.split(",")
        kins = [f.replace(" ", "").replace("$", "").replace("_{top}", "").replace("|", "").replace("<", "_") for f in kins]
        hist.OutputDirectory = figure_path + "/top-reconstruction/" + kins[0]
        hist.Filename = kins[1]
        hist.SaveFigure()


def DecayMode(stacks, ana = None):
    if ana is not None:
        data_p = ana.p_decaymode_topmass
        for dm in data_p:
            if dm not in stacks: stacks[dm] = {}
            for r in data_p[dm]:
                if r not in stacks[dm]: stacks[dm][r] = {}
                for p in data_p[dm][r]:
                    if p not in stacks[dm][r]: stacks[dm][r][p] = {"truth" : [], "pred" : []}
                    stacks[dm][r][p]["pred"] += data_p[dm][r][p]

        data_t = ana.t_decaymode_topmass
        for dm in data_t:
            if dm not in stacks: stacks[dm] = {}
            for r in data_t[dm]:
                if r not in stacks[dm]: stacks[dm][r] = {}
                for p in data_t[dm][r]:
                    if p not in stacks[dm][r]: stacks[dm][r][p] = {"truth" : [], "pred" : []}
                    stacks[dm][r][p]["truth"] += data_t[dm][r][p]
        return stacks

    for mode in stacks:
        for kin in stacks[mode]:
            hists = {"truth" : []}
            for k in stacks[mode][kin]:
                prc = mapping(k)
                if prc not in hists: hists[prc] = []
                hists[prc] += stacks[mode][kin][k]["pred"]
                hists["truth"] += stacks[mode][kin][k]["truth"]

            hist_t = TH1F()
            hist_t.Title = "Truth Tops"
            hist_t.xData = hists["truth"]
            del hists["truth"]

            for k in hists:
                tk = TH1F()
                tk.Title = k
                tk.xData = hists[k]
                hists[k] = tk
            mx = "had-" + str(mode.count("h")) + "_lep-" + str(mode.count("l")) + "_nbjets-" + mode.split("-")[-1]

            hist = TH1F()
            hist.Histogram = hist_t
            hist.Histograms = list(hists.values())
            hist.Title = "Kinematic Region: " + kin.replace("eta", "\eta") + " \n channel: " + mx.replace("_", " ")
            hist.xTitle = "Invariant Top Mass (GeV)"
            hist.yTitle = "Entries / 2 GeV"
            hist.Style = "ATLAS"
            hist.xMin = 0
            hist.xMax = 400
            hist.xBins = 200
            hist.xStep = 20
            hist.Overflow = False

            kins = kin.split(",")
            kins = [f.replace(" ", "").replace("$", "").replace("_{top}", "").replace("|", "").replace("<", "_") for f in kins]
            hist.OutputDirectory = figure_path + "/decaymode/" + mx + "/" + kins[0]
            hist.Filename = kins[1]
            hist.SaveFigure()

def TopScores(stacks, ana = None):
    if ana is not None:
        data_s = ana.prob_tops
        data_m = ana.p_topmass
        for dm in data_s:
            kins = dm.split(",")
            kins = [f.replace(" ", "").replace("$", "").replace("_{top}", "").replace("|", "").replace("<", "_") for f in kins]
            kins = kins[0]
            if kins not in stacks: stacks[kins] = {}
            for r in data_s[dm]:
                prc = mapping(r)
                if prc not in stacks[kins]: stacks[kins][prc] = {"score" : [], "mass" : []}
                stacks[kins][prc]["score"] += data_s[dm][r]
                stacks[kins][prc]["mass"]  += data_m[dm][r]
        return stacks

    for kin in stacks:
        for prc in stacks[kin]:
            th2 = TH2F()
            th2.xData = stacks[kin][prc]["mass"]
            th2.yData = stacks[kin][prc]["score"]
            th2.xBins = 100
            th2.yBins = 100
            th2.xMin = 0
            th2.yMin = 0
            th2.yMax = 1
            th2.xMax = 400
            th2.Title = "Invariant Mass vs Score of Reconstructed Top-Quark"
            th2.xTitle = "Invariant Mass of Top / 4 GeV"
            th2.yTitle = "Reconstruction Top Score"
            th2.OutputDirectory = figure_path + "/topscore/"
            th2.Filename = kin + "_" + prc
            th2.xStep = 20
            th2.yStep = 0.1
            th2.SaveFigure()


def EfficiencyEvent(stacks, ana = None):
    if ana is not None:
        data_p = ana.purity_tops
        data_e = ana.efficiency_tops
        for i in data_p:
            k = mapping(i)
            if k not in stacks: stacks[k] = {"x-data" : [], "y-data" : []}
            stacks[k]["y-data"] += data_p[i]
            stacks[k]["x-data"] += data_e[i]
        return stacks

    for i in stacks:
        tl = TLine()
        tl.Title = i
        tl.xData = stacks[i]["x-data"]
        tl.yData = stacks[i]["y-data"]
        tl.ErrorBars = True
        stacks[i] = tl

    tl = TLine()
    tl.Lines = list(stacks.values())
    tl.Title = "Purity of Top Reconstruction as a function of Efficiency"
    tl.xTitle = "$\\varepsilon_{tops}$"
    tl.yTitle = "purity"
    tl.xMax = 1
    tl.yMax = 1
    tl.xMin = 0
    tl.yMin = 0
    tl.OutputDirectory = figure_path + "/Efficiency"
    tl.Filename = "purity_epsilon"
    tl.SaveFigure()

def TopEfficiency(ana):
    p = Path(ana)
    files = [str(x) for x in p.glob("**/*.pkl") if str(x).endswith(".pkl")]
    files = list(set(files))
    stacks_c = {}
    stacks_t = {}
    stacks_s = {}
    stacks_x = {}
    for i in range(len(files)):
        pr = pickle.load(open(files[i], "rb"))
        print(files[i], (i+1) / len(files))
        stacks_x = EfficiencyEvent(stacks_x, pr)
        stacks_s = TopScores(stacks_s, pr)
        stacks_c = TopMassComparison(stacks_c, pr)
        #stacks_t = DecayMode(stacks_t, pr)
    EfficiencyEvent(stacks_x)
    TopScores(stacks_s)
    TopMassComparison(stacks_c)
    #DecayMode(stacks_c)
