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


def TopScores(stacks, ana = None):
    if ana is not None:
        data_s = ana.prob_tops
        data_m = ana.p_topmass
        for dm in data_s:
            kins = dm.split(",")[0].split("<")
            kins = (kins[0] + "-" + kins[2]).replace(" ", "")
            if kins not in stacks: stacks[kins] = {}
            for r in data_s[dm]:
                prc = mapping(r)
                if prc not in stacks[kins]: stacks[kins][prc] = {"score" : [], "mass" : []}
                stacks[kins][prc]["score"] += data_s[dm][r]
                stacks[kins][prc]["mass"]  += data_m[dm][r]
        return stacks

    for kin in stacks:
        if "score-project" in kin: continue
        for prc in stacks[kin]:
            th2 = TH2F()
            th2.xData = stacks[kin][prc]["mass"]
            th2.yData = stacks[kin][prc]["score"]
            th2.yMax = 1
            th2.yMin = 0
            th2.yBins = 100

            th2.xMax = 400
            th2.xMin = 0
            th2.xBins = 100

            th2.Title = "Reconstructed Top Score as a function of Top Candidate Invariant Mass"
            th2.xTitle = "Invariant Mass of Top Candidate / Entries per 1 GeV"
            th2.yTitle = "Score of Top Candidate / Entries per 0.01"
            th2.OutputDirectory = figure_path + "/topscore/" + prc
            th2.Filename = kin
            th2.xStep = 20
            th2.yStep = 0.1
            th2.SaveFigure()

    hists = []
    for prc in set(sum([list(stacks[i]) for i in stacks], [])):
        thx = TH1F()
        thx.Title = prc
        thx.xData = sum([stacks[k][prc]["score"] for k in stacks if prc in stacks[k]], [])
        hists.append(thx)

    tht = TH1F()
    tht.Histograms = hists
    tht.Title = "Top Candidate Score Projection"
    tht.xBins = 110
    tht.xMin = 0
    tht.xMax = 1.1
    tht.xTitle = "Top Candidate Score (Arb.)"
    tht.yTitle = "Top Candidate Score Entries / 0.01"
    tht.OutputDirectory = figure_path + "/topscore/"
    tht.Filename = "top_score_projection"
    tht.xStep = 0.1
    tht.SaveFigure()


def EfficiencyEvent(stacks, ana = None):
    if ana is not None:
        data_tru_ntru  = ana.n_tru_tops
        data_perf_tops = ana.ms_cut_perf_tops
        data_reco_tops = ana.ms_cut_reco_tops
        for i in data_tru_ntru:
            k = mapping(i)
            if k not in stacks: stacks[k] = {"ntruth" : 0}
            stacks[k]["ntruth"] += float(sum(data_tru_ntru[i]))
            for t in data_perf_tops:
                if t not in stacks[k]: stacks[k][t] = {"nperfect" : 0, "nreco" : 0}
                stacks[k][t]["nperfect"] += float(sum(data_perf_tops[t][i]))
                stacks[k][t]["nreco"]    += float(sum(data_reco_tops[t][i]))
        return stacks

    for i in stacks:
        ntru = stacks[i]["ntruth"]
        masses, scores, efficiency, fake, optimal = [], [], [], [], []
        for k in stacks[i]:
            if "ntruth" == k: continue
            del_mass  = float(k.split("-")[0])
            del_score = float(k.split("-")[1])
            if not stacks[i][k]["nreco"]: stacks[i][k]["nreco"] = 1
            masses.append(del_mass)
            scores.append(del_score)
            efficiency.append(stacks[i][k]["nperfect"] / ntru)
            fake.append(1 - stacks[i][k]["nperfect"] / stacks[i][k]["nreco"])
            optimal.append((stacks[i][k]["nperfect"] / stacks[i][k]["nreco"]) * (stacks[i][k]["nperfect"] / ntru))

        sc_ef = TH2F()
        sc_ef.Title = "Reconstruction Top Candidate Efficiency for Inclusion Size $\\Delta_{M} \\times \\Delta_{sc}$"
        sc_ef.xData = masses
        sc_ef.yData = scores
        sc_ef.Weights = efficiency

        sc_ef.xTitle = "Top Invariant Mass $\\Delta$ / 1 GeV"
        sc_ef.yTitle = "Top Candidate Score $\\Delta$ / 0.01"

        sc_ef.xMax = max(masses) + 1
        sc_ef.xMin = min(masses) - 1
        sc_ef.xBins = len(set(masses))+2
        sc_ef.xStep = 10

        sc_ef.yMax = max(scores) + 0.01
        sc_ef.yMin = min(scores) - 0.01
        sc_ef.yBins = len(set(scores))+2
        sc_ef.yStep = 0.02

        sc_ef.OutputDirectory = figure_path + "/Efficiency"
        sc_ef.Filename = "efficiency_" + i
        sc_ef.SaveFigure()

        sc_pu = TH2F()
        sc_pu.Title = "Reconstruction Top Candidate Fake Rate for Inclusion Size $\\Delta_{M} \\times \\Delta_{sc}$"
        sc_pu.xData = masses
        sc_pu.yData = scores
        sc_pu.Weights = fake

        sc_pu.xTitle = "Top Invariant Mass $\\Delta$ / 1 GeV"
        sc_pu.yTitle = "Top Candidate Score $\\Delta$ / 0.01"

        sc_pu.xMax = max(masses) + 1
        sc_pu.xMin = min(masses) - 1
        sc_pu.xBins = len(set(masses))+2
        sc_pu.xStep = 10

        sc_pu.yMax = max(scores) + 0.01
        sc_pu.yMin = min(scores) - 0.01
        sc_pu.yBins = len(set(scores))+2
        sc_pu.yStep = 0.02

        sc_pu.OutputDirectory = figure_path + "/Efficiency"
        sc_pu.Filename = "purity_" + i
        sc_pu.SaveFigure()

        sc_optim = TH2F()
        sc_optim.Title = "Optimal Inclusion Size $\\Delta_{M} \\times \\Delta_{sc}$ for Top Candidate " 
        sc_optim.Title += "$\\epsilon_{\\text{top}} \\times \\rho_{\\text{top}}$ (Efficiency x Purity)"
        sc_optim.xData = masses
        sc_optim.yData = scores
        sc_optim.Weights = optimal

        sc_optim.xTitle = "Top Invariant Mass $\\Delta$ / 1 GeV"
        sc_optim.yTitle = "Top Candidate Score $\\Delta$ / 0.01"

        sc_optim.xMax = max(masses) + 1
        sc_optim.xMin = min(masses) - 1
        sc_optim.xBins = len(set(masses))+2
        sc_optim.xStep = 10

        sc_optim.yMax = max(scores) + 0.01
        sc_optim.yMin = min(scores) - 0.01
        sc_optim.yBins = len(set(scores))+2
        sc_optim.yStep = 0.02

        sc_optim.OutputDirectory = figure_path + "/Efficiency"
        sc_optim.Filename = "optim_" + i
        sc_optim.SaveFigure()

def KinematicTops(stacks, ana = None):
    if ana is not None:
        truth_tops = ana.kin_truth_tops
        reco_tops  = ana.ms_kin_reco_tops
        perf_tops  = ana.ms_kin_perf_tops

        for kin in truth_tops:
            for fname in truth_tops[kin]:
                prc = mapping(fname)
                if prc not in stacks: stacks[prc] = {"ntruth" : {}, "reco" : {}, "perfect" : {}}
                if kin not in stacks[prc]["ntruth"]: stacks[prc]["ntruth"][kin] = 0
                stacks[prc]["ntruth"][kin] += sum(truth_tops[kin][fname])

        for para in reco_tops:
            for kin in reco_tops[para]:
                for fname in reco_tops[para][kin]:
                    prc = mapping(fname)
                    if para not in stacks[prc]["reco"]:    stacks[prc]["reco"][para] = {}
                    if para not in stacks[prc]["perfect"]: stacks[prc]["perfect"][para] = {}

                    if kin not in stacks[prc]["reco"][para]:    stacks[prc]["reco"][para][kin] = 0
                    if kin not in stacks[prc]["perfect"][para]: stacks[prc]["perfect"][para][kin] = 0

                    stacks[prc]["reco"][para][kin]    += sum(reco_tops[para][kin][fname])
                    try: stacks[prc]["perfect"][para][kin] += sum(perf_tops[para][kin][fname])
                    except KeyError: pass

        return stacks

    for prc in stacks:
        ntruth = stacks[prc]["ntruth"]
        reco   = stacks[prc]["reco"]
        perf   = stacks[prc]["perfect"]
        effic, pur = [], []
        integral, purity_efficiency = {}, {}
        for para in reco:
            pt_purity = {pt*100 : 0 for pt in range(16)}
            pt_efficiency = {pt*100 : 0 for pt in range(16)}

            integral[para] = 0
            purity_efficiency[para] = {"purity" : [], "efficiency" : []}

            for kin in reco[para]:

                try: tps = ntruth[kin]
                except KeyError: tps = 0
                try: rct = reco[para][kin]
                except KeyError: rct = 0
                try: pct = perf[para][kin]
                except KeyError: pct = 0

                kin = float(kin.split("<")[0])
                pt_purity[kin]     = float(pct) / float(rct) if rct else 0
                pt_efficiency[kin] = float(pct) / float(tps) if tps else 0

                integral[para] += pt_purity[kin]*pt_efficiency[kin]/float(len(pt_purity))
                purity_efficiency[para]["purity"].append(pt_purity[kin])
                purity_efficiency[para]["efficiency"].append(pt_efficiency[kin])

            keys = sorted(pt_purity)
            if len(pur) < 10:
                line = TLine()
                line.Title = "$\\Delta M = $" + para.split("-")[0] + ", $\\Delta S = $" + para.split("-")[1]
                line.xData = keys
                line.yData = [pt_purity[k] for k in keys]
                line.Marker = ""
                pur.append(line)

                le = TLine()
                le.Title = "$\\Delta M = $" + para.split("-")[0] + ", $\\Delta S = $" + para.split("-")[1]
                le.xData = keys
                le.yData = [pt_efficiency[k] for k in keys]
                le.Marker = ""
                effic.append(le)

        tl = TLine()
        tl.Lines = pur
        tl.Title = "Purity of Reconstructed Top Candidates in $p_T$ Kinematic Regions \n using Inclusion Window $\\Delta M \\times \\Delta S$"
        tl.xTitle = "$p_T$ of Reconstructed Top (GeV)"
        tl.yTitle = "Purity of Reconstructed Tops ($t_{\\text{Perfect}} / t_{\\text{Candidates}}$)"
        tl.xMax = 1600
        tl.yMax = 1.01
        tl.xMin = 0
        tl.yMin = 0
        tl.xStep = 100
        tl.OutputDirectory = figure_path + "/Efficiency"
        tl.Filename = prc + "-purity"
        tl.SaveFigure()

        tle = TLine()
        tle.Lines = effic
        tle.Title = "Reconstructed Top Efficiency in $p_T$ Kinematic Regions \n using Inclusion Window $\\Delta M \\times \\Delta S$"
        tle.xTitle = "$p_T$ of Reconstructed Top (GeV)"
        tle.yTitle = "Efficiency of Top Reconstruction ($t_{\\text{Perfect}} / t_{\\text{Truth}}$)"
        tle.xMax = 1600
        tle.yMax = 1.01
        tle.xMin = 0
        tle.yMin = 0
        tle.xStep = 100
        tle.OutputDirectory = figure_path + "/Efficiency"
        tle.Filename = prc + "-efficiency"
        tle.SaveFigure()

        mx, key = 0, ""
        for k in integral: mx, key = (integral[k], k) if mx < integral[k] else (mx, key)

        index = [i[0] for i in sorted(enumerate(purity_efficiency[key]["efficiency"]), key=lambda x:x[1])]
        purity_mx     = [purity_efficiency[key]["purity"][k] for k in index]
        efficiency_mx = [purity_efficiency[key]["efficiency"][k] for k in index]

        tlk = TLine()
        tlk.yData = purity_mx
        tlk.xData = efficiency_mx
        tlk.Title = "Reconstructed Top Purity as a function of Efficiency ($\\Delta$ Mass / $\\Delta$ Score - " + key.replace("-", "/") + ")"
        tlk.xTitle = "Efficiency $\\epsilon$ - $(N_{\\text{perfect}} / N_{\\text{truth}})$"
        tlk.yTitle = "Purity $\\rho_{\\text{top}}$ - $(N_{\\text{perfect}} / N_{\\text{truth}})$"
        tlk.xMax = 1.01
        tlk.yMax = 1.01
        tlk.xMin = 0
        tlk.yMin = 0
        tlk.xStep = 0.10
        tlk.OutputDirectory = figure_path + "/Efficiency"
        tlk.Filename = prc + "-efficiency_purity"
        tlk.SaveFigure()

        print(mx, key)
        break

def NtopClassification(stacks, ana = None):
    if ana is not None:
        eff = ana.ntops_efficiency
        for fname in eff:
            print(eff)
            exit()
            for fname in truth_tops[kin]:
                prc = mapping(fname)
                if prc not in stacks: stacks[prc] = {"ntruth" : {}, "reco" : {}, "perfect" : {}}
                if kin not in stacks[prc]["ntruth"]: stacks[prc]["ntruth"][kin] = 0
                stacks[prc]["ntruth"][kin] += sum(truth_tops[kin][fname])


def TopEfficiency(ana):
    p = Path(ana)
    files = [str(x) for x in p.glob("**/*.pkl") if str(x).endswith(".pkl")]
    files = list(set(files))
    stacks_s = {"score-project" : {}}
    stacks_c = {}
    stacks_x = {}
    stacks_k = {}
    stacks_n = {}
    #files = files[:1]
    for i in range(len(files)):
        print(files[i], (i+1) / len(files))
        pr = pickle.load(open(files[i], "rb"))

        stacks_s = TopScores(stacks_s, pr)
        stacks_k = KinematicTops(stacks_k, pr)
        stacks_x = EfficiencyEvent(stacks_x, pr)
        stacks_c = TopMassComparison(stacks_c, pr)
#        stacks_n = NtopClassification(stacks_n, pr)

    TopScores(stacks_s)
    KinematicTops(stacks_k)
    EfficiencyEvent(stacks_x)
    TopMassComparison(stacks_c)
#    NtopClassification(stacks_n)

