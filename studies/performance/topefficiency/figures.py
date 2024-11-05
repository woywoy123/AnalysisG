from AnalysisG.core.plotting import TH1F, TH2F, TLine, ROC
from AnalysisG.core import *
from pathlib import Path
from .algorithms import *
import pickle

global figure_path
global metalookup

def MakeData(inpt,  key):
    try: return inpt[key]
    except KeyError: pass
    dt = metalookup.GenerateData
    inpt[key] = dt
    return dt

def path(hist, subx = ""):
    hist.Style = "ATLAS"
    hist.OutputDirectory = figure_path + "/topefficiency" + subx
    return hist

def top_kinematic_region(stacks, tmp = None):
    if tmp is not None: return top_pteta(stacks, tmp)

    cols = {}
    prc_topscore = {}
    top_score_mass = {}
    ks_topscore_eta_pt = {}
    pt_topmass_prc = {"truth" : {}, "prediction" : {}}

    k = 0
    regions = list(set(list(stacks["truth"]) + list(stacks["prediction"])))
    for kin in sorted(regions):
        pt_r = kin.split(",")[0]
        tru_h = None

        dt = MakeData(pt_topmass_prc["truth"], pt_r)
        if kin in stacks["truth"]:
            fnames = list(stacks["truth"][kin])
            th = metalookup.GenerateData
            w = {f : stacks["truth"][kin][f]["weights"] for f in fnames}
            d = {f : stacks["truth"][kin][f]["value"]   for f in fnames}

            th.weights = w
            th.data    = d

            tru_h = TH1F()
            tru_h.Title   = "Truth"
            tru_h.xData   = th.data
            tru_h.Weights = th.weights
            tru_h.Hatch   = "\\\\////"
            tru_h.Color   = "black"

            dt.weights = w
            dt.data    = d


        if kin not in stacks["prediction"]: stacks["prediction"][kin] = {}
        if pt_r not in pt_topmass_prc["prediction"]: pt_topmass_prc["prediction"][pt_r] = {}
        if pt_r not in top_score_mass: top_score_mass[pt_r] = {"mass" : metalookup.GenerateData, "score" : metalookup.GenerateData}
        if pt_r not in prc_topscore: prc_topscore[pt_r] = {}

        hists = {}
        for prc in stacks["prediction"][kin]:
            _, tl, col = metalookup(prc).title
            dh = MakeData(hists, tl)

            cols[tl] = col
            mass  = {prc : stacks["prediction"][kin][prc]["value"]}
            score = {prc : stacks["top_score"][kin][prc]["value"]}
            mwei  = {prc : stacks["prediction"][kin][prc]["weights"]}
            swei  = {prc : stacks["top_score"][kin][prc]["weights"]}

            dh.weights = mwei
            dh.data    = mass

            dptr = MakeData(pt_topmass_prc["prediction"][pt_r], tl)
            dptr.weights = mwei
            dptr.data    = mass

            top_score_mass[pt_r]["mass"].weights = mwei
            top_score_mass[pt_r]["mass"].data = mass

            top_score_mass[pt_r]["score"].weights = swei
            top_score_mass[pt_r]["score"].data = score

            dsc = MakeData(prc_topscore[pt_r], tl)
            dsc.weights = swei
            dsc.data    = score

            stacks["top_score"][kin][prc] = []
            stacks["prediction"][kin][prc] = []

        tmp = []
        nums = []
        for prc in hists:
            prc_h = TH1F()
            prc_h.Title   = prc
            prc_h.Color   = cols[prc]
            prc_h.xData   = hists[prc].data
            prc_h.Weights = hists[prc].weights
            tmp.append(prc_h)
            nums.append(len(prc_h.xData))
        hists = tmp

        tlt = kin.replace("_", " \\leq p^{top}_T (GeV) \\leq ")
        tlt = tlt.replace("-", " \\leq | \\eta_{top} | \\leq ")

        reco = path(TH1F(), "/" + kin.split(",")[0])
        reco.Title = "Reconstructed Invariant Mass of Top Candidate within \n Kinematic Region: $" + tlt + "$"
        reco.Histograms = [hists[k] for k in sorted(range(len(nums)), key = lambda k : nums[k])]
        reco.Histogram = tru_h
        reco.Stacked = True
        reco.xStep = 20
        reco.Overflow = False
        reco.xTitle = "Invariant Mass of Candidate Top (GeV)"
        reco.yTitle = "Tops / ($1$ GeV)"
        reco.yMin = 0
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
        tru.xData   = pt_topmass_prc["truth"][kin].data
        tru.Weights = pt_topmass_prc["truth"][kin].weights
        tru.Hatch = "\\\\////"
        tru.Color = "black"

        nums = []
        hists = []
        for prc in pt_topmass_prc["prediction"][kin]:
            prc_h = TH1F()
            prc_h.Color   = cols[prc]
            prc_h.Title   = prc
            prc_h.xData   = pt_topmass_prc["prediction"][kin][prc].data
            prc_h.Weights = pt_topmass_prc["prediction"][kin][prc].weights
            hists.append(prc_h)
            nums.append(len(prc_h.xData))

        tlt = kin.replace("_", " \\leq p^{top}_T (GeV) \\leq ")
        reco = path(TH1F(), "/aggregated-pt/")
        reco.Title = "Reconstructed Invariant Mass of Top Candidate with \n Transverse Momentum: $" + tlt + "$"
        reco.Histograms = [hists[k] for k in sorted(range(len(nums)), key = lambda k : nums[k])]
        reco.Histogram = tru
        reco.xStep = 20
        reco.Stacked = True
        reco.Overflow = False
        reco.xTitle = "Invariant Mass of Candidate Top (GeV)"
        reco.yTitle = "Tops / ($1$ GeV)"
        reco.xMin = 0
        reco.yMin = 0
        reco.xMax = 400
        reco.xBins = 400
        reco.Filename = kin.split(", ")[0]
        reco.SaveFigure()

    for kin in prc_topscore:
        hists = {}
        for prc in prc_topscore[kin]:
            prc_h = TH1F()
            prc_h.Color   = cols[prc]
            prc_h.Title   = prc
            prc_h.xData   = prc_topscore[kin][prc].data
            #prc_h.Weights = prc_topscore[kin][prc].weights
            hists[prc]    = prc_h
        if not len(hists): continue

        tlt = kin.replace("_", " \\leq p^{top}_T (GeV) \\leq ")
        s_s = path(TH1F(), "/pt-score")
        s_s.Histograms = list(hists.values()) #[hists[x] for x in sorted(list(hists))]
        s_s.Title = "Reconstructed Top Candidate Score with \n Transverse Momentum $" + tlt + "$"
        s_s.xTitle = "MVA Score of Candidate Top (Arb.)"
        s_s.yTitle = "Tops / ($0.001$)"

        s_s.yMin = 0.0001
        s_s.xMin = 0
        s_s.xMax = 1.001
        s_s.xBins = 1001
        s_s.xStep = 0.05
        s_s.Density = True
        s_s.Stacked = True

        s_s.Filename = "mva-score_" + kin
        s_s.SaveFigure()

    for kin in top_score_mass:
        tlt = kin.replace("_", " \\leq p^{top}_T (GeV) \\leq ")

        mass_s = path(TH2F(), "/score-mass")
        mass_s.Title = "Reconstructed Top Candidate Score as a \n function of Invariant Mass for $" + tlt + "$"
        mass_s.xTitle = "Reconstructed Top Candidate Invariant Mass / ($1$ GeV)"
        mass_s.yTitle = "MVA Score of Candidate Top / ($0.01$)"

        mass_s.xMin  = 0
        mass_s.xMax  = 400
        mass_s.xBins = 400
        mass_s.xStep = 20

        mass_s.yMin = 0
        mass_s.yMax = 1.01
        mass_s.yStep = 0.05
        mass_s.yBins = 101

        mass_s.xData   = top_score_mass[kin]["mass"].data
        mass_s.yData   = top_score_mass[kin]["score"].data
        #mass_s.Weights = top_score_mass[kin]["score"].weights
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

def roc_data(stacks, data = None):
    if data is not None: return roc_data_get(stacks, data)

    rx = path(ROC())
    rx.Title = "ROC Classification for Multi-Tops"
    rx.Titles = ["0-Tops", "1-Tops", "2-Tops", "3-Tops", "4-Tops"]
    rx.xData = stacks["n-tops_p"]
    rx.Truth = stacks["n-tops_t"]
    rx.Filename = "ROC-multitop"
    rx.SaveFigure()

    rx = path(ROC())
    rx.Title = "ROC Classification Top Edge"
    rx.xData = stacks["edge_top_p"]
    rx.Truth = stacks["edge_top_t"]
    rx.Filename = "ROC-top-edge"
    rx.Binary = True
    rx.SaveFigure()

    rx = path(ROC())
    rx.Title = "ROC Resonance MVA"
    rx.xData = stacks["signal_p"]
    rx.Truth = stacks["signal_t"]
    rx.Filename = "ROC-signal"
    rx.Binary = True
    rx.SaveFigure()

def ntops_reco(stacks, data = None):
    def add_data(data_, key):
        try: return data_[key]
        except: pass
        data = metalookup.GenerateData
        data_[key] = data
        return data


    if data is not None: return ntops_reco_compl(stacks, data)

    fnames = list(stacks["weights"])
    scores = sorted(set(list(stacks["pred_ntops"][fnames[0]])))
    ntops = sorted(set(sum([stacks["tru_ntops"][fn] for fn in fnames], [])))
    cls_ntops = {}

    for fn in fnames:
        for i in range(len(stacks["tru_ntops"][fn])):
            n = stacks["tru_ntops"][fn][i]
            w = stacks["weights"][fn][i]
            w = {fn : [w]}
            ntk = str(n) + "-ntops"

            data = add_data(cls_ntops, ntk + ".tru")
            data.weights = w

            for sc in scores:
                prd = stacks["pred_ntops"][fn][sc][i]
                prf = stacks["perf_ntops"][fn][sc][i]

                # num correct
                data = add_data(cls_ntops, ntk + ".prd@"  + str(sc))
                if n == prd: data.weights = w

                data = add_data(cls_ntops, ntk + ".prf@"  + str(sc))
                if n == prf: data.weights = w

                for nt in ntops:
                    # candidates
                    data = add_data(cls_ntops, ntk + ".nprd@" + str(sc))
                    if nt == prd: data.weights = w

                    data = add_data(cls_ntops, ntk + ".nprf@" + str(sc))
                    if nt == prf: data.weights = w



    truths = {k : d for k, d in cls_ntops.items() if ".tru" in k}
    for key in truths:
        ntops = key.split(".tru")[0]
        data  = truths[key]
        st = sum(data.weights)

        eff_predt, pur_predt = {}, {}
        eff_prfct, pur_prfct = {}, {}
        for k, d in cls_ntops.items():
            if ntops not in k: continue
            if ".prd" not in k and ".prf" not in k: continue
            sc = float(k.split("@")[-1])
            sm = sum(d.weights)

            ki = k.replace("prd", "nprd") if ".prd" in k else k.replace("prf", "nprf")
            cm = sum(cls_ntops[ki].weights)
            if cm == 0: continue

            if ".prd" in k:
                eff_predt |= {sc : sm / st}
                pur_predt |= {sc : sm / cm}
                continue

            if ".prf" in k:
                eff_prfct |= {sc : sm / st}
                pur_prfct |= {sc : sm / cm}
                continue


        tlpf = TLine()
        tlpf.Title = "Perfect Tops"
        tlpf.xData = list(eff_prfct) #list(eff_prfct.values())
        tlpf.yData = list(eff_prfct.values())

        tlpd = TLine()
        tlpd.Title = "Predicted Tops"
        tlpd.xData = list(eff_predt) #.values())
        tlpd.yData = list(eff_predt.values())

        tl = path(TLine())
        tl.Lines = [tlpf, tlpd]
        tl.Title = "Multi-Top Reconstruction Efficiency: " + ntops
        tl.xTitle = "Reconstructed Top Score"
        tl.yTitle = "Efficiency"
        tl.xMax = 1.2
        tl.xMin = 0
        tl.yMax = 1.2
        tl.yMin = 0
        tl.Filename = "efficiency_" + ntops
        tl.SaveFigure()


        tlpf = TLine()
        tlpf.Title = "Perfect Tops"
        tlpf.xData = list(pur_prfct)
        tlpf.yData = list(pur_prfct.values())

        tlpd = TLine()
        tlpd.Title = "Predicted Tops"
        tlpd.xData = list(pur_predt)
        tlpd.yData = list(pur_predt.values())

        tl = path(TLine())
        tl.Lines = [tlpf, tlpd]
        tl.Title = "Multi-Top Reconstruction Purity: " + ntops
        tl.xTitle = "Reconstructed Top Score"
        tl.yTitle = "Purity"
        tl.xMax = 1.2
        tl.xMin = 0
        tl.yMax = 1.2
        tl.yMin = 0
        tl.Filename = "purity_" + ntops
        tl.SaveFigure()

def TopEfficiency(ana):
    p = Path(ana)
    files = [str(x) for x in p.glob("**/*.pkl") if str(x).endswith(".pkl")]
    files = list(set(files))
    files = sorted(files)
    #files = files[:1]

    stack_roc = {}
    stack_topkin = {}
    stack_ntops = {}
    for i in range(len(files)):
        print(files[i], (i+1) / len(files))
        pr = pickle.load(open(files[i], "rb"))
        stack_topkin = top_kinematic_region(stack_topkin, pr)
        stack_roc    = roc_data(stack_roc, pr)
        stack_ntops  = ntops_reco(stack_ntops, pr)

    #f = open("tmp.pkl", "wb")
    #pickle.dump(stack_topkin, f)
    #f.close()

    #f = open("tmp.pkl", "rb")
    #stack_topkin = pickle.load(f)
    #f.close()

    top_kinematic_region(stack_topkin)
    roc_data(stack_roc)
    ntops_reco(stack_ntops)

