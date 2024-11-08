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

def path(hist, subx = "", defaults = None):
    hist.UseLateX = False
    hist.Style = "ATLAS"
    hist.OutputDirectory = figure_path + "/topefficiency" + subx
    if defaults is None: return hist
    hist.xTitle = "Invariant Mass of Candidate Top (GeV)"
    hist.yTitle = "Tops / ($2$ GeV)"
    hist.xMin   = 0
    hist.yMin   = 0
    hist.xStep  = 20
    hist.xMax   = 400
    hist.xBins  = 200
    hist.Stacked   = True
    hist.Overflow  = False
    hist.ShowCount = True
    return hist

def top_kinematic_region(stacks, tmp = None):
    if tmp is not None: return top_pteta(stacks, tmp, metalookup)

    cols = {}
    top_score_mass = {}
    pt_topmass_prc = {"truth" : {}, "prediction" : {}}
    pt_topmass_cmp = {"truth" : {}, "prediction" : {}}

    regions = list(set(list(stacks["truth"]) + list(stacks["prediction"])))
    for kin in sorted(regions):
        pt_r = kin.split(",")[0]

        if kin not in stacks["prediction"]: stacks["prediction"][kin] = {}
        if kin in stacks["truth"]: data_tru = sum(stacks["truth"][kin].values())
        else: data_tru = metalookup.GenerateData

        if pt_r not in pt_topmass_cmp["prediction"]:
            pt_topmass_cmp["prediction"][pt_r] = {}
            pt_topmass_cmp["truth"][pt_r] = {}

        if kin in stacks["truth"]:
            for fn, val in stacks["truth"][kin].items():
                _, tl, col = mapping(metalookup(fn).DatasetName)
                cols[tl] = col
                if tl not in pt_topmass_cmp["truth"][pt_r]: pt_topmass_cmp["truth"][pt_r][tl] = []
                pt_topmass_cmp["truth"][pt_r][tl].append(val)

        tru_h = TH1F()
        tru_h.Title   = "Truth"
        tru_h.xData   = data_tru.data
        tru_h.Weights = data_tru.weights
        tru_h.Color   = "black"

        if pt_r not in pt_topmass_prc["truth"]: pt_topmass_prc["truth"][pt_r] = []
        if pt_r not in pt_topmass_prc["prediction"]: pt_topmass_prc["prediction"][pt_r] = {}
        if pt_r not in top_score_mass: top_score_mass[pt_r] = {"mass" : {}, "score" : {}}
        pt_topmass_prc["truth"][pt_r] += list(stacks["truth"][kin].values()) if kin in stacks["truth"] else []

        hists = {}
        for fn in stacks["prediction"][kin]:
            _, tl, col = mapping(metalookup(fn).DatasetName)
            if tl not in pt_topmass_prc["prediction"][pt_r]:
                pt_topmass_prc["prediction"][pt_r][tl] = []
                pt_topmass_cmp["prediction"][pt_r][tl] = []
                top_score_mass[pt_r]["mass"][tl]  = []
                top_score_mass[pt_r]["score"][tl] = []

            pt_topmass_prc["prediction"][pt_r][tl].append(stacks["prediction"][kin][fn])
            pt_topmass_cmp["prediction"][pt_r][tl].append(stacks["prediction"][kin][fn])
            top_score_mass[pt_r]["mass"][tl].append(stacks["prediction"][kin][fn])
            top_score_mass[pt_r]["score"][tl].append(stacks["top_score"][kin][fn])

            if tl not in hists: hists[tl] = []
            hists[tl].append(stacks["prediction"][kin][fn])
            cols[tl] = col

        for prc in hists:
            data_p = sum(hists[prc])
            prc_h = TH1F()
            prc_h.Title   = prc
            prc_h.Color   = cols[prc]
            prc_h.xData   = data_p.data
            prc_h.Weights = data_p.weights
            hists[prc] = prc_h

        tlt = kin.replace("_", " \\leq p^{top}_T (GeV) \\leq ")
        tlt = tlt.replace("-", " \\leq | \\eta_{top} | \\leq ")

        reco = path(TH1F(), "/" + kin.split(",")[0], True)
        reco.Title = "Reconstructed Invariant Mass of Top Candidate within Kinematic Region: $" + tlt + "$"
        reco.Histograms = list(hists.values())
        reco.Histogram = tru_h
        reco.Filename = kin.split(", ")[1]
        reco.SaveFigure()

    for kin in set(list(pt_topmass_prc["truth"]) + list(pt_topmass_prc["prediction"])):
        data_tru = sum(pt_topmass_prc["truth"][kin]) if len(pt_topmass_prc["truth"][kin]) else metalookup.GenerateData

        tru = TH1F()
        tru.Title = "Truth"
        tru.xData   = data_tru.data
        tru.Weights = data_tru.weights

        nums = []
        hists = []
        for prc, val in pt_topmass_prc["prediction"][kin].items():
            data_p = sum(val) if len(val) else metalookup.GenerateData

            prc_h = TH1F()
            prc_h.Title   = prc
            prc_h.Color   = cols[prc]
            prc_h.xData   = data_p.data
            prc_h.Weights = data_p.weights
            hists.append(prc_h)
            nums.append(len(prc_h.xData))

        tlt = kin.replace("_", " \\leq p^{top}_T (GeV) \\leq ")
        reco = path(TH1F(), "/aggregated-pt/", True)
        reco.Title = "Reconstructed Invariant Mass of Top Candidate with Transverse Momentum: $" + tlt + "$"
        reco.Histogram = tru
        reco.Histograms = [hists[k] for k in sorted(range(len(nums)), key = lambda k : nums[k])]
        reco.Filename = kin.split(", ")[0]
        reco.SaveFigure()

        for p in set(list(pt_topmass_cmp["truth"][kin]) + list(pt_topmass_cmp["prediction"][kin])):
            try: data_t    = sum(pt_topmass_cmp["truth"][kin][p])
            except: data_t = metalookup.GenerateData

            prc_t         = TH1F()
            prc_t.Title   = "Truth"
            prc_t.xData   = data_t.data
            prc_t.Weights = data_t.weights

            try: data_p    = sum(pt_topmass_cmp["prediction"][kin][p])
            except: data_p = metalookup.GenerateData
            prc_d         = TH1F()
            prc_d.Title   = "Reconstructed"
            prc_d.Color   = cols[p]
            prc_d.xData   = data_p.data
            prc_d.Weights = data_p.weights

            reco = path(TH1F(), "/aggregated-pt/" + kin.split(", ")[0] + "/", True)
            reco.Title = "Reconstructed Invariant Mass of Top Candidate with Transverse Momentum: $" + tlt + "$ (" + p + ")"
            reco.Histogram  = prc_t
            reco.Histograms = [prc_d]
            reco.Filename   = p.replace("\\", "").replace("$", "").replace("{", "").replace("}","")
            reco.SaveFigure()

    for kin in top_score_mass:
        hists = {}
        for prc, val in top_score_mass[kin]["score"].items():
            data_p = sum(val)

            prc_h         = TH1F()
            prc_h.Title   = prc
            prc_h.Color   = cols[prc]
            prc_h.xData   = data_p.data
            prc_h.Weights = data_p.weights
            hists[prc]    = prc_h
        if not len(hists): continue

        tlt = kin.replace("_", " \\leq p^{top}_T (GeV) \\leq ")
        s_s = path(TH1F(), "/pt-score")
        s_s.Histograms = list(hists.values())
        s_s.Title = "Reconstructed Top Candidate Score with \n Transverse Momentum $" + tlt + "$"
        s_s.xTitle = "MVA PageRanked Candidate Top Score (Arb.)"
        s_s.yTitle = "Tops / ($0.002$)"
        s_s.yMin = 1e-1
        s_s.xMin = 0
        s_s.xMax = 1
        s_s.xBins = 500
        s_s.xStep = 0.05
        s_s.yLogarithmic = True
        s_s.Stacked   = True
        s_s.ShowCount = True
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
        mass_s.yMax = 1.00
        mass_s.yStep = 0.05
        mass_s.yBins = 100
        mass_s.Color = "magma"

        data_m = sum(sum(top_score_mass[kin]["mass"].values(), [])).data
        data_s = sum(sum(top_score_mass[kin]["score"].values(), []))

        mass_s.xData   = data_m
        mass_s.yData   = data_s.data
        mass_s.Filename = "pt_range_" + kin
        mass_s.SaveFigure()

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

    content = {"truth" : {}, "prediction" : {}, "perfect" : {}}
    for fn in fnames:
        w = stacks["weights"][fn]
        p_nt_pred = stacks["pred_ntops"][fn]
        p_nt_perf = stacks["perf_ntops"][fn]
        t_nt      = stacks["tru_ntops"][fn]
        scores    = list(p_nt_pred)

        tru_ = add_data(content["truth"], fn)
        tru_.weights = {fn : w}
        tru_.data    = {fn : t_nt}

        for i in range(len(scores)):
            sc = scores[i]
            pred_nt = p_nt_pred[sc]
            perf_nt = p_nt_perf[sc]

            if fn not in content["prediction"]:
                content["perfect"][fn] = {}
                content["prediction"][fn] = {}

            perf_ = add_data(content["perfect"][fn], sc)
            perf_.weights = {fn : w}
            perf_.data    = {fn : perf_nt}

            pred_ = add_data(content["prediction"][fn], sc)
            pred_.weights = {fn : w}
            pred_.data    = {fn : pred_nt}

    merged = {"truth" : {}, "prediction" : {}, "perfect" : {}}
    for fn in fnames:
        _, prc, _ = content["truth"][fn].title(fn)
        if prc not in merged["truth"]: merged["truth"][prc] = []
        merged["truth"][prc].append(content["truth"][fn])
        content["truth"][fn] = None

        scores = list(content["prediction"][fn])
        if prc not in merged["prediction"]: merged["prediction"][prc] = {sc : [] for sc in scores}
        if prc not in merged["perfect"]:    merged["perfect"][prc]    = {sc : [] for sc in scores}

        for sc in scores:
            merged["prediction"][prc][sc].append(content["prediction"][fn][sc])
            merged["perfect"][prc][sc].append(content["perfect"][fn][sc])

        content["prediction"][fn] = None
        content["perfect"][fn] = None

    for prc in merged["truth"]:      merged["truth"][prc]      = sum(merged["truth"][prc])
    for prc in merged["perfect"]:    merged["perfect"][prc]    = {sc: sum(merged["perfect"][prc][sc]) for sc in merged["perfect"][prc]}
    for prc in merged["prediction"]: merged["prediction"][prc] = {sc: sum(merged["prediction"][prc][sc]) for sc in merged["prediction"][prc]}


    tl = path(TLine())
    tl.Title = "Efficiency Purity Top Reconstruction for \n n-Top Multiplicity Topologies"
    tl.xTitle = "Efficiency ($N_{perfect} / N_{truth}$)"
    tl.yTitle = "Purity ($N_{perfect} / N_{candidate}$)"
    tl.xMin = 0
    tl.yMin = 0
    tl.xMax = 1
    tl.yMax = 1
    tl.Filename = "purity_efficiency"

    pairs = {}
    for prc in merged["truth"]:
        tru_data = merged["truth"][prc].data
        weights  = merged["truth"][prc].weights
        num_tru  = sum([tru_data[i]*weights[i] for i in range(len(weights))])

        for sc in list(merged["perfect"][prc]):
            prf = merged["perfect"][prc][sc].data
            num_prf = sum([prf[i]*weights[i] for i in range(len(weights))])

            prd = merged["prediction"][prc][sc].data
            num_can = sum([prd[i]*weights[i] for i in range(len(weights))])

            if num_can == 0: continue
            purity = num_prf / num_can
            effi   = num_prf / num_tru
            pairs[effi] = purity

    tl.xData = sorted(pairs)
    tl.yData = [pairs[k] for k in sorted(pairs)]
    #tl.Lines.append(ln)
    tl.SaveFigure()

def TopEfficiency(ana):
    p = Path(ana)
    files = [str(x) for x in p.glob("**/*.pkl") if str(x).endswith(".pkl")]
    files = list(set(files))
    files = sorted(files)
#    files = files[:1]

    stack_roc = {}
    stack_topkin = {}
    stack_ntops = {}
    for i in range(len(files)):
        print(files[i], (i+1) / len(files))
        pr = pickle.load(open(files[i], "rb"))
        stack_topkin = top_kinematic_region(stack_topkin, pr)
        #stack_roc    = roc_data(stack_roc, pr)
        #stack_ntops  = ntops_reco(stack_ntops, pr)

    #f = open("tmp.pkl", "wb")
    #pickle.dump(stack_topkin, f)
    #f.close()

    #f = open("tmp.pkl", "rb")
    #stack_topkin = pickle.load(f)
    #f.close()

    top_kinematic_region(stack_topkin)
    #roc_data(stack_roc)
    #ntops_reco(stack_ntops)

