from AnalysisG.core.plotting import TH1F, TH2F, TLine, ROC
from AnalysisG.core import Meta
from pathlib import Path
from .algorithms import *
import pickle

global figure_path
global metacache
global metalookup

class MetaLookup:
    def __init__(self):
        self.metadata = {}
        self.matched = {}
        self.meta = None
        self._lint_atlas = 140.1

    def __find__(self, inpt):
        try: inpt = inpt.decode("utf-8")
        except: pass

        try: return self.matched[inpt]
        except: pass

        for i in self.metadata:
            if self.metadata[i].hash(inpt) not in i: continue
            self.matched[inpt] = self.metadata[i]
            return self.metadata[i]

    def __call__(self, inpt):
        self.meta = self.__find__(inpt)
        return self

    def title(self, inpt): return mapping(self.__find__(inpt).DatasetName)

    @property
    def DatasetName(self): return self.meta.DatasetName
    @property
    def SumOfWeights(self):
        return self.meta.SumOfWeights[b"sumWeights"]["total_events_weighted"]

    @property
    def CrossSection(self): return self.meta.crossSection
    @property
    def expected_events(self): return self.CrossSection*self._lint_atlas

def MakeData(inpt,  key):
    try: return inpt[key]
    except KeyError: pass
    dt = data(metalookup)
    inpt[key] = dt
    return dt

class data:

    def __init__(self, meta):
        self._weights = {}
        self._data    = {}
        self._sow     = {}
        self._exp     = {}
        self._meta    = meta

    def __populate__(self, inpt, trgt):
        for fname, val in inpt.items():
            key = self._meta(fname).DatasetName
            if key not in trgt: trgt[key] = []
            trgt[key] += val
            if key not in self._sow: self._sow[key] = {}
            if fname in self._sow[key]: continue
            self._sow[key][fname] = self._meta(fname).SumOfWeights
            self._exp[key] = self._meta(fname).expected_events

    def __rescale__(self):
        weights = []
        for i in self._sow:
            scale = self._exp[i] / sum(self._sow[i].values())
            weights += [l/scale for l in self._weights[i]]
        return weights

    @property
    def weights(self): return self.__rescale__()
    @weights.setter
    def weights(self, val): self.__populate__(val, self._weights)

    @property
    def data(self): return sum(list(self._data.values()), [])
    @data.setter
    def data(self, val): self.__populate__(val, self._data)

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
            th = data(metalookup)
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
        if pt_r not in top_score_mass: top_score_mass[pt_r] = {"mass" : data(metalookup), "score" : data(metalookup)}
        if pt_r not in prc_topscore: prc_topscore[pt_r] = {}

        hists = {}
        for prc in stacks["prediction"][kin]:
            _, tl, col = metalookup.title(prc)
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
            prc_h.Density = True
            prc_h.Color   = cols[prc]
            prc_h.Title   = prc
            prc_h.xData   = prc_topscore[kin][prc].data
            hists[prc]    = prc_h

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
        s_s.yLogarithmic = True

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
    if data is not None: return ntops_reco_compl(stacks, data, 2)
    w2 = sum(stacks["cls_ntop_w"][2])
    et = sum(stacks["e_ntop"][2]) / float(w2)
    pt = sum(stacks["e_ntop"][2]) / sum(stacks["p_ntop"][2])
    print(et, pt)

def TopEfficiency(ana):
    p = Path(ana)
    files = [str(x) for x in p.glob("**/*.pkl") if str(x).endswith(".pkl")]
    files = list(set(files))
    files = sorted(files)
    #files = files[:5]

    metl = MetaLookup()
    metl.metadata = metacache
    global metalookup
    metalookup = metl

    stack_roc = {}
    stack_topkin = {}
    stack_ntops = {}
    for i in range(len(files)):
        print(files[i], (i+1) / len(files))
        pr = pickle.load(open(files[i], "rb"))
        #stack_topkin = top_kinematic_region(stack_topkin, pr)
        stack_roc    = roc_data(stack_roc, pr)
        #stack_ntops  = ntops_reco(stack_ntops, pr)

    #f = open("tmp.pkl", "wb")
    #pickle.dump(stack_topkin, f)
    #f.close()

    #f = open("tmp.pkl", "rb")
    #stack_topkin = pickle.load(f)
    #f.close()

    #top_kinematic_region(stack_topkin)
    roc_data(stack_roc)
    #ntops_reco(stack_ntops)

