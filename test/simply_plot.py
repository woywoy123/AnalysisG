import AnalysisG
from AnalysisG.core import IO
from AnalysisG.core.plotting import *
from AnalysisG.generators import Analysis

class container:
    def __init__(self):
        self.proc = ""
        self._num_events = []
        self._sow_nominal = []
        self._sow_weights = []

        self._plot_data = []

        self._cross_section = 0
        self._nb_to_fb = 10**6
        self._lint_atlas = 140.1

    @property
    def cross_section(self): return self._cross_section*self._nb_to_fb

    @property
    def expected_events(self): return self.cross_section*self._lint_atlas

    @property
    def scale_factor(self): return self.expected_events / self.sow_weights

    @property
    def num_events(self): return sum(self._num_events)

    @property
    def sow_nominal(self): return sum(self._sow_nominal)

    @property
    def sow_weights(self): return sum(self._sow_weights)

    @property
    def hist(self):
        th = TH1F()
        sf = self.scale_factor
        th.Weights = [d*sf for d in self._sow_nominal]
        th.xData = self._plot_data
        th.Title = self.proc.split("_")[-1]
        return th

def get_met(pth, proc):
    smpl = IO(pth + proc + "/*")
    smpl.Trees = ["nominal"]
    smpl.Leaves = ["weight_mc", "met_met"]
    out = {"weight_mc" : [], "met" : []}
    for i in smpl:
        out["met"] += [i[b"nominal.met_met.met_met"]/1000]
        out["weight_mc"] += [i[b"nominal.weight_mc.weight_mc"]]
    return out

def compute_data(mtx, proc, root):
    c = container()
    c.proc = proc
    for i in mtx:
        if proc not in i: continue
        c._cross_section = mtx[i].crossSection
        c._num_events   += [mtx[i].SumOfWeights[b"sumWeights"]["total_events"]]
        c._sow_weights  += [mtx[i].SumOfWeights[b"sumWeights"]["processed_events_weighted"]]

    data = get_met(root, proc)
    c._sow_nominal = data["weight_mc"]
    c._plot_data   = data["met"]
    return c

def buffs(tl, val, tm):
    if isinstance(val, float): val = round(val, 5)
    l1, v1 = len(tl), str(val)
    return tm + "".join([" "]*(l1 - len(v1))) + v1 + " | "


root = "/home/tnom6927/Downloads/mc16/"
x = Analysis()
x.AddSamples(root + "MadGraphPythia8EvtGen_noallhad_ttH_tttt_m1000/*", "m1000")
x.AddSamples(root + "MadGraphPythia8EvtGen_noallhad_ttH_tttt_m900/*" , "m900")
x.AddSamples(root + "MadGraphPythia8EvtGen_noallhad_ttH_tttt_m800/*" , "m800")
x.AddSamples(root + "MadGraphPythia8EvtGen_noallhad_ttH_tttt_m700/*" , "m700")
x.AddSamples(root + "MadGraphPythia8EvtGen_noallhad_ttH_tttt_m600/*" , "m600")
x.AddSamples(root + "MadGraphPythia8EvtGen_noallhad_ttH_tttt_m500/*" , "m500")
x.AddSamples(root + "MadGraphPythia8EvtGen_noallhad_ttH_tttt_m400/*" , "m400")
x.AddSamples(root + "SM4topsNLO/*" , "tttt")
x.AddSamples(root + "ttH125/*" , "ttH")
x.SumOfWeightsTreeName = "sumWeights"
x.FetchMeta = True
x.Start()
mtx = x.GetMetaData

opt = {}
opt["m1000"] = compute_data(mtx, "MadGraphPythia8EvtGen_noallhad_ttH_tttt_m1000", root)
opt["m900" ] = compute_data(mtx, "MadGraphPythia8EvtGen_noallhad_ttH_tttt_m900" , root)
opt["m800" ] = compute_data(mtx, "MadGraphPythia8EvtGen_noallhad_ttH_tttt_m800" , root)
opt["m700" ] = compute_data(mtx, "MadGraphPythia8EvtGen_noallhad_ttH_tttt_m700" , root)
opt["m600" ] = compute_data(mtx, "MadGraphPythia8EvtGen_noallhad_ttH_tttt_m600" , root)
opt["m500" ] = compute_data(mtx, "MadGraphPythia8EvtGen_noallhad_ttH_tttt_m500" , root)
opt["m400" ] = compute_data(mtx, "MadGraphPythia8EvtGen_noallhad_ttH_tttt_m400" , root)
opt["tttt" ] = compute_data(mtx, "SM4topsNLO", root)
opt["ttH"  ] = compute_data(mtx, "ttH125", root)

titles = [
    "Sample Processed",
    "Exp. Events (uncut)",
    "x-section (fb)",
    "Gen. Events (unweighted)",
    "Sum of Weights (weight_mc)",
    "Sum of Weights (Tree)",
    "Scale factor (Exp. / sow Tree)",
    ""
]


h = []
print(" | ".join(titles))
for i in list(opt.values()):
    prx = i.proc.split("_")[-1]
    tmp = ""
    tmp = buffs(titles[0], prx, tmp)
    tmp = buffs(titles[1], i.expected_events, tmp)
    tmp = buffs(titles[2], i.cross_section, tmp)
    tmp = buffs(titles[3], i.num_events, tmp)
    tmp = buffs(titles[4], i.sow_nominal, tmp)
    tmp = buffs(titles[5], i.sow_weights, tmp)
    tmp = buffs(titles[6], i.scale_factor, tmp)
    h.append(i.hist)
    print(tmp)


comb = TH1F()
comb.Histograms = h
comb.Title = "Weighted Missing Transverse Energy"
comb.xTitle = "Missing ET (GeV)"
comb.yTitle = "Entries / (5 GeV)"
comb.xBins = 100
comb.xMin = 0
comb.xMax = 500
comb.yMin = 0
comb.xStep = 40
comb.Filename = "weighted_hists"
comb.Stacked = True
comb.SaveFigure()


