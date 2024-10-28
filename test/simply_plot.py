import AnalysisG
from AnalysisG.core import IO
from AnalysisG.core.plotting import *
from AnalysisG.generators import Analysis


def mapping(name):
    if "_singletop_"  in name: return "$t$"
    if "_tchan_"      in name: return "$t$"
    if "_Wt_"         in name: return "$Wt$"
    if "_ttbarHT1k_"  in name: return "$t\\bar{t}$"
    if "_ttbar_"      in name: return "$t\\bar{t}$"
    if "_ttbarHT1k5_" in name: return "$t\\bar{t}$"
    if "_ttbarHT6c_"  in name: return "$t\\bar{t}$"
    if "_ttH125_"     in name: return "$t\\bar{t}H$"
    if "_SM4topsNLO"  in name: return "$t\\bar{t}t\\bar{t}$"
    if "_tt_"         in name: return "$t\\bar{t}$"
    if "_ttee."       in name: return "$t\\bar{t}\\ell\\ell$"
    if "_ttmumu."     in name: return "$t\\bar{t}\\ell\\ell$"
    if "_tttautau."   in name: return "$t\\bar{t}\\ell\\ell$"
    if "_ttW."        in name: return "$t\\bar{t}V$"
    if "_ttZnunu."    in name: return "$t\\bar{t}V$"
    if "_ttZqq."      in name: return "$t\\bar{t}V$"
    if "_tW."         in name: return "$tV$"
    if "_tW_"         in name: return "$tV$"
    if "_tZ."         in name: return "$tV$"
    if "_WH125."      in name: return "$VH$"
    if "_WH125_"      in name: return "$VH$"
    if "_ZH125_"      in name: return "$VH$"
    if "_WplvWmqq"    in name: return "$V_{1}V_{2}$"
    if "_WpqqWmlv"    in name: return "$V_{1}V_{2}$"
    if "_WlvZqq"      in name: return "$V_{1}V_{2}$"
    if "_WqqZll"      in name: return "$V_{1}V_{2}$"
    if "_WqqZvv"      in name: return "$V_{1}V_{2}$"
    if "_ZqqZll"      in name: return "$V_{1}V_{2}$"
    if "_ZqqZvv"      in name: return "$V_{1}V_{2}$"
    if "_llll"        in name: return "$\\ell,\\nu$"
    if "_lllv"        in name: return "$\\ell,\\nu$"
    if "_llvv"        in name: return "$\\ell,\\nu$"
    if "_lvvv"        in name: return "$\\ell,\\nu$"
    if "_Wmunu_"      in name: return "$V\\ell\\nu$"
    if "_Wenu_"       in name: return "$V\\ell\\nu$"
    if "_Wtaunu_"     in name: return "$V\\ell\\nu$"
    if "_Zee_"        in name: return "$V\\ell\\ell$"
    if "_Ztautau_"    in name: return "$V\\ell\\ell$"
    if "_Zmumu_"      in name: return "$V\\ell\\ell$"
    if "ttH_tttt"     in name: return "$t\\bar{t}t\\bar{t}H_{" + name.split("tttt_m")[-1].split(".")[0] + "}$"
    print("----> " + name)
    exit()

class container:
    def __init__(self):
        self.proc = ""
        self._sow_nominal = []
        self._plot_data   = []
        self._weights     = 0
        self._passed      = 0
        self._proc_ev     = 0
        self._meta        = None
        self.lint_atlas   = 140.1

    def __add__(self, other):
        if other == 0:
            c               = container()
            c.proc          = self.proc
            c._meta         = self._meta
            c._proc_ev     += self.processed_events
            c._passed      += len(self._plot_data)
            c._plot_data   += self._plot_data

            s = self.filtereff*self.scale_factor
            c._sow_nominal += [i*s for i in self._sow_nominal]
            c._weights     += self.sow_weights
            return c

        self._proc_ev     += other.processed_events
        self._passed      += len(other._sow_nominal)
        self._plot_data   += other._plot_data
        self._weights     += other.sow_weights

        s = other.filtereff*other.scale_factor
        self._sow_nominal += [i*s for i in other._sow_nominal]
        return self

    def __radd__(self, other):
        if other == 0: return self
        self.__add__(other)
        return self

    @property
    def filtereff(self):
        s = self.processed_events
        if s == 0: s = 1
        return float(self._passed/s)

    @property
    def expected_events(self): return (self.cross_section*self.lint_atlas)

    @property
    def scale_factor(self): return self.expected_events / self.sow_weights

    @property
    def processed_events(self):
        if self._proc_ev > 0: return self._proc_ev
        return self._meta.SumOfWeights[b"sumWeights"]["processed_events"]

    @property
    def sow_nominal(self): return sum(self._sow_nominal)

    @property
    def cross_section(self): return self._meta.crossSection

    @property
    def sow_weights(self):
        if self._weights == 0: return self._meta.SumOfWeights[b"sumWeights"]["total_events_weighted"]
        return self._weights

    @property
    def num_events(self): return self._meta.totalEvents

    @property
    def DatasetName(self): return self._meta.DatasetName

    @property
    def hist(self):
        th = TH1F()
        th.Weights = self._sow_nominal
        th.xData = self._plot_data
        th.Title = self.proc
        return th

def compute_data(io_handle, meta):
    stacks = {}
    for i in io_handle:
        fname = i["filename"].decode("utf-8")
        try: c = stacks[fname]
        except:
            m = None
            c = container()
            for h, k in zip(meta.keys(), meta.values()):
                if k.hash(fname) not in h: continue
                m = k
                break
            if m is None: print("-> ", fname); exit()
            c.proc          = mapping(m.DatasetName)
            c._meta         = m
            stacks[fname]   = c
        c._plot_data.append(i[b"nominal.met_met.met_met"]/1000)
        c._sow_nominal.append(i[b"nominal.weight_mc.weight_mc"])
    return stacks

def buffs(tl, val, tm):
    if isinstance(val, float): val = round(val, 5)
    l1, v1 = len(tl), str(val)
    return tm + "".join([" "]*(l1 - len(v1))) + v1 + " | "


root = "/home/tnom6927/Downloads/mc16-full/*" #root = "/CERN/Samples/mc16-full/*"
x = Analysis()
x.FetchMeta = True
#x.SumOfWeightsTreeName = "sumWeights"
#x.AddSamples(root, "data")
#x.AddSamples(root, "lable")
#x.AddSamples(root + "ttH_tttt_m1000/*", "m1000")
#x.AddSamples(root + "ttH_tttt_m900/*" , "m900")
#x.AddSamples(root + "ttH_tttt_m800/*" , "m800")
#x.AddSamples(root + "ttH_tttt_m700/*" , "m700")
#x.AddSamples(root + "ttH_tttt_m600/*" , "m600")
#x.AddSamples(root + "ttH_tttt_m500/*" , "m500")
#x.AddSamples(root + "ttH_tttt_m400/*" , "m400")
#x.AddSamples(root + "SM4topsNLO/*" , "tttt")
#x.AddSamples(root + "ttH125/*" , "ttH")
#x.SumOfWeightsTreeName = "sumWeights"
#x.FetchMeta = True
x.Start()
mtx = x.GetMetaData
smpl = IO(root)
smpl.Trees = ["nominal"]
smpl.Leaves = ["weight_mc", "met_met"]

proc = {}
opt = compute_data(smpl, mtx)
for i in opt:
    c = opt[i]
    if "tttt_m" in c.DatasetName:
        if "tttt_m1000" in c.DatasetName: pass
        else: continue
    prc = mapping(c.DatasetName)
    if prc not in proc: proc[prc] = []
    proc[prc] += [c]

for i in proc: proc[i] = sum(proc[i])

#for i in mtx.values():
#    name = "Dataset Name: " + i.DatasetName
#    print(mapping(i.DatasetName))
#    print(i)
#    exit()
#    evw = i.SumOfWeights[b"sumWeights"]["processed_events_weighted"]
#    evt = i.SumOfWeights[b"sumWeights"]["total_events_weighted"]
#    name += " | processed events weighted (pew): " + str(evw)
#    name += " | total events weighted (tew): " + str(evt)
#    name += " | Difference (pew - tew): " + str(evw - evt)
#    print(name)
#    exit()
#exit()

titles = [
    "Sample Processed         ",
    "Exp. Events (uncut)",
    "x-section (fb)",
    "Gen. Events (unweighted)",
    "Sum of Weights (weight_mc)",
    "Sum of Weights (Tree)",
    "Scale factor (Exp. / sow Tree)",
    "Filter efficiency (passed events / processed)",
    ""
]

h = []
print(" | ".join(titles))
for i in list(proc.values()):
    prx = i.proc

    tmp = ""
    tmp = buffs(titles[0], prx, tmp)
    tmp = buffs(titles[1], i.expected_events, tmp)
    tmp = buffs(titles[2], i.cross_section, tmp)
    tmp = buffs(titles[3], i.num_events, tmp)
    tmp = buffs(titles[4], i.sow_nominal, tmp)
    tmp = buffs(titles[5], i.sow_weights, tmp)
    tmp = buffs(titles[6], i.scale_factor, tmp)
    tmp = buffs(titles[7], i.filtereff, tmp)
    h.append(i.hist)
    print(tmp)

comb = TH1F()
comb.yScaling = 5*4.8
comb.xScaling = 5*6.4
comb.yLogarithmic = True
comb.Histograms = h
comb.Title = "Weighted Missing Transverse Energy"
comb.xTitle = "Missing ET (GeV)"
comb.yTitle = "Entries / (2 GeV)"
comb.xBins = 500
comb.xMin = 0
comb.xMax = 1000
comb.yMin = 0.01
comb.xStep = 40
comb.Filename = "weighted_hists"
comb.Stacked = True
comb.SaveFigure()


