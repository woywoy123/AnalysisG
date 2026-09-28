from AnalysisG.core import TH1F, TH2F
from atomics import Dataloader
from utils import *

def default(tl):
    tl.Style = "ATLAS"
    tl.DPI = 300
    tl.TitleSize = 15
    tl.AutoScaling = True
    tl.LegendSize = 10
    tl.yScaling = 5
    tl.xScaling = 10
    tl.FontSize = 10
    tl.AxisSize = 10
    tl.LineWidth = 0.1
    return tl


def templ_th1f(name, xbins, xmin, xmax, xdata, xtitle, ytitle, hists = None, colrs = None):
    tf = default(TH1F())
    tf.Title = name
    tf.xData = xdata;   tf.xBins  = xbins
    tf.xTitle = xtitle; tf.yTitle = ytitle
    tf.xMin   = xmin;   tf.xMax   = xmax
    if hists is None: return tf
    tf.Histograms = hists
    if colrs is None: return tf
    for i in hists: i.Color = next(colrs)
    return tf

def basic_th1f(name, data):
    tf = TH1F()
    tf.Title = name
    tf.xData = data
    return tf

def make_truth(dt):
    prc = {}
    for i in dt:
        if "Truth" not in i: continue
        if len(i.masses) == 0: continue
        if i.var not in prc: prc[i.var] = {}
        if i.prc not in prc[i.var]: prc[i.var][i.prc] = []
        prc[i.var][i.prc] += i.masses

    for i in prc["RawTruth"]: prc["RawTruth"][i] = basic_th1f(i, prc["RawTruth"][i])
    th1 = templ_th1f(
            "", 400, 0, 400, [], 
            "Invariant Mass of Truth Tops (GeV)", 
            "Number of Tops (Arb.) / (1 GeV)", sort_hist(prc["RawTruth"].values()), colors()
    )
    th1.xStep = 20
    th1.Stacked = True
    th1.Filename = "TruthTopsRaw"
    th1.OutputDirectory = "./Plots/TruthInformation/"
    th1.SaveFigure()

    for i in prc["AdjTruth"]: prc["AdjTruth"][i] = basic_th1f(i, prc["AdjTruth"][i])
    th1 = templ_th1f(
            "", 400, 0, 400, [], 
            "Invariant Mass of Truth Tops (GeV)", 
            "Number of Tops (Arb.) / (1 GeV)", sort_hist(prc["AdjTruth"].values()), colors()
    )
    th1.xStep    = 20
    th1.Stacked  = True
    th1.Filename = "TruthTopsAdj"
    th1.OutputDirectory = "./Plots/TruthInformation/"
    th1.SaveFigure()


def entry(mdl, kfold, epoch, base_dir):
    dt = Dataloader(["MRK-1"], ["kfold-1"], ["epoch-1"], base_dir)
    make_truth(dt)

