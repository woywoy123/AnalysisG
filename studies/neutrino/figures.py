from AnalysisG.core.plotting import TH1F
from helper import compxl
from classes import loss

typx = ["topchildren", "truthjet", "jetchildren", "jetleptons"]
def statistics(data, verbose = False):
    losses = {i : loss() for i in typx}
    for i in range(len(data["topchildren"])):
        if data["topchildren"][i] is None: continue
        for k in typx: losses[k].add(data[k][i])
    if not verbose: return losses
    for i in typx: print(i + " " + losses[i].__str__())
    return losses

def Fancy(tl):
    if tl == "topchildren": return " (TopChildren) "
    if tl == "truthjet"   : return " (TruthJets) "
    if tl == "jetchildren": return " (JetsChildren) "
    if tl == "jetleptons" : return " (JetsLeptons) "
    if tl == "truth"      : return "Truth"
    if tl == "cuda_dnu"   : return "CUDA - Dynamic"
    if tl == "cuda_snu"   : return "CUDA - Static"
    if tl == "ref_dnu"    : return "Reference - Dynamic"
    if tl == "ref_snu"    : return "Reference - Static"
    return tl

def default(hist, pth):
    hist.Style = "ATLAS"
    if pth is not None: hist.OutputDirectory = "figures/" + pth
    hist.DPI = 250
    hist.TitleSize = 20
    hist.AutoScaling = True
    hist.Overflow = False
    hist.yScaling = 10*0.75
    hist.xScaling = 15*0.6
    hist.FontSize = 15
    hist.AxisSize = 15
    hist.LegendSize = 15
    return hist

def template(xdata, title, color = None, pth = None, params = {}):
    thl = default(TH1F(), pth) if xdata is None else TH1F()
    if xdata is not None: thl.xData = xdata
    if color is not None: thl.Color = color
    thl.xMin = 0
    thl.Alpha = 0.5
    thl.Title = title
    for i in params: 
        if i == "yTitle": params[i] = params[i].replace("<unit>", str((params["xMax"] - params["xMin"]) / params["xBins"]))
        setattr(thl, i, params[i])
    return thl

def top_masses(data):
    para = {
        "xTitle" : "Invariant Mass for Leptonic Tops (GeV)", 
        "yTitle" : "Entries / <unit> (GeV)", 
        "xMax"   : 400, "xMin": 0, "xStep" : 40, "xBins" : 400
    }

    for i in data:
        tru = data[i].truth
        cu_dnu, cu_snu = data[i].cuda_dnu, data[i].cuda_snu
        rf_dnu, rf_snu = data[i].ref_dnu, data[i].ref_snu

        th_tru  = template(tru.top1_masses   , Fancy("truth")   , "grey")
        thc_dnu = template(cu_dnu.top1_masses, Fancy("cuda_dnu"), "red")
        thc_snu = template(cu_dnu.top1_masses, Fancy("cuda_snu"), "blue")

        thr_dnu = template(rf_dnu.top1_masses, Fancy("ref_dnu"), "red" , None, {"Hatch" : "///"})
        thr_snu = template(rf_dnu.top1_masses, Fancy("ref_snu"), "blue", None, {"Hatch" : "///"})
        
        th = template(None, "Leptonic Top Invariant Mass" + Fancy(i) + "Neutrino 1", None, i, para)
        th.Histograms = [th_tru, thc_dnu, thc_snu, thr_dnu, thr_snu]
        th.Filename = "all_nu1"
        th.SaveFigure()

        th_tru  = template(tru.top2_masses   , Fancy("truth")   , "grey")
        thc_dnu = template(cu_dnu.top2_masses, Fancy("cuda_dnu"), "red")
        thc_snu = template(cu_dnu.top2_masses, Fancy("cuda_snu"), "blue")

        thr_dnu = template(rf_dnu.top2_masses, Fancy("ref_dnu"), "red" , None, {"Hatch" : "///"})
        thr_snu = template(rf_dnu.top2_masses, Fancy("ref_snu"), "blue", None, {"Hatch" : "///"})

        th = template(None, "Leptonic Top Invariant Mass" + Fancy(i) + "Neutrino 2", None, i, para)
        th.Histograms = [thc_dnu, thc_snu, thr_dnu, thr_snu, th_tru]
        th.Filename = "all_nu2"
        th.SaveFigure()

def entry(sl):
    data = compxl(sl)
    data = statistics(data)
    top_masses(data)    
