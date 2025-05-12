from AnalysisG.core.plotting import TH1F, TH2F
from .helper import compxl
from .classes import loss

global eps

typx = ["topchildren", "truthjet", "jetchildren", "jetleptons"]
def statistics(data, eps = None):
    losses = {i : loss(eps) for i in typx}
    for i in range(len(data["topchildren"])):
        if data["topchildren"][i] is None: continue
        for k in typx: losses[k].add(data[k][i])
    return losses


def colors():
    return iter(sorted(list(set([
        "aqua", "orange", "green","blue","olive","teal","gold",
        "darkblue","lime","crimson","magenta","orchid",
        "sienna","silver","salmon","chocolate", "navy", "indigo", 
    ]))))

def print_statistics(data):
    for i in typx: 
        print(i + " " + data[i].__str__())
        for k in data["KSTest"][i]:
            print(k + "-> ", data["KSTest"][i][k])

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
    if tl == "cuda"       : return "CUDA"
    if tl == "ref"        : return "Reference"
    return tl

def default(hist, pth):
    hist.Style = "ATLAS"
    if pth is not None: 
        if eps is None: pth = "-noeps/"  + pth
        else: pth = "-eps" + str(int(eps)) + "/" + pth
        hist.OutputDirectory = "figures" + pth
    hist.DPI = 300
    hist.TitleSize = 14
    hist.AutoScaling = True
    hist.Overflow = False
    hist.yScaling = 4 #10*0.75
    hist.xScaling = 4 #15*0.6
    hist.FontSize = 9
    hist.AxisSize = 9
    hist.LegendSize = 10
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

def KSTestingDynamic(data):
    para = {
        "xTitle" : "Invariant Mass of Leptonic Tops (GeV)", 
        "yTitle" : "Density (arb.) / <units> (GeV)", 
        "Alpha" : 0.25, 
        "xMax"   : 240, "xMin": 80, "xStep" : 20, "xBins" : 160,
        "Density" : True,
    }
    out = {"KSTest" : {}}
    for i in data:
        tru = data[i].truth
        cu_dnu = data[i].cuda_dnu
        rf_dnu = data[i].ref_dnu 
        out["KSTest"][i] = {}

        thc_dnu = template(cu_dnu.top1_masses, Fancy("cuda") , "red" , None, {"Density" : True})
        thr_dnu = template(rf_dnu.top1_masses, Fancy("ref")  , "blue", None, {"Hatch" : "////\\\\\\\\", "Marker" : "x", "Density" : True})
        th = template(None, "Reconstructed Tops ($\\nu_{1}$)", None, "ks-test/" + i, para)
        th.Histogram = thr_dnu
        th.Histograms = [thc_dnu]
        th.FX("chi2")
        th.Filename = "neutrino_1"
        th.SaveFigure()

        thc_dnu = template(cu_dnu.top2_masses, Fancy("cuda") , "red" , None, {"Density" : True})
        thr_dnu = template(rf_dnu.top2_masses, Fancy("ref")  , "blue", None, {"Hatch" : "////\\\\\\\\", "Marker" : "x", "Density" : True})
        th = template(None, "Reconstructed Tops ($\\nu_{2}$)", None, "ks-test/" + i, para)
        th.Histogram = thr_dnu
        th.Histograms = [thc_dnu]
        th.FX("chi2")
        th.Filename = "neutrino_2"
        th.SaveFigure()

        nex = dict(para)
        nex["Density"] = False
        nex["Alpha"] = 0.5
        nex["yTitle"] = "Reconstructed (Truth) Tops / <units> (GeV)"
        cu_tht_dnu = template(cu_dnu.target_top1 + cu_dnu.target_top2, Fancy("truth")   , "black", None, nex)
        r_cu_dnu   = template(cu_dnu.top1_masses + cu_dnu.top2_masses, Fancy("cuda_dnu"), "red"  , None, nex)
        th = template(None, "Reconstructed Leptonic Tops ($\\nu_{1}$ and $\\nu_{2}$)\n", None, "ks-test/" + i, nex)
        th.Histogram = cu_tht_dnu
        th.Histograms = [r_cu_dnu]
        th.FX("chi2")
        th.Filename = "cuda-dynamic"
        th.SaveFigure()
        
        rf_tht_dnu = template(rf_dnu.target_top1 + rf_dnu.target_top2, Fancy("truth")  , "black", None, nex)
        r_rf_dnu   = template(rf_dnu.top1_masses + rf_dnu.top2_masses, Fancy("ref_dnu"), "red"  , None, nex)
        th = template(None, "Reconstructed Leptonic Tops ($\\nu_{1}$ and $\\nu_{2}$)\n", None, "ks-test/" + i, nex)
        th.Histogram = rf_tht_dnu
        th.Histograms = [r_rf_dnu]
        th.FX("chi2")
        th.Filename = "ref-dynamic"
        th.SaveFigure()

        r_rf_dnu.xMax = 200
        r_rf_dnu.xMin = 120
        rf_tht_dnu.xMax = 200
        rf_tht_dnu.xMin = 120


        r_cu_dnu.xMax = 200
        r_cu_dnu.xMin = 120
        cu_tht_dnu.xMax = 200
        cu_tht_dnu.xMin = 120

        #out["KSTest"][i]["nunu-cuda-ref"]   = r_rf_dnu.KStest(r_cu_dnu)
        #out["KSTest"][i]["nunu-truth-ref"]  = rf_tht_dnu.KStest(r_rf_dnu)
        #out["KSTest"][i]["nunu-truth-cuda"] = cu_tht_dnu.KStest(r_cu_dnu)

    return out


def Chi2Distribution(data):
    para = {
        "xTitle" : "$\\chi^2$ Kinematic Error ($\\text{GeV}^2$)", 
        "yTitle" : "Density (Arb.) / <units>", 
        "Alpha" : 0.25, 
        "xMax"   : 4000, "xMin": 0, "xStep" : 800, "xBins" : 200,
        "Density" : False,
    }
    out = {"chi2" : {}}
    for i in data:
        cu_dnu, cu_snu = data[i].cuda_dnu, data[i].cuda_snu
        rf_dnu, rf_snu = data[i].ref_dnu , data[i].ref_snu

        thc_dnu = template(cu_dnu.chi2_nu1, Fancy("cuda_dnu") , "red" , None, {"Density" : True})
        thr_dnu = template(rf_dnu.chi2_nu1, Fancy("ref_dnu")  , "blue", None, {"Density" : True})

        thc_snu = template(cu_snu.chi2_nu1, Fancy("cuda_snu") , "red" , None, {"Hatch" : "////\\\\\\\\", "Density" : True, "Alpha" : 0.75})
        thr_snu = template(rf_snu.chi2_nu1, Fancy("ref_snu")  , "blue", None, {"Hatch" : "////\\\\\\\\", "Density" : True, "Alpha" : 0.75})
        th = template(None, "$\\chi^2$ for Reconstructed $\\nu_{1}$ ", None, "chi2/" + i, para)
        lx = [thc_snu, thr_snu, thc_dnu, thr_dnu]
        th.Histograms = lx
        th.Filename = "neutrino_1"
        th.SaveFigure()

        thc_dnu = template(cu_dnu.chi2_nu2, Fancy("cuda_dnu") , "red" , None, {"Density" : True})
        thr_dnu = template(rf_dnu.chi2_nu2, Fancy("ref_dnu")  , "blue", None, {"Density" : True})

        thc_snu = template(cu_snu.chi2_nu2, Fancy("cuda_snu") , "red" , None, {"Hatch" : "////\\\\\\\\", "Density" : True, "Alpha" : 0.75})
        thr_snu = template(rf_snu.chi2_nu2, Fancy("ref_snu")  , "blue", None, {"Hatch" : "////\\\\\\\\", "Density" : True, "Alpha" : 0.75})
        th = template(None, "$\\chi^2$ for Reconstructed $\\nu_{2}$ ", None, "chi2/" + i, para)
        lx = [thc_snu, thr_snu, thc_dnu, thr_dnu]
        th.Histograms = lx
        th.Filename = "neutrino_2"
        th.SaveFigure()

    return out

def EllipseDistance(data):
    def thx(tlt, xdata, ydata, i):
        xh = default(TH2F(), "ellipse/" + i)
        xh.Title = tlt
        xh.Color = "inferno"
        xh.xData = xdata
        xh.xBins = 100
        xh.xMin  = -50
        xh.xMax  = -10
        xh.xStep = 5.0
        xh.xTitle = "Ellipse Distance (Arb.)"

        xh.yData = ydata
        xh.yBins = 10
        xh.yMin  = 0
        xh.yMax  = 400
        xh.yStep = 40
        xh.yTitle = "Kinematic Error for Reconstructed Neutrino ($\\text{GeV}^2$)"
        return xh

    import math
    para = {
        "xTitle" : "Ellipse Distance (Arb.)", 
        "yTitle" : "Density (Arb.) / <units>", 
        "Alpha" : 0.25, 
        "xMax"   : 5, "xMin": -50, "xStep" : 5.0, "xBins" : 100,
    }
    title = "Distances of Elliptical Intersection"
    for i in data:
        cu_dnu, cu_snu = data[i].cuda_dnu, data[i].cuda_snu
        rf_dnu, rf_snu = data[i].ref_dnu , data[i].ref_snu

        thc_dnu = template(cu_dnu.distances, Fancy("cuda_dnu") , "red" , None, {"Density" : True})
        thr_dnu = template([math.log10(k if k > 0 else 1) for k in rf_dnu.distances], Fancy("ref_dnu")  , "blue", None, {"Density" : True})

        thc_snu = template(cu_snu.distances, Fancy("cuda_snu") , "red" , None, {"Hatch" : "////\\\\\\\\", "Density" : True})
        thr_snu = template([math.log10(k if k > 0 else 1) for k in rf_snu.distances], Fancy("ref_snu")  , "blue", None, {"Hatch" : "////\\\\\\\\", "Density" : True})
        th = template(None, title, None, "ellipse/" + i, para)
        th.Histograms = [thc_snu, thr_snu, thc_dnu, thr_dnu]
        th.Filename = "distances"
        th.SaveFigure()
   
        th = thx(Fancy("cuda_dnu"), cu_dnu.distances + cu_dnu.distances, cu_dnu.chi2_nu1 + cu_dnu.chi2_nu2, i)
        th.Filename = "correlation-cuda-dynamic"
        th.SaveFigure()

        th = thx(Fancy("cuda_snu"), cu_snu.distances + cu_snu.distances, cu_snu.chi2_nu1 + cu_snu.chi2_nu2, i)
        th.Filename = "correlation-cuda-static"
        th.SaveFigure()

        th = thx(Fancy("ref_dnu"), rf_dnu.distances + rf_dnu.distances, rf_dnu.chi2_nu1 + rf_dnu.chi2_nu2, i)
        th.Filename = "correlation-reference-dynamic"
        th.SaveFigure()

        th = thx(Fancy("ref_snu"), rf_snu.distances + rf_snu.distances, rf_snu.chi2_nu1 + rf_snu.chi2_nu2, i)
        th.Filename = "correlation-reference-static"
        th.SaveFigure()

def Chi2Symbolic(data):
    para = {
        "xTitle" : "Kinematic Error for Reconstructed Neutrino ($\\chi^2$) (Arb.)", 
        "yTitle" : "Density (Arb.) / <units>", 
        "Alpha" : 0.25, 
        "xMax"   : 4000, "xMin": 0, "xStep" : 400, "xBins" : 200,
        "Density" : True,
    }

    for i in data:
        cu_dnu, cu_snu = data[i].cuda_dnu, data[i].cuda_snu
        rf_dnu, rf_snu = data[i].ref_dnu , data[i].ref_snu
        
        th = template(None, "Kinematic Error by Leptonic Pairs", None, "chi2/" + i, para | {"Stacked" : False})
        hists = []
        col = colors()
        lx = list(cu_dnu.symbols)
        for k in sorted(lx): 
            ch = cu_dnu.symbols[k].chi2_nu1 + cu_dnu.symbols[k].chi2_nu2
            hists.append(template(ch, "$" + k + "$", next(col)))
        th.Histograms = hists
        th.Filename = "neutrino-cuda-symbolic"
        th.SaveFigure()


    para = {
        "xTitle" : "Ellipse Distance (Arb.)", 
        "yTitle" : "Density (Arb.) / <units>", 
        "Alpha" : 0.25, 
        "xMax"   : 5, "xMin": -50, "xStep" : 5.0, "xBins" : 100,
    }

    title = "Elliptical Intersection by Leptonic Pairs"
    for i in data:
        cu_dnu, cu_snu = data[i].cuda_dnu, data[i].cuda_snu
        rf_dnu, rf_snu = data[i].ref_dnu , data[i].ref_snu

        th = template(None, title, None, "ellipse/" + i, para | {"Stacked" : True, "Density" : True})
        hists = []
        col = colors()
        lx = list(cu_dnu.symbols)
        for k in sorted(lx): 
            ch = cu_dnu.symbols[k].distances
            hists.append(template(ch, "$" + k + "$", next(col)))
        th.Histograms = sorted(hists, reverse = False, key = lambda x: sum(x.counts))
        th.Filename = "neutrino-cuda-symbolic"
        th.SaveFigure()


def METError(data):
    for i in data:
        cu_dnu = data[i].cuda_dnu
    
        xh = default(TH2F(), "met/" + i)
        xh.Color = "inferno"
        xh.Title = "$\\chi^2$ Dependency on Missing Transverse Energy\n" + Fancy("cuda_dnu")
        xh.xData = [k / 1000.0 for k in cu_dnu.event_data["met"]]
        xh.xBins = 100
        xh.xMin  = 0
        xh.xMax  = 1000
        xh.xStep = 100.0
        xh.xTitle = "Missing Transverse Energy (GeV)"

        xh.yData = cu_dnu.chi2_nu1
        xh.yBins = 200
        xh.yMin  = 0
        xh.yMax  = 200
        xh.yStep = 20.0
        xh.yTitle = "Kinematic Error for Reconstructed Neutrino ($\\text{GeV}^2$)"
        xh.Filename = "correlation-cuda-dynamic-chi2"
        xh.SaveFigure()

        xh = default(TH2F(), "met/" + i)
        xh.Color = "inferno"
        xh.Title = "Elliptical Distance Dependency on Missing Transverse Energy"
        xh.xData = [k / 1000.0 for k in cu_dnu.event_data["met"]]
        xh.xBins = 200
        xh.xMin  = 0
        xh.xMax  = 200
        xh.xStep = 20.0
        xh.xTitle = "Missing Transverse Energy (GeV)"

        xh.yData = cu_dnu.distances
        xh.yBins = 100
        xh.yMin  = -40
        xh.yMax  = 0.0
        xh.yStep = 5.0
        xh.yTitle = "Elliptical Distance (Arb.)"
        xh.Filename = "correlation-cuda-dynamic-dist"
        xh.SaveFigure()








def validation(sl = None):
    data = compxl(sl if sl is not None else sl)
    data = statistics(data, eps)
    #data |= KSTestingDynamic(data)
    #Chi2Distribution(data)
    EllipseDistance(data)
    Chi2Symbolic(data)
    METError(data)
    #print_statistics(data)
