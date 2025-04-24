from AnalysisG.core.plotting import TH1F, TH2F
from helper import compxl
from classes import loss

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
        "darkblue","azure","lime","crimson","cyan","magenta","orchid",
        "sienna","silver","salmon","chocolate", "navy", "plum", "indigo", 
        "lightblue", "lavender", "ivory"
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
    if pth is not None: hist.OutputDirectory = "figures/" + pth
    hist.DPI = 300
    hist.TitleSize = 15
    hist.AutoScaling = True
    hist.Overflow = False
    hist.yScaling = 5 #10*0.75
    hist.xScaling = 5 #15*0.6
    hist.FontSize = 10
    hist.AxisSize = 10
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
        "xTitle" : "Invariant Mass for Leptonic Tops (GeV)", 
        "yTitle" : "Density (Arb.) / <units> (GeV)", 
        "Alpha" : 0.25, 
        "xMax"   : 400, "xMin": 0, "xStep" : 40, "xBins" : 400,
        "Density" : False,
    }
    out = {"KSTest" : {}}
    for i in data:
        tru = data[i].truth
        cu_dnu = data[i].cuda_dnu
        rf_dnu = data[i].ref_dnu 
        out["KSTest"][i] = {}

        thc_dnu = template(cu_dnu.top1_masses, Fancy("cuda") , "red" , None, {"Density" : True})
        thr_dnu = template(rf_dnu.top1_masses, Fancy("ref")  , "blue", None, {"Hatch" : "////\\\\\\\\", "Marker" : "x", "Density" : True})
        th = template(None, "Reconstructed Leptonic Top ($\\nu_{1}$) Shape Comparison " + Fancy(i), None, "ks-test/" + i, para)
        th.Histogram = thr_dnu
        th.Histograms = [thc_dnu]
        th.FX("chi2")
        th.Filename = "neutrino_1"
        th.SaveFigure()

        thc_dnu = template(cu_dnu.top2_masses, Fancy("cuda") , "red" , None, {"Density" : True})
        thr_dnu = template(rf_dnu.top2_masses, Fancy("ref")  , "blue", None, {"Hatch" : "////\\\\\\\\", "Marker" : "x", "Density" : True})
        th = template(None, "Reconstructed Leptonic Top ($\\nu_{2}$) Shape Comparison " + Fancy(i), None, "ks-test/" + i, para)
        th.Histogram = thr_dnu
        th.Histograms = [thc_dnu]
        th.FX("chi2")
        th.Filename = "neutrino_2"
        th.SaveFigure()

        nex = dict(para)
        nex["Density"] = False
        nex["Alpha"] = 0.5
        nex["yTitle"] = "Reconstructed (Truth) Leptonic Tops / <units> (GeV)"
        cu_tht_dnu = template(cu_dnu.target_top1 + cu_dnu.target_top2, Fancy("truth")   , "black", None, nex)
        r_cu_dnu   = template(cu_dnu.top1_masses + cu_dnu.top2_masses, Fancy("cuda_dnu"), "red" , None, nex)
        th = template(None, "Reconstructed Leptonic Tops ($\\nu_{1}$ and $\\nu_{2}$)" + Fancy(i), None, "ks-test/" + i, para)
        th.Histogram = cu_tht_dnu
        th.Histograms = [r_cu_dnu]
        th.FX("chi2")
        th.Filename = "cuda-dynamic"
        th.SaveFigure()
        
        rf_tht_dnu = template(rf_dnu.target_top1 + rf_dnu.target_top2, Fancy("truth")  , "black", None, nex)
        r_rf_dnu   = template(rf_dnu.top1_masses + rf_dnu.top2_masses, Fancy("ref_dnu"), "red"  , None, nex)
        th = template(None, "Reconstructed Leptonic Tops ($\\nu_{1}$ and $\\nu_{2}$)" + Fancy(i), None, "ks-test/" + i, para)
        th.Histogram = rf_tht_dnu
        th.Histograms = [r_rf_dnu]
        th.FX("chi2")
        th.Filename = "ref-dynamic"
        th.SaveFigure()

        out["KSTest"][i]["nunu-cuda-ref"]   = r_rf_dnu.KStest(r_cu_dnu)
        out["KSTest"][i]["nunu-truth-ref"]  = rf_tht_dnu.KStest(r_rf_dnu)
        out["KSTest"][i]["nunu-truth-cuda"] = cu_tht_dnu.KStest(r_cu_dnu)

    return out


def Chi2Distribution(data):
    para = {
        "xTitle" : "Kinematic Error for Reconstructed Neutrino ($\\chi^2$) (Arb.)", 
        "yTitle" : "Density (Arb.) / <units>", 
        "Alpha" : 0.25, 
        "xMax"   : 4000, "xMin": 0, "xStep" : 400, "xBins" : 200,
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
        th = template(None, "$\\chi^2$ Kinematic Error for $\\nu_{1}$ " + Fancy(i), None, "chi2/" + i, para)
        th.Histograms = [thc_dnu, thr_dnu, thc_snu, thr_snu]
        th.Filename = "neutrino_1"
        th.SaveFigure()

        thc_dnu = template(cu_dnu.chi2_nu2, Fancy("cuda_dnu") , "red" , None, {"Density" : True})
        thr_dnu = template(rf_dnu.chi2_nu2, Fancy("ref_dnu")  , "blue", None, {"Density" : True})

        thc_snu = template(cu_snu.chi2_nu2, Fancy("cuda_snu") , "red" , None, {"Hatch" : "////\\\\\\\\", "Density" : True, "Alpha" : 0.75})
        thr_snu = template(rf_snu.chi2_nu2, Fancy("ref_snu")  , "blue", None, {"Hatch" : "////\\\\\\\\", "Density" : True, "Alpha" : 0.75})
        th = template(None, "$\\chi^2$ Kinematic Error for $\\nu_{2}$ " + Fancy(i), None, "chi2/" + i, para)
        th.Histograms = [thc_dnu, thr_dnu, thc_snu, thr_snu]
        th.Filename = "neutrino_2"
        th.SaveFigure()

    return out

def EllipseDistance(data):
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
        th = template(None, title + Fancy(i), None, "ellipse/" + i, para)
        th.Histograms = [thc_snu, thr_snu, thc_dnu, thr_dnu]
        th.Filename = "distances"
        th.SaveFigure()
    
        xh = default(TH2F(), "ellipse/" + i)
        xh.Title = "$\\chi^2$ Elliptical Distance Dependency" + Fancy("cuda_dnu")
        xh.xData = cu_dnu.distances + cu_dnu.distances
        xh.xBins = 100
        xh.xMin  = -50
        xh.xMax  = 5.0
        xh.xStep = 5.0
        xh.xTitle = "Ellipse Distance (Arb.)"

        xh.yData = cu_dnu.chi2_nu1 + cu_dnu.chi2_nu2
        xh.yBins = 100
        xh.yMin  = 0
        xh.yMax  = 4000
        xh.yStep = 400
        xh.yTitle = "Kinematic Error for Reconstructed Neutrino ($\\chi^2$) (Arb.)"
        xh.Filename = "correlation-cuda-dynamic"
        xh.SaveFigure()




        xh = default(TH2F(), "ellipse/" + i)
        xh.Title = "$\\chi^2$ Elliptical Distance Dependency" + Fancy("cuda_snu")
        xh.xData = cu_snu.distances + cu_snu.distances
        xh.xBins = 100
        xh.xMin  = -50
        xh.xMax  = 5.0
        xh.xStep = 5.0
        xh.xTitle = "Ellipse Distance (Arb.)"

        xh.yData = cu_snu.chi2_nu1 + cu_snu.chi2_nu2
        xh.yBins = 100
        xh.yMin  = 0
        xh.yMax  = 4000
        xh.yStep = 400
        xh.yTitle = "Kinematic Error for Reconstructed Neutrino ($\\chi^2$) (Arb.)"
        xh.Filename = "correlation-cuda-static"
        xh.SaveFigure()



        xh = default(TH2F(), "ellipse/" + i)
        xh.Title = "$\\chi^2$ Elliptical Distance Dependency" + Fancy("ref_dnu")
        xh.xData = [math.log10(k if k > 0 else 1) for k in list(rf_dnu.distances + rf_dnu.distances)]
        xh.xBins = 100
        xh.xMin  = -50
        xh.xMax  = 5.0
        xh.xStep = 5.0
        xh.xTitle = "Ellipse Distance (Arb.)"

        xh.yData = rf_dnu.chi2_nu1 + rf_dnu.chi2_nu2
        xh.yBins = 100
        xh.yMin  = 0
        xh.yMax  = 4000
        xh.yStep = 400
        xh.yTitle = "Kinematic Error for Reconstructed Neutrino ($\\chi^2$) (Arb.)"
        xh.Filename = "correlation-reference-dynamic"
        xh.SaveFigure()


        xh = default(TH2F(), "ellipse/" + i)
        xh.Title = "$\\chi^2$ Elliptical Distance Dependency" + Fancy("ref_snu")
        xh.xData = [math.log10(k if k > 0 else 1) for k in list(rf_snu.distances + rf_snu.distances)]
        xh.xBins = 100
        xh.xMin  = -50
        xh.xMax  = 5.0
        xh.xStep = 5.0
        xh.xTitle = "Ellipse Distance (Arb.)"

        xh.yData = rf_snu.chi2_nu1 + rf_snu.chi2_nu2
        xh.yBins = 100
        xh.yMin  = 0
        xh.yMax  = 4000
        xh.yStep = 400
        xh.yTitle = "Kinematic Error for Reconstructed Neutrino ($\\chi^2$) (Arb.)"
        xh.Filename = "correlation-reference-static"
        xh.SaveFigure()


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
        
        th = template(None, "$\\chi^2$ Kinematic Error by Symbolics" + Fancy(i), None, "chi2/" + i, para | {"Stacked" : True})
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

    title = "Distances of Elliptical Intersection by Symbolics"
    for i in data:
        cu_dnu, cu_snu = data[i].cuda_dnu, data[i].cuda_snu
        rf_dnu, rf_snu = data[i].ref_dnu , data[i].ref_snu

        th = template(None, title + Fancy(i), None, "ellipse/" + i, para | {"Stacked" : True, "Density" : True})
        hists = []
        col = colors()
        lx = list(cu_dnu.symbols)
        for k in sorted(lx): 
            ch = cu_dnu.symbols[k].distances + cu_dnu.symbols[k].distances
            hists.append(template(ch, "$" + k + "$", next(col)))
        th.Histograms = hists
        th.Filename = "neutrino-cuda-symbolic"
        th.SaveFigure()


def METError(data):
    for i in data:
        cu_dnu = data[i].cuda_dnu
    
        xh = default(TH2F(), "met/" + i)
        xh.Title = "$\\chi^2$ Dependency on Missing Transverse Energy" + Fancy("cuda_dnu")
        xh.xData = [k / 1000.0 for k in cu_dnu.event_data["met"]]
        xh.xBins = 200
        xh.xMin  = 0
        xh.xMax  = 200
        xh.xStep = 20.0
        xh.xTitle = "Missing Transverse Energy (GeV)"

        xh.yData = cu_dnu.chi2_nu1
        xh.yBins = 200
        xh.yMin  = 0
        xh.yMax  = 200
        xh.yStep = 20.0
        xh.yTitle = "Kinematic Error for Reconstructed Neutrino ($\\chi^2$) (Arb.)"
        xh.Filename = "correlation-cuda-dynamic"
        xh.SaveFigure()


def entry(sl = None):
    data = compxl(sl if sl is not None else sl)
    data = statistics(data, 600)
    Chi2Distribution(data)
    EllipseDistance(data)
    Chi2Symbolic(data)
    #METError(data)
    data |= KSTestingDynamic(data)
    print_statistics(data)
