from AnalysisG.core.plotting import TH1F, TH2F, TLine
from .common import *
import math

def template_hist(title, xdata, color):
    th1t = TH1F()
    th1t.Title = title
    th1t.Alpha = 0.3
    th1t.xData = xdata
    th1t.Color = color
    return th1t

def LossStatistics():
    a = topchildren_nunu_build()
    b = toptruthjets_nunu_build()
    c = topjetchild_nunu_build()
    d = topdetector_nunu_build()

    a_nevents = len(a["truth_nux"]["tmass"]["n1"])
    a_c1 = sum(a["r1_cu"]["missed"]) * 100 / a_nevents
    a_c2 = sum(a["r2_cu"]["missed"]) * 100 / a_nevents
    a_r1 = sum(a["r1_rf"]["missed"]) * 100 / a_nevents
    a_r2 = sum(a["r2_rf"]["missed"]) * 100 / a_nevents

    b_nevents = len(b["truth_nux"]["tmass"]["n1"])
    b_c1 = sum(b["r1_cu"]["missed"]) * 100 / b_nevents
    b_c2 = sum(b["r2_cu"]["missed"]) * 100 / b_nevents
    b_r1 = sum(b["r1_rf"]["missed"]) * 100 / b_nevents
    b_r2 = sum(b["r2_rf"]["missed"]) * 100 / b_nevents

    c_nevents = len(c["truth_nux"]["tmass"]["n1"])
    c_c1 = sum(c["r1_cu"]["missed"]) * 100 / c_nevents
    c_c2 = sum(c["r2_cu"]["missed"]) * 100 / c_nevents
    c_r1 = sum(c["r1_rf"]["missed"]) * 100 / c_nevents
    c_r2 = sum(c["r2_rf"]["missed"]) * 100 / c_nevents

    d_nevents = len(d["truth_nux"]["tmass"]["n1"])
    d_c1 = sum(d["r1_cu"]["missed"]) * 100 / d_nevents
    d_c2 = sum(d["r2_cu"]["missed"]) * 100 / d_nevents
    d_r1 = sum(d["r1_rf"]["missed"]) * 100 / d_nevents
    d_r2 = sum(d["r2_rf"]["missed"]) * 100 / d_nevents

    sx = 3
    print("------------ Truth Children ------------")
    print("raw events:", a_nevents)
    print("cuda - dyn:", round(a_c1, sx), "cuda-static:", round(a_c2, sx))
    print("ref - dyn:" , round(a_r1, sx), "ref-static:" , round(a_r2, sx))

    print("------------ Truth Jets + Truth Children ------------")
    print("raw events:", b_nevents)
    print("cuda - dyn:", round(b_c1, sx), "cuda-static:", round(b_c2, sx))
    print("ref - dyn:" , round(b_r1, sx), "ref-static:" , round(b_r2, sx))

    print("------------ Jets + Truth Children ------------")
    print("raw events:", c_nevents)
    print("cuda - dyn:", round(c_c1, sx), "cuda-static:", round(c_c2, sx))
    print("ref - dyn:" , round(c_r1, sx), "ref-static:" , round(c_r2, sx))

    print("------------ Jets + Leptons ------------")
    print("raw events:", d_nevents)
    print("cuda - dyn:", round(d_c1, sx), "cuda-static:", round(d_c2, sx))
    print("ref - dyn:" , round(d_r1, sx), "ref-static:" , round(d_r2, sx))

def template_nunu_top_mass(dt, a, b, c, d, mode, out_path):
    r1_cu, r2_cu = dt["r1_cu"], dt["r2_cu"]
    r1_rf, r2_rf = dt["r1_rf"], dt["r2_rf"]
    truth_nux = dt["truth_nux"]

    for i in [("n1", a), ("n2", b)]:
        n, f = i
        th1t = template_hist("Truth"              , truth_nux["tmass"][n], "red")
        th1c = template_hist("CUDA - Dynamic"     , r1_cu["tmass"][n]    , "blue")
        th1r = template_hist("Reference - Dynamic", r1_rf["tmass"][n]    , "green")

        th = path(TH1F(), out_path)
        th.Title = "Invariant Top Quark Mass derived from Neutrino " + n.replace("n", "") + ": " + mode
        th.Histograms = [th1t, th1c, th1r]
        th.xBins = 400
        th.xMax = 400
        th.xMin = 0
        th.xStep = 40
        th.Overflow = False
        th.Density = True
        th.xTitle = "Invariant Top Mass (GeV)"
        th.yTitle = "Density (Arb.) / 1 GeV"
        th.Filename = "Figure.10." + f
        th.SaveFigure()

    for i in [("n1", c), ("n2", d)]:
        n, f = i
        th1t = template_hist("Truth"             , truth_nux["tmass"][n], "red")
        th1c = template_hist("CUDA - Static"     , r2_cu["tmass"][n]    , "blue")
        th1r = template_hist("Reference - Static", r2_rf["tmass"][n]    , "green")

        th = path(TH1F(), out_path)
        th.Title = "Invariant Top Quark Mass derived from Neutrino " + n.replace("n", "") + ": " + mode
        th.Histograms = [th1t, th1c, th1r]
        th.xBins = 400
        th.xMax = 400
        th.xMin = 0
        th.xStep = 40
        th.Overflow = False
        th.Density = True
        th.xTitle = "Invariant Top Mass (GeV)"
        th.yTitle = "Density (Arb.) / 1 GeV"
        th.Filename = "Figure.10." + f
        th.SaveFigure()

def distance_chi2(dt, mode, out):
    r1_cu, r2_cu = dt["r1_cu"], dt["r2_cu"]
    r1_rf, r2_rf = dt["r1_rf"], dt["r2_rf"]
    truth_nux = dt["truth_nux"]

    dst_1 = [-1 * i for i in r1_cu["dst"]]
    dst_2 = [-1 * i for i in r2_cu["dst"]]

    cdst_1 = template_hist("CUDA (Dynamic)", dst_1, "red")
    cdst_2 = template_hist("CUDA (Static)" , dst_2, "blue")

    tdst = path(TH1F(), out)
    tdst.Histogram  = cdst_1
    tdst.Histograms = [cdst_2]
    tdst.Stacked = True
    tdst.Title = "Double Neutrino Distance Metric Between Ellipses \n (" + mode + ")"
    tdst.yTitle = "Events (Arb.) / 0.1"
    tdst.xMin = 0
    tdst.xMax = 50
    tdst.xStep = 5
    tdst.xBins = 50
    tdst.xTitle = "$-\\log_{10}$ - Ellipse Intersection Distance"
    tdst.Filename = "Figure.10.e"
    tdst.SaveFigure()

    dst_mass = path(TH2F(), out)
    dst_mass.Title = "Invariant Top Mass with Associated Ellipse Intersection Distance \n (" + mode + ")"
    dst_mass.xData = r1_cu["tmass"]["n1"]
    dst_mass.xMin = 0
    dst_mass.xMax = 400
    dst_mass.xStep = 20
    dst_mass.xBins = 400
    dst_mass.xTitle = "Invariant Top Mass (GeV)"

    dst_mass.yData = dst_1
    dst_mass.yMin = 0
    dst_mass.yMax = 50
    dst_mass.yStep = 5
    dst_mass.yBins = 50
    dst_mass.yTitle = "$-\\log_{10}$ - Ellipse Intersection Distance"
    dst_mass.Filename = "Figure.10.f"
    dst_mass.Color = "RdBu_r"
    dst_mass.SaveFigure()


    dst_ch2 = path(TH2F(), out)
    dst_ch2.Title = "$\\chi^2$ Error for Neutrino 1 Kinematics with Associated Ellipse Intersection Distance \n (" + mode + ")"
    dst_ch2.xData = [math.log10(x) for x in r1_cu["chi2"]["n1"]]
    dst_ch2.xMin = -10
    dst_ch2.xMax = 10
    dst_ch2.xStep = 2
    dst_ch2.xBins = 200
    dst_ch2.xTitle = "$\\log_{10}(\\chi^2)$"

    dst_ch2.yData = dst_1
    dst_ch2.yMin = 0
    dst_ch2.yMax = 50
    dst_ch2.yStep = 5
    dst_ch2.yBins = 50
    dst_ch2.yTitle = "$-\\log_{10}$ - Ellipse Intersection Distance"
    dst_ch2.Filename = "Figure.10.g"
    dst_ch2.Color = "RdBu_r"
    dst_ch2.SaveFigure()







