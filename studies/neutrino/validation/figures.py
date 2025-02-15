from AnalysisG.core.plotting import TH1F, TH2F, TLine
from .common import *
from .helper_fig import *
import math


def mass_differential(dt, mode, out):
    r1_cu, r2_cu = dt["r1_cu"], dt["r2_cu"]
    r1_rf, r2_rf = dt["r1_rf"], dt["r2_rf"]
    truth_nux = dt["truth_nux"]

    tmass_r1_n1, wmass_r1_n1 = get_population(r1_cu, truth_nux, "n1")
    tmass_r1_n2, wmass_r1_n2 = get_population(r1_cu, truth_nux, "n2")

    tmass_r2_n1, wmass_r2_n1 = get_population(r2_cu, truth_nux, "n1")
    tmass_r2_n2, wmass_r2_n2 = get_population(r2_cu, truth_nux, "n2")

    thcu_r1 = template_hist("CUDA - Dynamic", tmass_r1_n1 + tmass_r1_n2, "red")
    thcu_r2 = template_hist("CUDA - Static" , tmass_r2_n1 + tmass_r2_n2, "blue")

    th = path(TH1F(), out)
    th.Histograms = [thcu_r1, thcu_r2]
    th.xBins = 100
    th.xMax = 0.2
    th.xMin = -0.2
    th.xStep = 0.1
    th.xTitle = "Invariant Top Mass Error ($(M^r_{top} - M^t_{top}) / M^t_{top}$)"
    th.yTitle = "Tops (Arb.)"
    th.Filename = "Figure.10.h"
    th.SaveFigure()


    tmass_r1_n1, wmass_r1_n1 = get_population(r1_rf, truth_nux, "n1")
    tmass_r1_n2, wmass_r1_n2 = get_population(r1_rf, truth_nux, "n2")

    tmass_r2_n1, wmass_r2_n1 = get_population(r2_rf, truth_nux, "n1")
    tmass_r2_n2, wmass_r2_n2 = get_population(r2_rf, truth_nux, "n2")

    thcu_r1 = template_hist("Reference - Dynamic", tmass_r1_n1 + tmass_r1_n2, "red")
    thcu_r2 = template_hist("Reference - Static" , tmass_r2_n1 + tmass_r2_n2, "blue")

    th = path(TH1F(), out)
    th.Histograms = [thcu_r1, thcu_r2]
    th.xBins = 100
    th.xMax = 0.2
    th.xMin = -0.2
    th.xStep = 0.1
    th.xTitle = "Invariant Top Mass Error ($(M^r_{top} - M^t_{top}) / M^t_{top}$)"
    th.yTitle = "Tops (Arb.)"
    th.Filename = "Figure.10.i"
    th.SaveFigure()


def chi2_neutrinos(dt, mode, out):
    r1_cu, r2_cu = dt["r1_cu"], dt["r2_cu"]
    r1_rf, r2_rf = dt["r1_rf"], dt["r2_rf"]
    truth_nux = dt["truth_nux"]

    chi2_r1_n1 = r1_cu["chi2"]["n1"]
    chi2_r1_n2 = r1_cu["chi2"]["n2"]
    chi2_r2_n1 = r2_cu["chi2"]["n1"]
    chi2_r2_n2 = r2_cu["chi2"]["n2"]

    thcu_r1 = template_hist("CUDA - Dynamic", [math.log10(i) for i in list(chi2_r1_n1 + chi2_r1_n2)], "red")
    thcu_r2 = template_hist("CUDA - Static" , [math.log10(i) for i in list(chi2_r2_n1 + chi2_r2_n2)], "blue")
    thcu_r1.Alpha = 0.5
    thcu_r2.Alpha = 0.5

    th = path(TH1F(), out)
    th.Title = "$\\chi^2$ Distribution of Neutrino Candidates"
    th.Histograms = [thcu_r1, thcu_r2]
    th.xBins = 100
    th.xMax = 10
    th.xMin = 0
    th.xStep = 0.5
    th.Density = True
    th.yLogarithmic = True
    th.yTitle = "Neutrino Candidates"
    th.xTitle = "$\\chi^2$ of Neutrino Candidates"
    th.Filename = "Figure.10.j"
    th.SaveFigure()


    chi2_r1_n1 = r1_rf["chi2"]["n1"]
    chi2_r1_n2 = r1_rf["chi2"]["n2"]
    chi2_r2_n1 = r2_rf["chi2"]["n1"]
    chi2_r2_n2 = r2_rf["chi2"]["n2"]

    thcu_r1 = template_hist("Reference - Dynamic", [math.log10(i) for i in list(chi2_r1_n1 + chi2_r1_n2)], "red")
    thcu_r2 = template_hist("Reference - Static" , [math.log10(i) for i in list(chi2_r2_n1 + chi2_r2_n2)], "blue")
    thcu_r1.Alpha = 0.5
    thcu_r2.Alpha = 0.5

    th = path(TH1F(), out)
    th.Title = "$\\chi^2$ Distribution of Neutrino Candidates"
    th.Histograms = [thcu_r1, thcu_r2]
    th.xBins = 100
    th.xMax = 10
    th.xMin = 0
    th.xStep = 0.5
    th.Density = True
    th.yLogarithmic = True
    th.yTitle = "Neutrino Candidates"
    th.xTitle = "$\\chi^2$ of Neutrino Candidates"
    th.Filename = "Figure.10.k"
    th.SaveFigure()


def define_metrics(dt, tlt):
    r1_cu, r2_cu = dt["r1_cu"], dt["r2_cu"]
    r1_rf, r2_rf = dt["r1_rf"], dt["r2_rf"]
    truth_nux = dt["truth_nux"]

    # -------- Case one ----------- #
    n_events = len(truth_nux["tmass"]["n1"])
    c1, c2 = sum(r1_cu["missed"]), sum(r2_cu["missed"])
    r1, r2 = sum(r1_rf["missed"]), sum(r2_rf["missed"])

    # --------- Case two: Inconsistent candidates ---------- #
    tmass_c1_n1, wmass_c1_n1 = get_population(r1_cu, truth_nux, "n1")
    tmass_c1_n2, wmass_c1_n2 = get_population(r1_cu, truth_nux, "n2")
    loss_c1, ok_c1 = get_efficiency(tmass_c1_n1, tmass_c1_n2, wmass_c1_n1, wmass_c1_n2, 0.2)

    tmass_c2_n1, wmass_c2_n1 = get_population(r2_cu, truth_nux, "n1")
    tmass_c2_n2, wmass_c2_n2 = get_population(r2_cu, truth_nux, "n2")
    loss_c2, ok_c2 = get_efficiency(tmass_c2_n1, tmass_c2_n2, wmass_c2_n1, wmass_c2_n2, 0.2)

    tmass_r1_n1, wmass_r1_n1 = get_population(r1_rf, truth_nux, "n1")
    tmass_r1_n2, wmass_r1_n2 = get_population(r1_rf, truth_nux, "n2")
    loss_r1, ok_r1 = get_efficiency(tmass_r1_n1, tmass_r1_n2, wmass_r1_n1, wmass_r1_n2, 0.2)

    tmass_r2_n1, wmass_r2_n1 = get_population(r2_rf, truth_nux, "n1")
    tmass_r2_n2, wmass_r2_n2 = get_population(r2_rf, truth_nux, "n2")
    loss_r2, ok_r2 = get_efficiency(tmass_r2_n1, tmass_r2_n2, wmass_r2_n1, wmass_r2_n2, 0.2)

    sx = 3
    print("------------ " + tlt + " ------------")
    print("raw events:", n_events)
    print("------ Case 1: Missed Reconstruction -------")
    print("cuda - dyn:", round(c1 / n_events, sx), "cuda-static:", round(c2 / n_events, sx))
    print("ref - dyn:" , round(r1 / n_events, sx), "ref-static:" , round(r2 / n_events, sx))

    print("------ Case 2: Inconsistent Parents --------")
    print("cuda - dyn:", round((c1 - loss_c1)/ n_events, sx), "cuda-static:", round((c2 - loss_c2) / n_events, sx))
    print("ref - dyn:" , round((r1 - loss_r1)/ n_events, sx), "ref-static:" , round((r2 - loss_r2) / n_events, sx))

def topchildren_nunu(ana):
    out = "top-children"
    dt = topchildren_nunu_build(ana)
    template_nunu_top_mass(dt, "a", "b", "c", "d", " \n Truth Children", out)
    test_implementations(dt, "TopChildren", out, [150, 200], 100, 4, "0.5")
    distance_chi2(dt, "TopChildren", "top-children")
    mass_differential(dt, "", "top-children")
    define_metrics(dt, "TopChildren")
    #chi2_neutrinos(dt, "TopChildren", out)

def toptruthjets_nunu(ana):
    out = "truthjets-children"
    dt = toptruthjets_nunu_build(ana)
    template_nunu_top_mass(dt, "a", "b", "c", "d", " \n Truth Jets with Leptonic Truth Children", out)
    test_implementations(dt, "TruthJets", out, [150, 200], 100, 4, "0.5")
    distance_chi2(dt, "Truth Jets with Leptonic Truth Children", "truthjets-children")
    mass_differential(dt, "", "truthjets-children")
    define_metrics(dt, "TruthJet Leptonic TopChildren")

def topjetchild_nunu(ana):
    out = "jets-children"
    dt = topjetchild_nunu_build(ana)
    template_nunu_top_mass(dt, "a", "b", "c", "d", " \n Detector Jets with Leptonic Truth Children", out)
    test_implementations(dt, "JetsChildren", out, [100, 250], 150, 10, "1.0")
    distance_chi2(dt, "Detector Jets with Leptonic Truth Children", "jets-children")
    mass_differential(dt, "", "jets-children")
    define_metrics(dt, "Detector Jets Leptonic TopChildren")

def topdetector_nunu(ana):
    out = "jets-detector"
    dt = topdetector_nunu_build(ana)
    template_nunu_top_mass(dt, "a", "b", "c", "d", " \n Detector Jets and Leptons", out)
    test_implementations(dt, "JetsLeptons", out, [100, 250], 150, 10, "1.0")
    distance_chi2(dt, "Detector Jets and Leptons", "jets-detector")
    mass_differential(dt, "", "jets-detector")
    define_metrics(dt, "Detector Jets and Leptons")

def nunuValidation(ana):
    topchildren_nunu(ana)
    #toptruthjets_nunu(ana)
    #topjetchild_nunu(ana)
    #topdetector_nunu(ana)
    #LossStatistics()
    pass



