from AnalysisG.core.plotting import TH1F, TH2F

global figure_path
global mass_point

def path(hist):
    hist.Style = "ATLAS"
    hist.OutputDirectory = figure_path + "/top-kinematics/" + mass_point
    hist.Overflow = False
    return hist

def top_pt(ana):

    thr = TH1F()
    thr.Title = "Resonance"
    thr.xData = ana.res_top_kinematics["pt"]

    ths = TH1F()
    ths.Title = "Spectator"
    ths.xData = ana.spec_top_kinematics["pt"]

    tha = path(TH1F())
    tha.Histograms = [thr, ths]
    tha.Title = "$p_T$ of Truth Tops"
    tha.xTitle = "$p_T$ (GeV)"
    tha.yTitle = "Density (Arb.) / ($10$ GeV)"
    tha.Density = True
    tha.xMin = 0
    tha.xMax = 1000
    tha.xBins = 100
    tha.xStep = 100
    tha.Filename = "Figure.1.a"
    tha.SaveFigure()

def top_energy(ana):

    thr = TH1F()
    thr.Title = "Resonance"
    thr.xData = ana.res_top_kinematics["energy"]

    ths = TH1F()
    ths.Title = "Spectator"
    ths.xData = ana.spec_top_kinematics["energy"]

    tha = path(TH1F())
    tha.Histograms = [thr, ths]
    tha.Title = "Energy of Truth Tops"
    tha.xTitle = "Energy (GeV)"
    tha.yTitle = "Density (Arb.) / ($15$ GeV)"
    tha.Density = True
    tha.xMin = 0
    tha.xMax = 1500
    tha.xBins = 100
    tha.xStep = 150
    tha.Filename = "Figure.1.b"
    tha.SaveFigure()

def top_phi(ana):

    thr = TH1F()
    thr.Title = "Resonance"
    thr.xData = ana.res_top_kinematics["phi"]

    ths = TH1F()
    ths.Title = "Spectator"
    ths.xData = ana.spec_top_kinematics["phi"]

    tha = path(TH1F())
    tha.Histograms = [thr, ths]
    tha.Title = "Azimuthal Angle of Truth Tops"
    tha.xTitle = "Azimuthal Angle ($\\phi$)"
    tha.yTitle = "Density (Arb.) / (0.1)"
    tha.Density = True
    tha.xMin = -3.5
    tha.xMax = 3.5
    tha.xBins = 70
    tha.xStep = 0.5
    tha.Filename = "Figure.1.c"
    tha.SaveFigure()

def top_eta(ana):

    thr = TH1F()
    thr.Title = "Resonance"
    thr.xData = ana.res_top_kinematics["eta"]

    ths = TH1F()
    ths.Title = "Spectator"
    ths.xData = ana.spec_top_kinematics["eta"]

    tha = path(TH1F())
    tha.Histograms = [thr, ths]
    tha.Title = "Pseudorapidity of Truth Tops"
    tha.xTitle = "Pseudorapidity ($\\eta$)"
    tha.yTitle = "Density (Arb.) / 0.1"
    tha.Density = True
    tha.xMin = -6
    tha.xMax = 6
    tha.xBins = 120
    tha.xStep = 1
    tha.Filename = "Figure.1.d"
    tha.SaveFigure()

def top_pt_energy(ana):

    th = path(TH2F())
    th.Filename = "Figure.1.e"
    th.Title = "Energy as a Function of $p_T$ for Resonant Truth-Tops"

    th.xData = ana.res_top_kinematics["pt"]
    th.xMin = 0
    th.xMax = 800
    th.xBins = 160
    th.xStep = 100
    th.xTitle = "$p_T$ of Resonant Truth Top / 5 GeV"

    th.yData = ana.res_top_kinematics["energy"]
    th.yMin = 0
    th.yMax = 800
    th.yBins = 160
    th.yStep = 100
    th.yTitle = "Energy of Resonant Truth Top / 5 GeV"
    th.SaveFigure()

    th = path(TH2F())
    th.Filename = "Figure.1.f"
    th.Title = "Energy as a Function of $p_T$ for Spectator Truth-Tops"
    th.xData = ana.spec_top_kinematics["pt"]
    th.xMin = 0
    th.xMax = 800
    th.xBins = 160
    th.xStep = 100
    th.xTitle = "$p_T$ of Spectator Truth Top / 5 GeV"

    th.yData = ana.spec_top_kinematics["energy"]
    th.yMin = 0
    th.yMax = 800
    th.yBins = 160
    th.yStep = 100
    th.yTitle = "Energy of Spectator Truth Top / 5 GeV"
    th.SaveFigure()

def top_delta_r(ana):

    thr = TH1F()
    thr.Title = "Resonance Pairs"
    thr.xData = ana.deltaR["RR"]

    ths = TH1F()
    ths.Title = "Spectator Pairs"
    ths.xData = ana.deltaR["SS"]

    thx = TH1F()
    thx.Title = "Mixture (Resonance - Spectator)"
    thx.xData = ana.deltaR["RS"]

    tha = path(TH1F())
    tha.Histograms = [thx, thr, ths]
    tha.Title = "$\\Delta R$ of Truth-Top Pairs (All Possible Permutations)"
    tha.xTitle = "$\\Delta R$ Between Possible Pair Permutations"
    tha.yTitle = "Density (Arb.) / 0.05"
    tha.Density = True
    tha.xMin = 0
    tha.xMax = 6
    tha.xBins = 120
    tha.xStep = 0.5
    tha.Filename = "Figure.1.g"
    tha.SaveFigure()

def top_pair_mass(ana):
    thr = TH1F()
    thr.Title = "Resonance Pairs"
    thr.xData = ana.mass_combi["RR"]

    ths = TH1F()
    ths.Title = "Spectator Pairs"
    ths.xData = ana.mass_combi["SS"]

    thx = TH1F()
    thx.Title = "Mixture (Resonance - Spectator)"
    thx.xData = ana.mass_combi["RS"]

    tha = path(TH1F())
    tha.Histograms = [thx, thr, ths]
    tha.Title = "Invariant Mass of Truth-Top Pairs (All Possible Permutations)"
    tha.xTitle = "Invariant Mass of Possible Pair Permutations (GeV)"
    tha.yTitle = "Entries / 5 GeV"
    tha.Density = True
    tha.xMin = 200
    tha.xMax = 1400
    tha.xBins = 240
    tha.xStep = 100
    tha.Filename = "Figure.1.h"
    tha.SaveFigure()

    th = path(TH2F())
    th.Title = "Invariant Mass as a Function of $\\Delta R$ for Possible Top Pair Permutations"
    th.Filename = "Figure.1.i"

    th.xData = sum(ana.deltaR.values(), [])
    th.xMin = 0
    th.xMax = 4
    th.xBins = 100
    th.xStep = 0.25
    th.xTitle = "$\\Delta R$ Between Pair Permutations / 0.04"

    th.yData = sum(ana.mass_combi.values(), [])
    th.yMin = 200
    th.yMax = 1200
    th.yBins = 100
    th.yStep = 100
    th.yTitle = "Invariant Mass of Truth-Top Pair Permutation / 10 GeV"
    th.SaveFigure()

def TopKinematics(ana):
    top_pt(ana)
    top_energy(ana)
    top_phi(ana)
    top_eta(ana)
    top_pt_energy(ana)
    top_delta_r(ana)
    top_pair_mass(ana)


