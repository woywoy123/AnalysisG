from AnalysisG.core.plotting import TH1F, TH2F

global figure_path
global mass_point

def path(hist):
    hist.OutputDirectory = figure_path + "/top-kinematics/" + mass_point
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
    tha.Title = "Transverse Momenta of Truth Tops (" + mass_point +")"
    tha.xTitle = "Transverse Momenta (GeV)"
    tha.yTitle = "Entries (Arb.)"
    tha.xMin = 0
    tha.xMax = 1500
    tha.xBins = 100
    tha.xStep = 150
    tha.Filename = "Figure.3.a"
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
    tha.Title = "Energy of Truth Tops (" + mass_point + ")"
    tha.xTitle = "Energy (GeV)"
    tha.yTitle = "Entries"
    tha.xMin = 0
    tha.xMax = 1500
    tha.xBins = 100
    tha.xStep = 150
    tha.Filename = "Figure.3.b"
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
    tha.Title = "Azimuthal Angle of Truth Tops (" + mass_point + ")"
    tha.xTitle = "Azimuthal Angle ($\\phi$)"
    tha.yTitle = "Entries (arb.)"
    tha.xMin = -3
    tha.xMax = 3
    tha.xBins = 30
    tha.xStep = 0.5
    tha.Filename = "Figure.3.c"
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
    tha.Title = "Pseudorapidity of Truth Tops (" + mass_point + ")"
    tha.xTitle = "Pseudorapidity ($\\eta$)"
    tha.yTitle = "Entries (arb.)"
    tha.xMin = -6
    tha.xMax = 6
    tha.xBins = 60
    tha.xStep = 1
    tha.Filename = "Figure.3.d"
    tha.SaveFigure()

def top_pt_energy(ana):

    th = path(TH2F())
    th.Filename = "Figure.3.e"
    th.Title = "Energy as a Function of Transverse Momenta for Resonant Truth-Tops (" + mass_point + ")"

    th.xData = ana.res_top_kinematics["pt"]
    th.xMin = 0
    th.xMax = 1500
    th.xBins = 1000
    th.xStep = 150
    th.xTitle = "Transverse Momenta (GeV)"

    th.yData = ana.res_top_kinematics["energy"]
    th.yMin = 0
    th.yMax = 1500
    th.yBins = 1000
    th.yStep = 150
    th.yTitle = "Energy (GeV)"
    th.SaveFigure()

    th = path(TH2F())
    th.Filename = "Figure.3.f"
    th.Title = "Energy as a Function of Transverse Momenta for Spectator Truth-Tops (" + mass_point + ")"
    th.xData = ana.spec_top_kinematics["pt"]
    th.xMin = 0
    th.xMax = 1500
    th.xBins = 1000
    th.xStep = 150
    th.xTitle = "Transverse Momenta (GeV)"

    th.yData = ana.spec_top_kinematics["energy"]
    th.yMin = 0
    th.yMax = 1500
    th.yBins = 1000
    th.yStep = 150
    th.yTitle = "Energy (GeV)"
    th.SaveFigure()

def top_delta_r(ana):

    thr = TH1F()
    thr.Title = "Resonance"
    thr.xData = ana.deltaR["RR"]

    ths = TH1F()
    ths.Title = "Spectator"
    ths.xData = ana.deltaR["SS"]

    thx = TH1F()
    thx.Title = "Mixture"
    thx.xData = ana.deltaR["RS"]

    tha = path(TH1F())
    tha.Histograms = [thx, thr, ths]
    tha.Title = "$\\Delta$ R of Truth-Top Pairs (All permutations) (" + mass_point + ")"
    tha.xTitle = "$\\Delta$ R (arb.)"
    tha.yTitle = "Entries (arb.)"
    tha.xMin = 0
    tha.xMax = 5
    tha.xBins = 100
    tha.xStep = 0.5
    tha.Filename = "Figure.3.g"
    tha.SaveFigure()

def top_pair_mass(ana):
    thr = TH1F()
    thr.Title = "Resonance"
    thr.xData = ana.mass_combi["RR"]

    ths = TH1F()
    ths.Title = "Spectator"
    ths.xData = ana.mass_combi["SS"]

    thx = TH1F()
    thx.Title = "Mixture"
    thx.xData = ana.mass_combi["RS"]

    tha = path(TH1F())
    tha.Histograms = [thx, thr, ths]
    tha.Title = "Invariant Mass of Truth-Top Pairs (All permutations) (" + mass_point + ")"
    tha.xTitle = "Invariant Mass (GeV)"
    tha.yTitle = "Entries (arb.)"
    tha.xMin = 0
    tha.xMax = 3000
    tha.xBins = 1000
    tha.xStep = 300
    tha.Filename = "Figure.3.h"
    tha.SaveFigure()

    th = path(TH2F())
    th.Title = "Invariant Mass as a Function of $\\Delta$R for Truth-Top Pairs (" + mass_point + ")"
    th.Filename = "Figure.3.j"

    th.xData = sum(ana.deltaR.values(), [])
    th.xMin = 0
    th.xMax = 5
    th.xBins = 1000
    th.xStep = 0.5
    th.xTitle = "$\\Delta$R (arb.)"

    th.yData = sum(ana.mass_combi.values(), [])
    th.yMin = 0
    th.yMax = 1500
    th.yBins = 1000
    th.yStep = 150
    th.yTitle = "Invariant Mass (GeV)"
    th.SaveFigure()

def TopKinematics(ana):
    top_pt(ana)
    top_energy(ana)
    top_phi(ana)
    top_eta(ana)
    top_pt_energy(ana)
    top_delta_r(ana)
    top_pair_mass(ana)


