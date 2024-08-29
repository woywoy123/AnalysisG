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
    tha.Title = "Transverse Momenta of Truth Tops (Resonance at " + mass_point.split(".")[1] +" GeV)"
    tha.xTitle = "Transverse Momenta (GeV)"
    tha.yTitle = "Entries / $15$ GeV"
    tha.xMin = 0
    tha.Density = True
    tha.xMax = 1500
    tha.xBins = 100
    tha.xStep = 150
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
    tha.Title = "Energy of Truth Tops (Resonance at " + mass_point.split(".")[1] +" GeV)"
    tha.xTitle = "Energy (GeV)"
    tha.yTitle = "Entries / $15$ GeV"
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
    tha.Title = "Azimuthal Angle of Truth Tops (Resonance at " + mass_point.split(".")[1] +" GeV)"
    tha.xTitle = "Azimuthal Angle ($\\phi$)"
    tha.yTitle = "Entries / 0.1"
    tha.xMin = -3.5
    tha.Density = True
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
    tha.Title = "Pseudorapidity of Truth Tops (Resonance at " + mass_point.split(".")[1] +" GeV)"
    tha.xTitle = "Pseudorapidity ($\\eta$)"
    tha.yTitle = "Entries / 0.1"
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
    th.Title = "Energy as a Function of Transverse Momenta for Resonant Truth-Tops \n (Resonance at " + mass_point.split(".")[1] +" GeV)"

    th.xData = ana.res_top_kinematics["pt"]
    th.xMin = 0
    th.xMax = 1000
    th.xBins = 200
    th.xStep = 200
    th.xTitle = "Transverse Momenta of Resonant Truth Top / 5 GeV"

    th.yData = ana.res_top_kinematics["energy"]
    th.yMin = 0
    th.yMax = 1000
    th.yBins = 200
    th.yStep = 200
    th.yTitle = "Energy of Resonant Truth Top / 5 GeV"
    th.SaveFigure()

    th = path(TH2F())
    th.Filename = "Figure.1.f"
    th.Title = "Energy as a Function of Transverse Momenta for Spectator Truth-Tops \n (Resonance at " + mass_point.split(".")[1] +" GeV)"
    th.xData = ana.spec_top_kinematics["pt"]
    th.xMin = 0
    th.xMax = 1000
    th.xBins = 200
    th.xStep = 200
    th.xTitle = "Transverse Momenta of Spectator Truth Top / 5 GeV"

    th.yData = ana.spec_top_kinematics["energy"]
    th.yMin = 0
    th.yMax = 1000
    th.yBins = 200
    th.yStep = 200
    th.yTitle = "Energy of Spectator Truth Top / 5 GeV"
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
    tha.Title = "$\\Delta$ R of Truth-Top Pairs (All permutations) \n (Resonance at " + mass_point.split(".")[1] +" GeV)"
    tha.xTitle = "$\\Delta$ R between All Possible Truth-Top Pair Permutations"
    tha.yTitle = "Entries / 0.05"
    tha.xMin = 0
    tha.xMax = 6
    tha.Density = True
    tha.xBins = 120
    tha.xStep = 0.5
    tha.Filename = "Figure.1.g"
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
    tha.Title = "Invariant Mass of Truth-Top Pairs (All permutations) \n (Resonance at " + mass_point.split(".")[1] +" GeV)"
    tha.xTitle = "Invariant Mass of Permutating All Possible Truth-Top Pairs (GeV)"
    tha.yTitle = "Entries / 5 GeV"
    tha.Density = True
    tha.xMin = 0
    tha.xMax = 2000
    tha.xBins = 400
    tha.xStep = 200
    tha.Filename = "Figure.1.h"
    tha.SaveFigure()

    th = path(TH2F())
    th.Title = "Invariant Mass as a Function of $\\Delta$R for All Truth-Top Pair Permutations \n (Resonance at " + mass_point.split(".")[1] +" GeV)"
    th.Filename = "Figure.1.i"

    th.xData = sum(ana.deltaR.values(), [])
    th.xMin = 0
    th.xMax = 4
    th.xBins = 400
    th.xStep = 0.5
    th.xTitle = "$\\Delta$R Between Truth-Top Pair Permutation / 0.01"

    th.yData = sum(ana.mass_combi.values(), [])
    th.yMin = 0
    th.yMax = 1200
    th.yBins = 400
    th.yStep = 200
    th.yTitle = "Invariant Mass of Truth-Top Pair Permutation / 3 GeV"
    th.SaveFigure()

def TopKinematics(ana):
    pass
#    top_pt(ana)
#    top_energy(ana)
#    top_phi(ana)
#    top_eta(ana)
#    top_pt_energy(ana)
#    top_delta_r(ana)
#    top_pair_mass(ana)


