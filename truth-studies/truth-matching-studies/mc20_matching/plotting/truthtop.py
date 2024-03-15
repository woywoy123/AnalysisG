from AnalysisG.Plotting import TH1F, TH2F

def settings():
    settings = {
            "Style" : "ROOT",
            "OutputDirectory" : "./plt_plots/truth-tops",
            "Histograms" : [],
            "Histogram" : None,
            "LegendLoc" : "upper right"
    }
    return {i : x for i, x in settings.items()}


def plot_top_masses(ana):

    tru_top = TH1F()
    tru_top.Title = "Truth-Top"
    tru_top.xData = [] #ana.truth_top

    tru_ch = TH1F()
    tru_ch.Title = "Truth-Children"
    tru_ch.xData = [] #ana.truth_children["all"]

    tru_tj = TH1F()
    tru_tj.Title = "Truth-Jets (Truth Leptons and Neutrinos)"
    tru_tj.xData = [] #ana.truth_physics["hadronic"]

    tru_j = TH1F()
    tru_j.Title = "Jets (Truth Leptons and Neutrinos)"
    tru_j.xData = [] # ana.jets_truth_leps["hadronic"]

    tru_jl = TH1F()
    tru_jl.Title = "Jets Leptons (Truth Neutrinos)"
    tru_jl.xData = ana.detector["leptonic"]

    sett = settings()
    all_t = TH1F(**sett)
    all_t.Histograms = [tru_jl, tru_j, tru_tj, tru_ch, tru_top]
    all_t.Title = "Top Truth Matching Scheme for Varying Level of Monte Carlo Truth"
    all_t.xTitle = "Invariant Mass (GeV)"
    all_t.yTitle = "Entries <unit>"
    all_t.xMin = 0
    all_t.xMax = 400
    all_t.xBins = 400
    all_t.xStep = 20
    all_t.yScaling = 10
    all_t.xScaling = 20
    all_t.FontSize = 20
    all_t.LabelSize = 20
    all_t.OverFlow = False
    all_t.yLogarithmic = False
    all_t.Filename = "Figure.1.a"
    all_t.SaveFigure()
    exit()

def plot_dr(ana):
    sett = settings()
    th = TH1F(**sett)
    th1 = TH1F()
    th1.xData = ana.mtt_dr["dr"]
    th1.Title = "$\\Delta R$"
    th.Title = "$\\Delta R$ of Post-FSR Truth Top pairs"
    th.Histogram = th1
    th.xMin = 0
    th.xMax = 6
    th.xBins = 100
    th.xStep = 0.5
    th.yTitle = "Entries"
    th.xTitle = "$\\Delta$ R (arb. units)"
    th.Filename = "Figure.1.b"
    th.SaveFigure()


def plot_mtt(ana):
    sett = settings()
    th = TH1F(**sett)
    th1 = TH1F()
    th1.xData = ana.mtt_dr["mass"]
    th1.Title = "Pairs"
    th.Title = "Invariant Mass Distribution of Post-FSR Truth Top Pairs (bruteforced)"
    th.Histogram = th1
    th.xMin = 0
    th.xMax = 2000
    th.xBins = 1000
    th.xStep = 200
    th.yTitle = "Entries"
    th.xTitle = "Invariant Mass of Top Pairs (GeV)"
    th.Filename = "Figure.1.c"
    th.SaveFigure()

def plot_top_kinematics(ana):

    data = ana.tops_kinematics
    sett = settings()
    th = TH1F(**sett)
    th1 = TH1F()
    th1.xData = data["pt"]
    th1.Title = "top-pt"
    th.Title = "Transverse Momentum of Post-FSR Truth Tops"
    th.Histogram = th1
    th.xMin = 0
    th.xMax = 1000
    th.xBins = 400
    th.xStep = 100
    th.yTitle = "Entries"
    th.xTitle = "Transverse Momenta of individual Tops (GeV)"
    th.Filename = "Figure.1.d"
    th.SaveFigure()

    th = TH1F(**sett)
    th1 = TH1F()
    th1.xData = data["eta"]
    th1.Title = "top-eta"
    th.Title = "The $\\eta$ Distribution of individual Tops"
    th.Histogram = th1
    th.xMin = -6
    th.xMax = 6
    th.xBins = 400
    th.xStep = 1
    th.yTitle = "Entries"
    th.xTitle = "$\\eta$ (arb. units)"
    th.Filename = "Figure.1.e"
    th.SaveFigure()

    th = TH1F(**sett)
    th1 = TH1F()
    th1.xData = data["phi"]
    th1.Title = "top-phi"
    th.Title = "The Azimuthal ($\\phi$) angular Distribution of individual Tops"
    th.Histogram = th1
    th.xMin = -3.5
    th.xMax = 3.5
    th.xStep = 1
    th.xBins = 400
    th.yTitle = "Entries"
    th.xTitle = "$\\phi$ (radians)"
    th.Filename = "Figure.1.f"
    th.SaveFigure()

    th = TH1F(**sett)
    th1 = TH1F()
    th1.xData = data["energy"]
    th1.Title = "top-energy"
    th.Title = "Energy Distribution of Post-FSR Truth Tops"
    th.Histogram = th1
    th.xMin = 150
    th.xMax = 1500
    th.xBins = 500
    th.xStep = 150
    th.yTitle = "Entries"
    th.xTitle = "Energy of individual Tops (GeV)"
    th.Filename = "Figure.1.g"
    th.SaveFigure()

def plot_top_attributes(ana):

    data = ana.tops_attributes
    sett = settings()
    th = TH1F(**sett)
    th1 = TH1F()
    th1.xData = data["charge"]
    th1.Title = "top-charge"
    th.Title = "Electric charge of individual Tops"
    th.Histogram = th1
    th.xMin = -1
    th.xMax = 1
    th.xBins = 9
    th.xStep = 1
    th.yTitle = "Entries"
    th.xTitle = "Top charge"
    th.Filename = "Figure.1.h"
    th.SaveFigure()

    th = TH1F(**sett)
    th1 = TH1F()
    th1.xData = data["barcode"]
    th1.Title = "barcode"
    th.Title = "Post-FSR Top barcodes"
    th.Histogram = th1
    th.xMin = 0
    th.xMax = 1000
    th.xBins = 1000
    th.xStep = 100
    th.yTitle = "Entries"
    th.xTitle = "Barcodes"
    th.Filename = "Figure.1.i"
    th.SaveFigure()

    th = TH1F(**sett)
    th1 = TH1F()
    th1.xData = data["status"]
    th1.Title = "status"
    th.Title = "Post-FSR Top status codes"
    th.Histogram = th1
    th.xMin = 0
    th.xMax = 100
    th.xBins = 100
    th.xStep = 10
    th.yTitle = "Entries"
    th.xTitle = "Status Code"
    th.Filename = "Figure.1.j"
    th.SaveFigure()

def plot_ntops(ana):
    sett = settings()
    th = TH1F(**sett)
    th.xData = ana.event_ntops
    th.Title = "Number of Truth Tops found in Events"
    th.xTitle = "n-tops"
    th.xMin = 0
    th.xMax = 8
    th.xBins = 8
    th.xStep = 1
    th.xBinCentering = True
    th.yTitle = "Entries"
    th.Filename = "Figure.1.k"
    th.SaveFigure()


def TruthTops(ana):
    plot_top_masses(ana)
    plot_dr(ana)
    plot_mtt(ana)
    plot_top_kinematics(ana)
    plot_top_attributes(ana)
    plot_ntops(ana)
