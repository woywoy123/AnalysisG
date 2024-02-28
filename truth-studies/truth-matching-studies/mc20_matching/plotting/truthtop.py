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
    data_top = ana.tops_mass["all"]
    sett = settings()
    th = TH1F(**sett)
    th1 = TH1F()
    th1.xData = data_top
    th1.Title = "truth-top"
    th.Title = "Invariant Mass Distribution of Post-FSR Truth Tops"
    th.Histogram = th1
    th.xMin = 171
    th.xMax = 173
    th.xStep = 1
    th.xBins = 1000
    th.yTitle = "Entries"
    th.xTitle = "Invariant (Truth Top) Mass (GeV)"
    th.Filename = "Figure.1.a"
    th.SaveFigure()

def plot_dr(ana):
    data = ana.mtt_dr["dr"]
    sett = settings()
    th = TH1F(**sett)
    th1 = TH1F()
    th1.xData = data
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
    data = ana.mtt_dr["mass"]
    sett = settings()
    th = TH1F(**sett)
    th1 = TH1F()
    th1.xData = data
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




def TruthTops(ana):
    plot_top_masses(ana)
    plot_dr(ana)
    plot_mtt(ana)
    plot_top_kinematics(ana)
    plot_top_attributes(ana)
    plot_ntops(ana)
