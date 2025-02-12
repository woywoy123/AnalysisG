from AnalysisG.core.plotting import TH1F, TH2F
from AnalysisG.core import *
from .helper_figs import *
import math

def missing_energy(data):
    missing = GetMissingEnergy(data)
    th = path(TH2F(), "missing_energy_delta")
    th.Title = "Missing Energy Differential Between Observed and Children/Neutrino"
    th.xData = missing["children"]
    th.yData = missing["neutrino"]
    th.xTitle = "Observed - Children (GeV)"
    th.yTitle = "Observed - Neutrino (GeV)"
    th.Color = "BrBG"

    th.xStep = 100
    th.xBins = 200
    th.xMin = -500
    th.xMax =  500

    th.yStep = 100
    th.yBins = 200
    th.yMin = -500
    th.yMax =  500
    th.SaveFigure()

    th = path(TH1F(), "fraction_met_neutrino")
    th.Title = "Fractional Energy Contributions of Truth Neutrinos"
    th.yTitle = "Events / 0.01"
    th.xData = [abs((i - j)/ j) for i, j in zip(missing["neutrino_met"], missing["observed_met"])]
    th.xTitle = "Fractional Missing Energy of Neutrinos ($(E_{\\nu} - E_{obs.}) / E_{obs.}$)"
    th.Color = "blue"
    th.xStep = 0.1
    th.xBins = 200
    th.xMin = 0
    th.xMax =  2.0
    th.SaveFigure()

def double_neutrino(data):
    masses = GetMasses(data)

    th1 = TH1F()
    th1.Color = "red"
    th1.Title = "Truth"
    th1.xData = masses["top-mass-child"]

    th2 = TH1F()
    th2.Color = "blue"
    th2.Title = "Observed Missing Energy"
    th2.xData = masses["top-mass-cobs"]

    th3 = TH1F()
    th3.Color = "green"
    th3.Title = "Missing Energy (Truth Neutrinos)"
    th3.xData = masses["top-mass-cmet"]

    th = path(TH1F(), "top-masses-cmb")
    th.Title = "Reconstructed Invariant Mass of Top Quarks Using \n Different Missing Energy"
    th.Histograms = [th1, th2, th3]
    th.xTitle = "Invariant Top Mass (GeV)"
    th.yTitle = "Reconstructed Tops"
    th.xMin = 100
    th.xStep = 20
    th.xMax = 300
    th.xBins = 200
    th.SaveFigure()

    th1 = TH1F()
    th1.Color = "red"
    th1.Title = "Truth"
    th1.xData = masses["top-mass-child"]

    th2 = TH1F()
    th2.Color = "blue"
    th2.Title = "Observed Missing Energy"
    th2.xData = masses["top-mass-robs"]

    th3 = TH1F()
    th3.Color = "green"
    th3.Title = "Missing Energy (Truth Neutrinos)"
    th3.xData = masses["top-mass-rmet"]

    th = path(TH1F(), "top-masses-ref")
    th.Title = "Reconstructed Invariant Mass of Top Quarks Using \n Different Missing Energy - Reference"
    th.Histograms = [th1, th2, th3]
    th.xTitle = "Invariant Top Mass (GeV)"
    th.yTitle = "Reconstructed Tops"
    th.xMin = 100
    th.xStep = 20
    th.xMax = 300
    th.xBins = 200
    th.SaveFigure()


