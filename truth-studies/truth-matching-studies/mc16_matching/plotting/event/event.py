from AnalysisG.Plotting import TH1F, TH2F
global figure_path

def settings():
    settings = {
            "Style" : "ROOT",
            "OutputDirectory" : figure_path + "truth-event/figures/",
            "Histograms" : [],
            "Histogram" : None,
            "LegendLoc" : "upper right"
    }
    return settings

def missing_met_difference(ana):
    data = ana.met_data

    th_dnu = TH1F()
    th_dnu.Title = "$\\Delta M_{ET}$ (Truth Neutrinos)"
    th_dnu.xData = sum([data["delta_met_nus"][x] for x in data["delta_met_nus"]], [])

    th_dch = TH1F()
    th_dch.Title = "$\\Delta M_{ET}$ (All Children)"
    th_dch.xData = sum([data["delta_met_children"][x] for x in data["delta_met_children"]], [])

    sett = settings()
    th_m = TH1F(**sett)
    th_m.Histograms = [th_dnu, th_dch]
    th_m.Title = "Missing Transverse Energy Differential between Measurement and Truth Neutrino/Children"
    th_m.xTitle = "$\\Delta M_{ET}$ (GeV)"
    th_m.yTitle = "Entries <unit>"
    th_m.xBins = 400
    th_m.xStep = 100
    th_m.xMin = 0
    th_m.xMax = 1000
    th_m.Filename = "Figure.6.a"
    th_m.SaveFigure()

    hists = []
    for i in data["delta_met_nus"]:
        th_dnu = TH1F()
        th_dnu.Title = "Number of Nu " + str(i)
        th_dnu.xData = data["delta_met_nus"][i]
        hists.append(th_dnu)

    sett = settings()
    th_m = TH1F(**sett)
    th_m.Histograms = hists
    th_m.Title = "Missing Transverse Energy Differential between Measurement and Truth Neutrino"
    th_m.xTitle = "$\\Delta M_{ET}$ (GeV)"
    th_m.yTitle = "Entries <unit>"
    th_m.xBins = 400
    th_m.xStep = 100
    th_m.xMin = 0
    th_m.xMax = 1000
    th_m.Stack = True
    th_m.Filename = "Figure.6.b"
    th_m.SaveFigure()

    hists = []
    for i in data["delta_met_children"]:
        th_dnu = TH1F()
        th_dnu.Title = "Number of Nu " + str(i)
        th_dnu.xData = data["delta_met_children"][i]
        hists.append(th_dnu)

    sett = settings()
    th_m = TH1F(**sett)
    th_m.Histograms = hists
    th_m.Title = "Missing Transverse Energy Differential between Measurement and Truth Children"
    th_m.xTitle = "$\\Delta M_{ET}$ (GeV)"
    th_m.yTitle = "Entries <unit>"
    th_m.xBins = 400
    th_m.xStep = 100
    th_m.xMin = 0
    th_m.xMax = 1000
    th_m.Stack = True
    th_m.Filename = "Figure.6.c"
    th_m.SaveFigure()


    th_meas = TH1F()
    th_meas.Title = "Measured"
    th_meas.xData = sum([data["met"][i] for i in data["met"] if i > 0], [])

    th_dnu = TH1F()
    th_dnu.Title = "Truth-Neutrinos"
    th_dnu.xData = sum([data["truth-nus"][i] for i in data["truth-nus"] if i > 0], [])

    sett = settings()
    th_m = TH1F(**sett)
    th_m.Histograms = [th_meas, th_dnu]
    th_m.Title = "Original Measured Missing Transverse compared to Truth-Neutrinos"
    th_m.xTitle = "$M_{ET}$ (GeV)"
    th_m.yTitle = "Entries <unit>"
    th_m.xBins = 400
    th_m.xStep = 100
    th_m.xMin = 0
    th_m.xMax = 1000
    th_m.Filename = "Figure.6.d"
    th_m.SaveFigure()

def num_leptons(ana):

    data = ana.num_leptons
    modes = {i : data[i] for i in ["0L", "1L", "2L", "3L", "4L"]}

    sett = settings()
    th_m = TH1F(**sett)
    th_m.xLabels = modes
    th_m.Title = "Fraction of Leptonic Decay Modes of Top-Quark Pairs"
    th_m.yTitle = "Fraction"
    th_m.xTitle = "Number of Leptons"
    th_m.Filename = "Figure.6.e"
    th_m.Normalize = True
    th_m.SaveFigure()

    sett = settings()
    th_m = TH1F(**sett)
    th_m.xLabels = {j : data[i] for i, j in zip(["2LOS", "2LSS"], ["Opposite Sign", "Same Sign"])}
    th_m.Title = "Fraction of Dileptonic Top-Quark Pairs Decaying into Same/Opposite Sign Leptons"
    th_m.yTitle = "Fraction"
    th_m.xTitle = "Dilepton Sign Comparison"
    th_m.Filename = "Figure.6.f"
    th_m.Normalize = True
    th_m.SaveFigure()


def missing_met_mode(ana):
    pass



def TruthEvent(ana):
    missing_met_difference(ana)
    num_leptons(ana)
    missing_met_mode(ana)
