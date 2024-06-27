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

def settings_th2f():
    settings = {
            "Style" : "ROOT",
            "OutputDirectory" : figure_path + "truth-event/figures/",
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

    sett = settings()
    th_m = TH1F(**sett)
    th_m.xLabels = {i : data[i] for i in ["0L", "1L", "2L", "3L", "4L"]}
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

    modes = ["0L", "1L", "2L", "3L", "4L", "2LOS", "2LSS"]
    figure_title = [
            "hadronic",
            "Single Lepton",
            "Di-Leptonic",
            "Tri-Leptonic",
            "Quad-Leptonic",
            "Opposite Sign",
            "Same Sign"
    ]
    figure_file = ["g", "h", "i", "j", "k", "l", "m"]

    for mode, fig, ext in zip(modes, figure_title, figure_file):
        sett = settings_th2f()
        th_xy = TH2F(**sett)
        th_xy.Title = "Missing Tranverse Momenta in Cartesian Coordinates for " + fig + " Decay Mode"
        th_xy.xTitle = "x-Direction of Missing Tranverse Momenta (GeV)"
        th_xy.xData = ana.met_cartesian["met_x"][mode]
        th_xy.xMin = -1000
        th_xy.xMax =  1000
        th_xy.xBins = 500
        th_xy.xOverFlow = True

        th_xy.yTitle = "y-Direction of Missing Tranverse Momenta (GeV)"
        th_xy.yData = ana.met_cartesian["met_y"][mode]
        th_xy.yMin = -1000
        th_xy.yMax =  1000
        th_xy.yBins = 500
        th_xy.yOverFlow = True

        th_xy.Filename = "Figure.6." + ext
        th_xy.Color = "tab20c"
        th_xy.SaveFigure()


    figure_file = ["n", "o", "p", "q", "r", "s", "t"]
    for mode, fig, ext in zip(modes, figure_title, figure_file):
        sett = settings_th2f()
        th_xy = TH2F(**sett)
        th_xy.Title = "$\\Delta$ Missing Tranverse Momenta in Cartesian Coordinates for " + fig + " Decay Mode"
        th_xy.xTitle = "x-Direction of $\\Delta$-Missing Tranverse Momenta (GeV)"
        th_xy.xData = ana.met_cartesian["delta_met_x"][mode]
        th_xy.xMin = -1000
        th_xy.xMax =  1000
        th_xy.xBins = 500
        th_xy.xOverFlow = True

        th_xy.yTitle = "y-Direction of $\\Delta$-Missing Tranverse Momenta (GeV)"
        th_xy.yData = ana.met_cartesian["delta_met_y"][mode]
        th_xy.yMin = -1000
        th_xy.yMax =  1000
        th_xy.yBins = 500
        th_xy.yOverFlow = True

        th_xy.Filename = "Figure.6." + ext
        th_xy.Color = "tab20c"
        th_xy.SaveFigure()


def njets_ntruthjets(ana):
    max_ = max(ana.njet_data["truth-jets"]+ana.njet_data["jets"])

    sett = settings_th2f()
    th_xy = TH2F(**sett)
    th_xy.Title = "Number of Truth Jets against Detector Jets"
    th_xy.xTitle = "Number of Truth Jets"
    th_xy.xData = ana.njet_data["truth-jets"]
    th_xy.xMin = 0
    th_xy.xMax = max_+1
    th_xy.xStep = 1

    th_xy.yTitle = "Number of Detector Jets"
    th_xy.yData = ana.njet_data["jets"]
    th_xy.yMin = 0
    th_xy.yMax = max_+1
    th_xy.yStep = 1

    th_xy.Filename = "Figure.6.u"
    th_xy.SaveFigure()


    max_ = max(ana.njet_data["b-truth-jets"]+ana.njet_data["b-jets"])

    sett = settings_th2f()
    th_xy = TH2F(**sett)
    th_xy.Title = "Number of Truth b-Jets against Detector b-Jets"
    th_xy.xTitle = "Number of Truth b-Jets"
    th_xy.xData = ana.njet_data["b-truth-jets"]
    th_xy.xMin = 0
    th_xy.xMax = max_+1
    th_xy.xStep = 1

    th_xy.yTitle = "Number of Detector b-Jets"
    th_xy.yData = ana.njet_data["b-jets"]
    th_xy.yMin = 0
    th_xy.yMax = max_+1
    th_xy.yStep = 1

    th_xy.Filename = "Figure.6.v"
    th_xy.SaveFigure()

def nleptons(ana):
    max_ = max(ana.nleptons["truth"]+ana.nleptons["detector"])

    sett = settings_th2f()
    th_xy = TH2F(**sett)
    th_xy.Title = "Number of Truth Leptons (from Children) against Detector Leptons"
    th_xy.xTitle = "Number of Truth Leptons"
    th_xy.xData = ana.nleptons["truth"]
    th_xy.xMin = 0
    th_xy.xMax = max_+1
    th_xy.xStep = 1

    th_xy.yTitle = "Number of Detector Leptons"
    th_xy.yData = ana.nleptons["detector"]
    th_xy.yMin = 0
    th_xy.yMax = max_+1
    th_xy.yStep = 1

    th_xy.Filename = "Figure.6.w"
    th_xy.SaveFigure()


def TruthEvent(ana):
    missing_met_difference(ana)
    num_leptons(ana)
    missing_met_mode(ana)
    njets_ntruthjets(ana)
    nleptons(ana)
