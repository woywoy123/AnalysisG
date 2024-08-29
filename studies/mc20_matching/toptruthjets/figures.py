from AnalysisG.core.plotting import TH1F, TH2F

global figure_path
global mass_point

def path(hist):
    hist.OutputDirectory = figure_path + "/top-truthjets/" + mass_point
    return hist

def top_mass_truthjets(ana):

    hists = []
    for x in ["leptonic", "hadronic"]:
        th = TH1F()
        th.Title = x
        th.xData = ana.top_mass[x][""][""]
        hists += [th]

    th_ = path(TH1F())
    th_.Histograms = hists
    th_.Title = "Invariant Top-Quark Mass from Matched Truth Jets (and Truth Leptons/Neutrinos)"
    th_.xTitle = "Invariant Mass (GeV)"
    th_.yTitle = "Entries (Arb.)"
    th_.Filename = "Figure.5.a"
    th_.xBins = 500
    th_.xMin  = 0
    th_.xMax  = 500
    th_.xStep = 50
    th_.SaveFigure()


    for x, name in zip(["leptonic", "hadronic"], ["b", "c"]):
        hists = []
        for nj in ana.top_mass["ntruthjets"][x]:
            th = TH1F()
            th.Title = str(nj) + "-TruthJets"
            th.xData = ana.top_mass["ntruthjets"][x][nj]
            hists += [th]

        th_ = path(TH1F())
        th_.Histograms = hists
        th_.Title = "Invariant Top-Quark Mass for " + x + " Decays for n-TruthJet Contributions"
        th_.xTitle = "Invariant Mass (GeV)"
        th_.yTitle = "Entries (Arb.)"
        th_.Filename = "Figure.5." + name
        th_.xBins = 500
        th_.xMin  = 0
        th_.xMax  = 500
        th_.xStep = 50
        th_.Stacked = True
        th_.SaveFigure()


    for x, name in zip(["leptonic", "hadronic"], ["d", "e"]):
        hists = []
        for nj in ana.top_mass["merged_tops"][x]:
            th = TH1F()
            th.Title = str(nj) + "-Tops"
            th.xData = ana.top_mass["merged_tops"][x][nj]
            hists += [th]

        th_ = path(TH1F())
        th_.Histograms = hists
        th_.Title = "Invariant Top-Quark Mass for " + x + " Decays with \n n-Top Contributions in Truth Jets"
        th_.xTitle = "Invariant Mass (GeV)"
        th_.yTitle = "Entries (Arb.)"
        th_.Filename = "Figure.5." + name
        th_.xBins = 500
        th_.xMin  = 0
        th_.xMax  = 500
        th_.xStep = 50
        th_.Stacked = True
        th_.SaveFigure()

def top_truthjet_cluster(ana):

    hists = []
    maps = ["background", "resonant-leptonic", "resonant-hadronic", "spectator-leptonic", "spectator-hadronic"]
    for x in maps:
        th = TH1F()
        th.Title = x
        th.xData = ana.truthjet_top[x]["dr"]
        hists += [th]

    th_ = path(TH1F())
    th_.Histograms = hists
    th_.Title = "$\\Delta$R Between Truth-Jets matched to Mutal Top \n compared to Background (Other Truth-Jets)"
    th_.xTitle = "$\\Delta$R (Arb.)"
    th_.yTitle = "Entries"
    th_.Filename = "Figure.5.f"
    th_.xBins = 500
    th_.xMin  = 0
    th_.xMax  = 6
    th_.xStep = 1
    th_.yLogarithmic = True
    th_.SaveFigure()

    names = ["Resonance (Leptonic)", "Resonance (Hadronic)", "Spectator (Leptonic)", "Spectator (Hadronic)"]
    fig_ = ["g", "h", "i", "j"]
    map_m = maps[1:]

    for n, j, f in zip(names, map_m, fig_):
        th2 = path(TH2F())
        th2.Title = "$\\Delta$R Between Truth-Jets from " + n + " Top \n as a Function of the Top's Energy"
        th2.xData = ana.truthjet_top[j]["energy"]
        th2.yData = ana.truthjet_top[j]["dr"]

        th2.xTitle = "Originating Top Energy (GeV)"
        th2.yTitle = "$\\Delta R$ (Arb.)"

        th2.xBins = 500
        th2.yBins = 500

        th2.xMin = 0
        th2.xMax = 1500

        th2.yMin = 0
        th2.yMax = 6

        th2.Filename = "Figure.5." + f
        th2.SaveFigure()

    fig_ = ["k", "l", "m", "n"]
    for n, j, f in zip(names, map_m, fig_):
        th2 = path(TH2F())
        th2.Title = "$\\Delta$R Between Truth-Jets from " + n + " Top \n as a Function of the Top's Transverse Momenta"
        th2.xData = ana.truthjet_top[j]["energy"]
        th2.yData = ana.truthjet_top[j]["dr"]

        th2.xTitle = "Originating Top Transverse Momenta (GeV)"
        th2.yTitle = "$\\Delta R$ (Arb.)"

        th2.xBins = 500
        th2.yBins = 500

        th2.xMin = 0
        th2.xMax = 1500

        th2.yMin = 0
        th2.yMax = 6

        th2.Filename = "Figure.5." + f
        th2.SaveFigure()

def truthjet_partons(ana):
    data = ana.truthjet_partons
    maps = ["resonant-leptonic", "resonant-hadronic", "spectator-leptonic", "spectator-hadronic", "background"]

    symbolic = {}
    energy = {}
    transverse = {}
    for x in range(len(maps)):
        mode = maps[x]
        for k in data[mode]:
            if k not in symbolic:
                symbolic[k] = []
                energy[k] = []
                transverse[k] = []
            symbolic[k] += data[mode][k]["dr"]
            energy[k] += data[mode][k]["parton-energy"]
            transverse[k] += data[mode][k]["parton-pt"]

    hists = []
    for x in symbolic:
        th_ = TH1F()
        th_.xData = symbolic[x]
        th_.Title = x
        hists.append(th_)

    th_p = path(TH1F())
    th_p.Histograms = hists
    th_p.Title = "$\\Delta$R Between the Truth-Jet Axis and Ghost Matched Partons"
    th_p.xTitle = "$\\Delta$R (Arb.)"
    th_p.yTitle = "Entries (Arb.)"
    th_p.Filename = "Figure.5.o"
    th_p.xBins = 500
    th_p.xMin  = 0
    th_p.xMax  = 0.6
    th_p.xStep = 0.1
    th_p.yLogarithmic = True
    th_p.SaveFigure()

    th2 = path(TH2F())
    th2.Title = "$\\Delta$R Between Truth-Jet Axis and Ghost Matched Partons \n as a function of Ghost Parton Energy"
    th2.xData = sum([x for x in energy.values()], [])
    th2.yData = sum([x for x in symbolic.values()], [])
    th2.xTitle = "Ghost Matched Parton Energy (GeV)"
    th2.yTitle = "$\\Delta R$ between Truth-Jet Axis and Ghost Matched Partons (Arb.)"

    th2.xBins = 500
    th2.yBins = 500

    th2.xMin = 0
    th2.xMax = 600

    th2.yMin = 0
    th2.yMax = 0.6

    th2.Filename = "Figure.5.p"
    th2.SaveFigure()

    th2 = path(TH2F())
    th2.Title = "$\\Delta$R Between Truth-Jet Axis and Ghost Matched Partons \n as a function of Ghost Parton Transverse Momenta"
    th2.xData = sum([x for x in transverse.values()], [])
    th2.yData = sum([x for x in symbolic.values()], [])
    th2.xTitle = "Ghost Matched Transverse Momenta (GeV)"
    th2.yTitle = "$\\Delta R$ between Truth-Jet Axis and Ghost Matched Partons (Arb.)"

    th2.xBins = 500
    th2.yBins = 500

    th2.xMin = 0
    th2.xMax = 600

    th2.yMin = 0
    th2.yMax = 0.6

    th2.Filename = "Figure.5.q"
    th2.SaveFigure()

    names = ["Resonance (Leptonic)", "Resonance (Hadronic)", "Spectator (Leptonic)", "Spectator (Hadronic)"]
    fig_ = ["r", "s", "t", "u"]
    for x in range(len(names)):
        key = maps[x]
        name = names[x]
        hists = []
        for sym in data[key]:
            th_ = TH1F()
            th_.xData = data[key][sym]["dr"]
            th_.Title = sym
            hists.append(th_)

        th_p = path(TH1F())
        th_p.Histograms = hists
        th_p.Title = "$\\Delta$R Between the Truth-Jet Axis and Ghost Matched Partons \n for " + name + " Matched Truth Jets"
        th_p.xTitle = "$\\Delta$R (Arb.)"
        th_p.yTitle = "Entries"
        th_p.Filename = "Figure.5." + fig_[x]
        th_p.xBins = 500
        th_p.xMin  = 0
        th_p.xMax  = 0.6
        th_p.xStep = 0.1
        th_p.yLogarithmic = True
        th_p.SaveFigure()

def truthjet_contribution(ana):
    data = ana.truthjets_contribute

    th2 = path(TH2F())
    th2.Title = "Ghost Parton Energy Ratio as a function of Number of Partons \n For all Truth Jets"
    th2.xData = data[""]["all"]["n-partons"]
    th2.yData = data[""]["all"]["energy"]
    th2.xTitle = "Number of Ghost Matched Partons"
    th2.yTitle = "Energy Ratio (Truth Jet / Sum Ghost Partons) (Arb.)"

    th2.xBins = 20
    th2.yBins = 500

    th2.xMin = 0
    th2.xMax = 20
    th2.xStep = 4

    th2.yMin = 0
    th2.yMax = 2
    th2.yStep = 0.2

    th2.Filename = "Figure.5.v"
    th2.SaveFigure()


    th2 = path(TH2F())
    th2.Title = "Ghost Parton Transverse Momenta Ratio as a function of Number of Partons \n For all Truth Jets"
    th2.xData = data[""]["all"]["n-partons"]
    th2.yData = data[""]["all"]["pt"]
    th2.xTitle = "Number of Ghost Matched Partons"
    th2.yTitle = "Transverse Momenta Ratio (Truth Jet / Sum Ghost Partons) (Arb.)"

    th2.xBins = 20
    th2.yBins = 500

    th2.xMin = 0
    th2.xMax = 20
    th2.xStep = 4

    th2.yMin = 0
    th2.yMax = 2
    th2.yStep = 0.2

    th2.Filename = "Figure.5.w"
    th2.SaveFigure()

    hists = []
    for nt in data["n-tops"]:
        th_ = TH1F()
        th_.xData = data["n-tops"][nt]["energy_r"]
        th_.Title = str(nt) + "-Tops"
        hists.append(th_)

    th_p = path(TH1F())
    th_p.Histograms = hists
    th_p.Title = "Energy Ratio of n-Top Contributions to a given Truth-Jet"
    th_p.xTitle = "Top-Parton Energy Sum / All Top-Parton Energy Sum (Arb.)"
    th_p.yTitle = "Entries (Arb.)"
    th_p.Filename = "Figure.5.x"
    th_p.xBins = 500
    th_p.xMin  = 0
    th_p.xMax  = 1
    th_p.xStep = 0.1
    th_p.yLogarithmic = True
    th_p.SaveFigure()

    th2 = path(TH2F())
    th2.Title = "Truth-Jet Mass as a function of n-Top Contributions"
    th2.xData = ana.truthjet_mass["n-tops"]
    th2.yData = ana.truthjet_mass["all"]
    th2.xTitle = "Number of Tops"
    th2.yTitle = "Truth-Jet Mass (GeV)"

    th2.xBins = 4
    th2.yBins = 500

    th2.xMin = 0
    th2.xStep = 1
    th2.xMax = 4

    th2.yMin = 0
    th2.yMax = 100
    th2.yStep = 10

    th2.Filename = "Figure.5.y"
    th2.SaveFigure()

def TopTruthJets(ana):
    top_mass_truthjets(ana)
    top_truthjet_cluster(ana)
    truthjet_partons(ana)
    truthjet_contribution(ana)
