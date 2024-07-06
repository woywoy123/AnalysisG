from AnalysisG.Plotting import TH1F, TH2F
global figure_path

def settings():
    settings = {
            "Style" : "ROOT",
            "OutputDirectory" : figure_path + "truth-jets/figures/",
            "Histograms" : [],
            "Histogram" : None,
            "FontSize" : 15,
            "LabelSize" : 20,
            "xScaling" : 10,
            "yScaling" : 12,
            "LegendLoc" : "upper right"
    }
    return settings

def settings_th2f():
    settings = {
            "Style" : "ROOT",
            "OutputDirectory" : figure_path + "truth-jets/figures/",
    }
    return settings

def top_mass_truthjets(ana):

    sett = settings()
    for x in ["leptonic", "hadronic"]:
        th = TH1F()
        th.Title = x
        th.xData = ana.top_mass[x]
        sett["Histograms"] += [th]

    th_ = TH1F(**sett)
    th_.Title = "Invariant Top-Quark Mass from Matched Truth Jets (and Truth Leptons/Neutrinos)"
    th_.xTitle = "Invariant Mass (GeV)"
    th_.yTitle = "Entries <unit>"
    th_.Filename = "Figure.7.a"
    th_.xBins = 500
    th_.xMin  = 0
    th_.xMax  = 500
    th_.xStep = 50
    th_.SaveFigure()


    for x, name in zip(["leptonic", "hadronic"], ["b", "c"]):
        sett = settings()
        for nj in ana.top_mass["ntruthjets"][x]:
            th = TH1F()
            th.Title = str(nj) + "-TruthJets"
            th.xData = ana.top_mass["ntruthjets"][x][nj]
            sett["Histograms"] += [th]

        th_ = TH1F(**sett)
        th_.Title = "Invariant Top-Quark Mass for " + x + " Decays for n-TruthJet Contributions"
        th_.xTitle = "Invariant Mass (GeV)"
        th_.yTitle = "Entries <unit>"
        th_.Filename = "Figure.7." + name
        th_.xBins = 500
        th_.xMin  = 0
        th_.xMax  = 500
        th_.xStep = 50
        th_.Stack = True
        th_.SaveFigure()


    for x, name in zip(["leptonic", "hadronic"], ["d", "e"]):
        sett = settings()
        for nj in ana.top_mass["merged_tops"][x]:
            th = TH1F()
            th.Title = str(nj) + "-Tops"
            th.xData = ana.top_mass["merged_tops"][x][nj]
            sett["Histograms"] += [th]

        th_ = TH1F(**sett)
        th_.Title = "Invariant Top-Quark Mass for " + x + " Decays with \n n-Top Contributions in Truth Jets"
        th_.xTitle = "Invariant Mass (GeV)"
        th_.yTitle = "Entries <unit>"
        th_.Filename = "Figure.7." + name
        th_.xBins = 500
        th_.xMin  = 0
        th_.xMax  = 500
        th_.xStep = 50
        th_.Stack = True
        th_.SaveFigure()

def top_truthjet_cluster(ana):

    sett = settings()
    maps = ["background", "resonant-leptonic", "resonant-hadronic", "spectator-leptonic", "spectator-hadronic"]
    for x in maps:
        th = TH1F()
        th.Title = x
        th.xData = ana.truthjet_top[x]["dr"]
        sett["Histograms"] += [th]

    th_ = TH1F(**sett)
    th_.Title = "$\\Delta$R Between Truth-Jets matched to Mutal Top \n compared to Background (Other Truth-Jets)"
    th_.xTitle = "$\\Delta$R (arb.)"
    th_.yTitle = "Entries"
    th_.Filename = "Figure.7.f"
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
        sett = settings_th2f()
        th2 = TH2F(**sett)
        th2.Title = "$\\Delta$R Between Truth-Jets from " + n + " Top \n as a Function of the Top's Energy"
        th2.xData = ana.truthjet_top[j]["energy"]
        th2.yData = ana.truthjet_top[j]["dr"]

        th2.xTitle = "Originating Top Energy (GeV)"
        th2.yTitle = "$\\Delta R$ (arb.)"

        th2.xBins = 500
        th2.yBins = 500

        th2.xMin = 0
        th2.xMax = 1500

        th2.yMin = 0
        th2.yMax = 6

        th2.yOverFlow = True
        th2.xOverFlow = True

        th2.Color = "tab20c"

        th2.Filename = "Figure.7." + f
        th2.SaveFigure()

    fig_ = ["k", "l", "m", "n"]
    for n, j, f in zip(names, map_m, fig_):
        sett = settings_th2f()
        th2 = TH2F(**sett)
        th2.Title = "$\\Delta$R Between Truth-Jets from " + n + " Top \n as a Function of the Top's Transverse Momenta"
        th2.xData = ana.truthjet_top[j]["energy"]
        th2.yData = ana.truthjet_top[j]["dr"]

        th2.xTitle = "Originating Top Transverse Momenta (GeV)"
        th2.yTitle = "$\\Delta R$ (arb.)"

        th2.xBins = 500
        th2.yBins = 500

        th2.xMin = 0
        th2.xMax = 1500

        th2.yMin = 0
        th2.yMax = 6

        th2.yOverFlow = True
        th2.xOverFlow = True

        th2.Color = "tab20c"

        th2.Filename = "Figure.7." + f
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

    sett = settings()
    for x in symbolic:
        th_ = TH1F()
        th_.xData = symbolic[x]
        th_.Title = x
        sett["Histograms"].append(th_)

    th_p = TH1F(**sett)
    th_p.Title = "$\\Delta$R Between the Truth-Jet Axis and Ghost Matched Partons"
    th_p.xTitle = "$\\Delta$R (arb.)"
    th_p.yTitle = "Entries"
    th_p.Filename = "Figure.7.o"
    th_p.xBins = 500
    th_p.xMin  = 0
    th_p.xMax  = 0.6
    th_p.xStep = 0.1
    th_p.yLogarithmic = True
    th_p.SaveFigure()

    sett = settings_th2f()
    th2 = TH2F(**sett)
    th2.Title = "$\\Delta$R Between Truth-Jet Axis and Ghost Matched Partons \n as a function of Ghost Parton Energy"
    th2.xData = sum([x for x in energy.values()], [])
    th2.yData = sum([x for x in symbolic.values()], [])
    th2.xTitle = "Ghost Matched Parton Energy (GeV)"
    th2.yTitle = "$\\Delta R$ between Truth-Jet Axis and Ghost Matched Partons (arb.)"

    th2.xBins = 500
    th2.yBins = 500

    th2.xMin = 0
    th2.xMax = 600

    th2.yMin = 0
    th2.yMax = 0.6

    th2.yOverFlow = True
    th2.xOverFlow = True

    th2.Filename = "Figure.7.p"
    th2.SaveFigure()

    sett = settings_th2f()
    th2 = TH2F(**sett)
    th2.Title = "$\\Delta$R Between Truth-Jet Axis and Ghost Matched Partons \n as a function of Ghost Parton Transverse Momenta"
    th2.xData = sum([x for x in transverse.values()], [])
    th2.yData = sum([x for x in symbolic.values()], [])
    th2.xTitle = "Ghost Matched Transverse Momenta (GeV)"
    th2.yTitle = "$\\Delta R$ between Truth-Jet Axis and Ghost Matched Partons (arb.)"

    th2.xBins = 500
    th2.yBins = 500

    th2.xMin = 0
    th2.xMax = 600

    th2.yMin = 0
    th2.yMax = 0.6

    th2.yOverFlow = True
    th2.xOverFlow = True

    th2.Filename = "Figure.7.q"
    th2.SaveFigure()


    names = ["Resonance (Leptonic)", "Resonance (Hadronic)", "Spectator (Leptonic)", "Spectator (Hadronic)"]
    fig_ = ["r", "s", "t", "u"]
    for x in range(len(names)):
        key = maps[x]
        name = names[x]
        sett = settings()
        for sym in data[key]:
            th_ = TH1F()
            th_.xData = data[key][sym]["dr"]
            th_.Title = sym
            sett["Histograms"].append(th_)

        th_p = TH1F(**sett)
        th_p.Title = "$\\Delta$R Between the Truth-Jet Axis and Ghost Matched Partons \n for " + name + " Matched Truth Jets"
        th_p.xTitle = "$\\Delta$R (arb.)"
        th_p.yTitle = "Entries"
        th_p.Filename = "Figure.7." + fig_[x]
        th_p.xBins = 500
        th_p.xMin  = 0
        th_p.xMax  = 0.6
        th_p.xStep = 0.1
        th_p.yLogarithmic = True
        th_p.SaveFigure()

def truthjet_contribution(ana):
    data = ana.truthjets_contribute

    sett = settings_th2f()
    th2 = TH2F(**sett)
    th2.Title = "Ghost Parton Energy Ratio as a function of Number of Partons \n For all Truth Jets"
    th2.xData = data["all"]["n-partons"]
    th2.yData = data["all"]["energy"]
    th2.xTitle = "Number of Ghost Matched Partons"
    th2.yTitle = "Energy Ratio (Truth Jet / Sum Ghost Partons) (arb.)"

    th2.xBins = 20
    th2.yBins = 500

    th2.xMin = 0
    th2.xMax = 20
    th2.xStep = 4

    th2.yMin = 0
    th2.yMax = 2
    th2.yStep = 0.2

    th2.Filename = "Figure.7.v"
    th2.SaveFigure()


    sett = settings_th2f()
    th2 = TH2F(**sett)
    th2.Title = "Ghost Parton Transverse Momenta Ratio as a function of Number of Partons \n For all Truth Jets"
    th2.xData = data["all"]["n-partons"]
    th2.yData = data["all"]["pt"]
    th2.xTitle = "Number of Ghost Matched Partons"
    th2.yTitle = "Transverse Momenta Ratio (Truth Jet / Sum Ghost Partons) (arb.)"

    th2.xBins = 20
    th2.yBins = 500

    th2.xMin = 0
    th2.xMax = 20
    th2.xStep = 4

    th2.yMin = 0
    th2.yMax = 2
    th2.yStep = 0.2

    th2.Filename = "Figure.7.w"
    th2.SaveFigure()


    sett = settings()
    for nt in data["n-tops"]:
        th_ = TH1F()
        th_.xData = data["n-tops"][nt]["energy_r"]
        th_.Title = str(nt) + "-Tops"
        sett["Histograms"].append(th_)

    th_p = TH1F(**sett)
    th_p.Title = "Energy Ratio of n-Top Contributions to a given Truth-Jet"
    th_p.xTitle = "Top-Parton Energy Sum / All Top-Parton Energy Sum"
    th_p.yTitle = "Entries"
    th_p.Filename = "Figure.7.x"
    th_p.xBins = 500
    th_p.xMin  = 0
    th_p.xMax  = 1
    th_p.xStep = 0.1
    th_p.yLogarithmic = True
    th_p.SaveFigure()

    sett = settings_th2f()
    th2 = TH2F(**sett)
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

    th2.Color = "tab20c"
    th2.Filename = "Figure.7.y"
    th2.SaveFigure()

def TopTruthJets(ana):
    top_mass_truthjets(ana)
    top_truthjet_cluster(ana)
    truthjet_partons(ana)
    truthjet_contribution(ana)
