from AnalysisG.core.plotting import TH1F, TH2F

global figure_path
global mass_point

def path(hist):
    hist.Style = "ATLAS"
    hist.OutputDirectory = figure_path + "/top-truthjets/" + mass_point
    return hist

def top_mass_truthjets(ana):

    hists = []
    for x in ["Leptonic", "Hadronic"]:
        th = TH1F()
        th.Title = x
        th.xData = ana.top_mass[x.lower()][""][""]
        hists += [th]

    th_ = path(TH1F())
    th_.Histograms = hists
    th_.Title = "Normalized Invariant Mass of Matched Truth Tops \n using Truth Jets (and Truth Leptons/Neutrinos)"
    th_.xTitle = "Invariant Mass of Matched Top (GeV)"
    th_.yTitle = "Density (Arb.) / ($1$ GeV)"
    th_.Filename = "Figure.5.a"
    th_.xBins = 400
    th_.xMin  = 0
    th_.xMax  = 400
    th_.xStep = 20
    th_.Density = True
    th_.Overflow = False
    th_.SaveFigure()

    for kx, name in zip(["Leptonic", "Hadronic"], ["b", "c"]):
        x = kx.lower()
        hists = {}
        for nj in sorted(ana.top_mass["ntruthjets"][x]):
            data = ana.top_mass["ntruthjets"][x][nj]
            if int(nj) > 3: title = "$\\geq 4$ Truth Jets"
            else: title = str(nj) + " Truth Jets"
            if title not in hists: hists[title] = []
            hists[title] += data

        for k in hists:
            th = TH1F()
            th.Title = k
            th.xData = hists[k]
            hists[k] = th

        th_ = path(TH1F())
        th_.Histograms = [hists[k] for k in sorted(hists)][1:] + [[hists[k] for k in sorted(hists)][0]]
        th_.Title = "Normalized Invariant Mass of " + kx + " Decaying Truth Top \n Segmented by n-TruthJet Contributions"
        th_.xTitle = "Invariant Mass of Matched Top (GeV)"
        th_.yTitle = "Density (Arb.) / ($1$ GeV)"
        th_.Filename = "Figure.5." + name
        th_.xBins = 400
        th_.xMin  = 0
        th_.xMax  = 400
        th_.xStep = 20
        th_.Stacked = True
        th_.Density = True
        th_.Overflow = False
        th_.SaveFigure()

    for kx, name in zip(["Leptonic", "Hadronic"], ["d", "e"]):
        x = kx.lower()
        hists = []
        for nj in sorted(ana.top_mass["merged_tops"][x]):
            th = TH1F()
            th.Title = str(nj) + "-Tops"
            th.xData = ana.top_mass["merged_tops"][x][nj]
            hists += [th]

        th_ = path(TH1F())
        th_.Histograms = hists
        th_.Title = "Normalized Invariant Mass of " + kx + " Decaying Truth Tops \n Segmented by n-Tops Contributing to matched Truth Jets"
        th_.xTitle = "Invariant Mass of Matched Top (GeV)"
        th_.yTitle = "Density (Arb.) / ($1$ GeV)"
        th_.Filename = "Figure.5." + name
        th_.xBins = 400
        th_.xMin  = 0
        th_.xMax  = 400
        th_.xStep = 20
        th_.Stacked = True
        th_.Density = True
        th_.Overflow = False
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
    th_.Title = "Normalized $\\Delta$R Between Truth-Jets matched to a Mutal Top \n compared to Background (Other Truth-Jets)"
    th_.xTitle = "$\\Delta$R Between Truth-Jets (Arb.)"
    th_.yTitle = "Density (Arb.) / $0.02$"
    th_.Filename = "Figure.5.f"
    th_.xBins = 300
    th_.xMin  = 0
    th_.xMax  = 6
    th_.xStep = 0.4
    th_.Stacked = True
    th_.Density = True
    th_.Overflow = False
    th_.SaveFigure()

    names = ["Resonance (Leptonic)", "Resonance (Hadronic)", "Spectator (Leptonic)", "Spectator (Hadronic)"]
    fig_ = ["g", "h", "i", "j"]
    map_m = maps[1:]

    for n, j, f in zip(names, map_m, fig_):
        th2 = path(TH2F())
        th2.Title = "$\\Delta$R Between Truth-Jets from " + n + " Top \n as a Function of the Top's Energy"
        th2.xData = ana.truthjet_top[j]["energy"]
        th2.yData = ana.truthjet_top[j]["dr"]

        th2.xTitle = "Truth Top Energy / ($3$ GeV)"
        th2.yTitle = "$\\Delta R$ (Arb.) / $0.02$"

        th2.xBins = 500
        th2.yBins = 200

        th2.xMin = 0
        th2.xMax = 1500
        th2.xStep = 100

        th2.yMin = 0
        th2.yMax = 4.0
        th2.yStep = 0.2
        th2.Color = "RdBu_r"

        th2.Filename = "Figure.5." + f
        th2.SaveFigure()

    fig_ = ["k", "l", "m", "n"]
    for n, j, f in zip(names, map_m, fig_):
        th2 = path(TH2F())
        th2.Title = "$\\Delta$R Between Truth-Jets from " + n + " Top \n as a Function of the Top's Transverse Momentum"
        th2.xData = ana.truthjet_top[j]["energy"]
        th2.yData = ana.truthjet_top[j]["dr"]

        th2.xTitle = "Truth Top $p_T$ / ($3$ GeV)"
        th2.yTitle = "$\\Delta R$ (Arb.) / $0.02$"

        th2.xBins = 500
        th2.yBins = 200

        th2.xMin = 0
        th2.xMax = 1500
        th2.xStep = 100

        th2.yMin = 0
        th2.yMax = 4.0
        th2.yStep = 0.2
        th2.Color = "RdBu_r"

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
            title = "null"
            if k == "c" or k == "d" or k == "s" or k == "u": title = "light-quark"
            if k == "b": title = "b-quark"
            if k == "g": title = "gluon"
            if title not in symbolic:
                symbolic[title] = []
                energy[title] = []
                transverse[title] = []
            symbolic[title] += data[mode][k]["dr"]
            energy[title] += data[mode][k]["parton-energy"]
            transverse[title] += data[mode][k]["parton-pt"]

    hists = []
    for x in symbolic:
        th_ = TH1F()
        th_.xData = symbolic[x]
        th_.Title = x
        hists.append(th_)

    th_p = path(TH1F())
    th_p.Histograms = hists
    th_p.Title = "$\\Delta$R Distribution Between \n the Truth-Jet Axis and Ghost Matched Partons"
    th_p.xTitle = "$\\Delta$R Between Truth-Jet Axis and Matched Ghost Parton (Arb.)"
    th_p.yTitle = "Entries / $0.005$"
    th_p.Filename = "Figure.5.o"
    th_p.xBins = 120
    th_p.xMin  = 0
    th_p.xMax  = 0.6
    th_p.xStep = 0.05
    th_p.Stacked = True
    th_p.SaveFigure()

    th2 = path(TH2F())
    th2.Title = "$\\Delta$R Between Truth-Jet Axis and Ghost Matched Partons \n as a function of Ghost Parton Energy"
    th2.xData = sum([x for x in energy.values()], [])
    th2.yData = sum([x for x in symbolic.values()], [])
    th2.xTitle = "Ghost Matched Parton Energy / ($2$ GeV)"
    th2.yTitle = "$\\Delta R$ Between Truth-Jet Axis and Parton (Arb.) / 0.005"

    th2.xBins = 200
    th2.yBins = 100

    th2.xStep = 40
    th2.yStep = 0.05

    th2.xMin = 0
    th2.xMax = 400

    th2.yMin = 0
    th2.yMax = 0.5

    th2.Color = "RdBu_r"
    th2.Filename = "Figure.5.p"
    th2.SaveFigure()

    th2 = path(TH2F())
    th2.Title = "$\\Delta$R Between Truth-Jet Axis and Ghost Matched Partons \n as a function of Ghost Parton $p_T$"
    th2.xData = sum([x for x in transverse.values()], [])
    th2.yData = sum([x for x in symbolic.values()], [])
    th2.xTitle = "Ghost Matched Parton $p_T$ / ($1$ GeV)"
    th2.yTitle = "$\\Delta R$ between Truth-Jet Axis and Parton (Arb.) / 0.005"

    th2.xBins = 250
    th2.yBins = 100

    th2.xStep = 25
    th2.yStep = 0.05

    th2.xMin = 0
    th2.xMax = 250

    th2.yMin = 0
    th2.yMax = 0.5

    th2.Color = "RdBu_r"
    th2.Filename = "Figure.5.q"
    th2.SaveFigure()

    names = ["Resonance (Leptonic)", "Resonance (Hadronic)", "Spectator (Leptonic)", "Spectator (Hadronic)"]
    fig_ = ["r", "s", "t", "u"]
    for x in range(len(names)):
        key = maps[x]
        name = names[x]
        hists = {}
        for k in data[key]:
            title = "null"
            if k == "c" or k == "d" or k == "s" or k == "u": title = "light-quark"
            if k == "b": title = "b-quark"
            if k == "g": title = "gluon"
            if title not in hists: hists[title] = []
            hists[title] += data[key][k]["dr"]


        for k in hists:
            th_ = TH1F()
            th_.Title = k
            th_.xData = hists[k]
            hists[k] = th_

        th_p = path(TH1F())
        th_p.Histograms = list(hists.values())
        th_p.Title = "$\\Delta$R Between the Truth-Jet Axis and Ghost Matched Partons \n for " + name + " Matched Truth Jets"
        th_p.xTitle = "$\\Delta$R Between Truth-Jets (Arb.)"
        th_p.yTitle = "Entries / $0.02$"
        th_p.Filename = "Figure.5." + fig_[x]
        th_p.xBins = 300
        th_p.xMin  = 0
        th_p.xMax  = 0.6
        th_p.xStep = 0.05
        th_p.Stacked = True
        th_p.SaveFigure()

def truthjet_contribution(ana):
    data = ana.truthjets_contribute

    th2 = path(TH2F())
    th2.Title = "Truth Jet to Total Ghost Parton Energy Ratio \n as a function of Number of Constributing Partons"
    th2.xData = data[""]["all"]["n-partons"]
    th2.yData = data[""]["all"]["energy"]
    th2.xTitle = "Number of Ghost Matched Partons"
    th2.yTitle = "Energy Ratio (Truth Jet / $\\Sigma$ Ghost Partons)"

    th2.xBins = 10
    th2.yBins = 120

    th2.xMin = 1
    th2.xMax = 11
    th2.xStep = 1

    th2.yMin = 0.4
    th2.yMax = 1.6
    th2.yStep = 0.1
    th2.Color = "RdBu_r"

    th2.Filename = "Figure.5.v"
    th2.SaveFigure()

    th2 = path(TH2F())
    th2.Title = "Truth Jet to Total Ghost Parton $p_T$ Ratio \n as a function of Number of Contributing Partons"
    th2.xData = data[""]["all"]["n-partons"]
    th2.yData = data[""]["all"]["pt"]
    th2.xTitle = "Number of Ghost Matched Partons"
    th2.yTitle = "$p_T$ Ratio (Truth Jet / $\\Sigma$ Ghost Partons)"

    th2.xBins = 10
    th2.yBins = 120

    th2.xMin = 1
    th2.xMax = 11
    th2.xStep = 1

    th2.yMin = 0.4
    th2.yMax = 1.6
    th2.yStep = 0.1
    th2.Color = "RdBu_r"

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
    th_p.Title = "Energy Ratio of n-Top Contributions to Matched Truth-Jet"
    th_p.xTitle = "Energy Ratio ($\\text{top}_{i}$ / $\\Sigma_{i}^{n} \\text{top}_{i}$)"
    th_p.yTitle = "Entries / $0.01$"
    th_p.Filename = "Figure.5.x"
    th_p.xBins = 100
    th_p.xMin  = 0
    th_p.xMax  = 1
    th_p.xStep = 0.1
    th_p.Stacked = True
    th_p.yLogarithmic = True
    th_p.SaveFigure()

    th2 = path(TH2F())
    th2.Title = "Truth-Jet Invariant Mass as a function of n-Top Contributions"
    th2.xData = ana.truthjet_mass["n-tops"]
    th2.yData = ana.truthjet_mass["all"]
    th2.xTitle = "Number of matched Tops"
    th2.yTitle = "Truth-Jet Invariant Mass / ($1$ GeV)"

    th2.xBins = 3
    th2.yBins = 100

    th2.xMin = 1
    th2.xStep = 1
    th2.xMax = 4

    th2.yMin = 0
    th2.yMax = 100
    th2.yStep = 10
    th2.Color = "RdBu_r"

    th2.Filename = "Figure.5.y"
    th2.SaveFigure()

def TopTruthJets(ana):
    top_mass_truthjets(ana)
    top_truthjet_cluster(ana)
    truthjet_partons(ana)
    truthjet_contribution(ana)
