from AnalysisG.core.plotting import TH1F
colors = ["red", "green", "blue", "orange", "magenta", "cyan", "pink"]
global itr

def default(tl):
    tl.Style = r"ATLAS"
    tl.DPI = 250
    tl.TitleSize = 20
    tl.AutoScaling = False
    tl.Overflow = False
    tl.yScaling = 10*0.75
    tl.xScaling = 15*0.6
    tl.FontSize = 15
    tl.Alpha = 0.5
    tl.AxisSize = 14

def path(hist):
    hist.OutputDirectory = "figures"
    default(hist)
    return hist

def topmass(sel):
    global itr
    itr = iter(colors)
    hists = []

    data = []
    for i in sel.truth_tops: data += [t.Mass / 1000.0 for t in i]

    #tx = TH1F()
    #tx.Color = next(itr)
    #tx.Title = "Truth Top"
    #tx.xData = data
    #hists.append(tx)

    #data = []
    #for i in sel.top_children: data += [t.Mass / 1000.0 for t in i]
    #tx = TH1F()
    #tx.Color = next(itr)
    #tx.Title = "Truth Children"
    #tx.xData = data
    #hists.append(tx)

    data = []
    for i in sel.truth_jets: data += [t.Mass / 1000.0 for t in i]
    tx = TH1F()
    tx.Color = next(itr)
    tx.Title = "Truth Jets"
    tx.xData = data
    hists.append(tx)

    data = []
    for i in sel.jets_children: data += [t.Mass / 1000.0 for t in i]
    tx = TH1F()
    tx.Color = next(itr)
    tx.Title = "Jets Children"
    tx.xData = data
    hists.append(tx)

    data = []
    for i in sel.jets_leptons: data += [t.Mass / 1000.0 for t in i]
    tx = TH1F()
    tx.Color = next(itr)
    tx.Title = "Jets Leptons"
    tx.xData = data
    hists.append(tx)


    th = path(TH1F())
    th.Histograms = hists
    th.Title = "Truth Matched Top Mass"
    th.xTitle = "Invariant Mass (GeV)"
    th.xStep = 40
    th.xBins = 400
    th.xMin = 0
    th.xMax = 400
    th.yMin = 1
#    th.yMax = 1
#    th.Density = True
    th.yLogarithmic = True
    th.yTitle = "Tops (arb.)"
    th.Filename = "TopMass"
    th.SaveFigure()

def entry(sel):
    topmass(sel)



