from AnalysisG.Plotting import TH1F, TH2F
global figure_path

def settings():
    settings = {
            "Style" : "ROOT",
            "OutputDirectory" : figure_path + "children-kinematics/figures/",
            "Histograms" : [],
            "Histogram" : None,
            "FontSize" : 15,
            "LabelSize" : 20,
            "xScaling" : 10,
            "yScaling" : 12,
            "LegendLoc" : "upper right",
            "yTitle" : "Entries (arb.)"
    }
    return settings

def settings_th2f():
    settings = {
            "Style" : "ROOT",
            "OutputDirectory" : figure_path + "top/figures/",
    }
    return settings

def top_children_dr(ana):
    data = ana.dr_children_top

    th_lep = TH1F()
    th_lep.Title = "Leptonic"
    th_lep.xData = data["lep"]["all"]

    th_had = TH1F()
    th_had.Title = "Hadronic"
    th_had.xData = data["had"]["all"]

    sett = settings()
    all_t = TH1F(**sett)
    all_t.Histograms = [th_lep, th_had]
    all_t.Title = "$\\Delta R$ between Truth Children and Mutual Top for \n Different Decay Modes"
    all_t.xTitle = "$\\Delta R$ (arb.)"
    all_t.xMin = 0
    all_t.xMax = 6
    all_t.xStep = 1
    all_t.xBins = 100
    all_t.Filename = "Figure.2.a"
    all_t.SaveFigure()

    hist = []
    leptonic = data["lep"]
    for sym in leptonic:
        if sym == "all": continue
        th_lep = TH1F()
        th_lep.Title = sym
        th_lep.xData = leptonic[sym]
        hist.append(th_lep)

    sett = settings()
    all_t = TH1F(**sett)
    all_t.Histograms = hist
    all_t.Title = "$\\Delta R$ between Truth Children and Mutual Top (Leptonic Mode) \n Partitioned into Particle Symbol"
    all_t.xTitle = "$\\Delta R$ (arb.)"
    all_t.xMin = 0
    all_t.xMax = 6
    all_t.xStep = 1
    all_t.xBins = 100
    all_t.Filename = "Figure.2.b"
    all_t.SaveFigure()


    hist = []
    hadronic = data["had"]
    for sym in hadronic:
        if sym == "all": continue
        th_lep = TH1F()
        th_lep.Title = sym
        th_lep.xData = hadronic[sym]
        hist.append(th_lep)

    sett = settings()
    all_t = TH1F(**sett)
    all_t.Histograms = hist
    all_t.Title = "$\\Delta R$ between Truth Children and Mutual Top (Hadronic Mode) \n Partitioned into Particle Symbol"
    all_t.xTitle = "$\\Delta R$ (arb.)"
    all_t.xMin = 0
    all_t.xMax = 6
    all_t.xStep = 1
    all_t.xBins = 100
    all_t.Filename = "Figure.2.c"
    all_t.SaveFigure()

    data = ana.dr_children_cluster
    mut = TH1F()
    mut.Title = "mutual"
    mut.xData = data["Mutual"]
    mut.Normalize = True

    nmut = TH1F()
    nmut.Title = "non-mutual"
    nmut.xData = data["non-Mutual"]
    nmut.Normalize = True

    sett = settings()
    all_t = TH1F(**sett)
    all_t.Histograms = [mut, nmut]
    all_t.Title = "$\\Delta R$ between all Truth Children Pair Permutations (Normalized)"
    all_t.xTitle = "$\\Delta R$ (arb.)"
    all_t.yTitle = "Fraction (arb.)"
    all_t.xMin = 0
    all_t.xMax = 6
    all_t.xStep = 1
    all_t.xBins = 100
    all_t.Normalize = True
    all_t.Filename = "Figure.2.d"
    all_t.SaveFigure()


def top_children_fractions(ana):
    data = ana.fractional






def ChildrenKinematics(ana):
    top_children_dr(ana)
    top_children_fractions(ana)
