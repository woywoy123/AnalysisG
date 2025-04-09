from AnalysisG.core.plotting import TH1F, TH2F

global figure_path
global mass_point

def colors(colx = ["blue", "green", "orange", "purple", "magenta", "aqua",  "orange",  "green",  "blue",  "olive",  "teal",  "gold",  "azure",  "lime",  "crimson",  "cyan", "magenta",  "orchid",  "sienna",  "silver",  "salmon",  "darkblue"]): return iter(colx)


def path(hist):
    hist.Style = "ATLAS"
    hist.OutputDirectory = figure_path + "/childrenkinematics/" + mass_point
    hist.Overflow = True
    return hist

def pdgid_mapping(pdgid):
    if "nu" in pdgid: title = "$\\nu_{\\ell}$"
    elif "e" in pdgid or "tau" in pdgid or "mu" in pdgid: title = "$\\ell$"
    elif "u" in pdgid or "c" in pdgid or "d" in pdgid or "s" in pdgid: title = "\\texttt{light-quark}"
    else: title = pdgid
    return title

def apply(hist):
    col = colors()
    for k in hist:
        th = TH1F()
        th.Title = "$" + k + "$"
        th.xData = hist[k]
        th.Color = next(col)
        hist[k] = th
    return list(hist.values())

def kinematics_pt(ana, mp):
    rkin = ana.res_kinematics
    skin = ana.spec_kinematics

    rth_pt = TH1F()
    rth_pt.Title = "Resonance"
    rth_pt.xData = rkin["pt"]
    rth_pt.Color = "red"

    sth_pt = TH1F()
    sth_pt.Title = "Spectator"
    sth_pt.xData = skin["pt"]
    sth_pt.Color = "blue"

    th = path(TH1F())
    th.Histograms = [rth_pt, sth_pt]
    th.Title  = "$p_T$ for Top Children from Top-Philic Mass Injection at " + mp #Distributions for \n Spectator and Resonant Top Children"
    th.xTitle = "$p_T$ of Top Children (GeV)"
    th.yTitle = "Density (Arb.) / (10 GeV)"
    th.xBins = 60
    th.xStep = 40
    th.xMin = 0
    th.xMax = 600
    th.Density = True
    th.Filename = "Figure.3.a"
    th.SaveFigure()

    hist = {}
    for pdgid in sorted(ana.res_pdgid_kinematics):
        title = pdgid_mapping(pdgid)
        if title not in hist: hist[title] = []
        hist[title] += ana.res_pdgid_kinematics[pdgid]["pt"]

    hist = apply(hist)
    th = path(TH1F())
    th.Histograms = hist
    th.Title  = "Normalized $p_T$ Distributions for Top Children from Resonant Top Segmented by Parton Type"
    th.xTitle = "$p_T$ of Top Children (GeV)"
    th.yTitle = "Density (Arb.) / (10 GeV)"
    th.xBins = 60
    th.xStep = 40
    th.xMin = 0
    th.xMax = 600
    th.Stacked = True
    th.Density = True
    th.yLogarithmic = True
    th.Filename = "Figure.3.b"
    th.SaveFigure()
    
    hist = {}
    for pdgid in sorted(ana.spec_pdgid_kinematics):
        title = pdgid_mapping(pdgid)
        if title not in hist: hist[title] = []
        hist[title] += ana.spec_pdgid_kinematics[pdgid]["pt"]


    hist = apply(hist)
    th = path(TH1F())
    th.Histograms = hist
    th.Title  = "Normalized $p_T$ Distribution for Top Children from Spectator Top Segmented by Parton Type"
    th.xTitle = "$p_T$ of Top Children (GeV)"
    th.yTitle = "Density (Arb.) / (10 GeV)"
    th.xBins = 60
    th.xStep = 40
    th.xMin = 0
    th.xMax = 600
    th.Stacked = True
    th.Density = True
    th.yLogarithmic = True
    th.Filename = "Figure.3.c"
    th.SaveFigure()


def kinematics_eta(ana):
    rkin = ana.res_kinematics
    skin = ana.spec_kinematics

    rth_pt = TH1F()
    rth_pt.Title = "Resonance"
    rth_pt.xData = rkin["eta"]

    sth_pt = TH1F()
    sth_pt.Title = "Spectato"
    sth_pt.xData = skin["eta"]

    th = path(TH1F())
    th.Histograms = [rth_pt, sth_pt]
    th.Title = "Normalized Pseudorapidity ($\\eta$) Distribution for Spectator and Resonance Top Children"
    th.xTitle = "Pseudorapidity ($\\eta$) (Arb.)"
    th.yTitle = "Density (Arb.) / (0.05)"
    th.xBins = 240
    th.xStep = 1
    th.xMin = -6
    th.xMax =  6
    th.Density = True
    th.Filename = "Figure.3.d"
    th.SaveFigure()


    hist = {}
    for pdgid in sorted(ana.res_pdgid_kinematics):
        title = pdgid_mapping(pdgid)
        if title not in hist: hist[title] = []
        hist[title] += ana.res_pdgid_kinematics[pdgid]["eta"]

    hist = apply(hist)
    th = path(TH1F())
    th.Histograms = hist
    th.Title  = "Normalized Pseudorapidity ($\\eta$) Distributions for Top Children from Resonant Top Segmented by Parton Type"
    th.xTitle = "Pseudorapidity ($\\eta$) of Top Children (Arb.)"
    th.yTitle = "Density (Arb.) / (0.05)"
    th.xBins = 240
    th.xStep = 1
    th.xMin = -6
    th.xMax =  6
    th.Stacked = True
    th.Density = True
    th.Filename = "Figure.3.e"
    th.SaveFigure()

    hist = {}
    for pdgid in sorted(ana.spec_pdgid_kinematics):
        title = pdgid_mapping(pdgid)
        if title not in hist: hist[title] = []
        hist[title] += ana.spec_pdgid_kinematics[pdgid]["eta"]

    hist = apply(hist)
    th = path(TH1F())
    th.Histograms = hist
    th.Title  = "Normalized Pseudorapidity ($\\eta$) Distributions for Top Children from Spectator Top Segmented by Parton Type"
    th.xTitle = "Pseudorapidity ($\\eta$) of Top Children (Arb.)"
    th.yTitle = "Density (Arb.) / (0.05)"
    th.xBins = 240
    th.xStep = 1
    th.xMin = -6
    th.xMax =  6
    th.Stacked = True
    th.Density = True
    th.Filename = "Figure.3.f"
    th.SaveFigure()

def kinematics_phi(ana):
    rkin = ana.res_kinematics
    skin = ana.spec_kinematics

    rth_pt = TH1F()
    rth_pt.Title = "Resonance"
    rth_pt.xData = rkin["phi"]

    sth_pt = TH1F()
    sth_pt.Title = "Spectato"
    sth_pt.xData = skin["phi"]

    th = path(TH1F())
    th.Histograms = [rth_pt, sth_pt]
    th.Title = "Normalized Azimuthal Angle ($\\phi$) Distributions for Spectator and Resonant Top Children"
    th.xTitle = "Azimuthal Angle ($\\phi$) (rad)"
    th.yTitle = "Density (Abr.) / (0.1)"
    th.xBins = 350
    th.xStep = 1
    th.xMin = -3.5
    th.xMax =  3.5
    th.Density = True
    th.Stacked = True
    th.Filename = "Figure.3.g"
    th.SaveFigure()

    hist = {}
    for pdgid in sorted(ana.res_pdgid_kinematics):
        title = pdgid_mapping(pdgid)
        if title not in hist: hist[title] = []
        hist[title] += ana.res_pdgid_kinematics[pdgid]["phi"]

    hist = apply(hist)
    th = path(TH1F())
    th.Histograms = hist
    th.Title = "Normalized Azimuthal Angle ($\\phi$) Distributions for Top Children from Resonant Top Segmented by Parton Type"
    th.xTitle = "Azimuthal Angle ($\\phi$) (rad)"
    th.yTitle = "Density (Abr.) / (0.02)"
    th.xBins = 350
    th.xStep = 1
    th.xMin = -3.5
    th.xMax =  3.5
    th.Stacked = True
    th.Density = True
    th.Filename = "Figure.3.h"
    th.SaveFigure()

    hist = {}
    for pdgid in sorted(ana.spec_pdgid_kinematics):
        title = pdgid_mapping(pdgid)
        if title not in hist: hist[title] = []
        hist[title] += ana.spec_pdgid_kinematics[pdgid]["phi"]

    hist = apply(hist)
    th = path(TH1F())
    th.Histograms = hist
    th.Title = "Normalized Azimuthal Angle ($\\phi$) Distributions for Top Children from Spectator Top Segmented by Parton Type"
    th.xTitle = "Azimuthal Angle ($\\phi$) (rad)"
    th.yTitle = "Density (Abr.) / (0.0)"
    th.xBins = 350
    th.xStep = 1
    th.xMin = -3.5
    th.xMax =  3.5
    th.Stacked = True
    th.Density = True
    th.Filename = "Figure.3.i"
    th.SaveFigure()

def kinematics_energy(ana, mp):
    rkin = ana.res_kinematics
    skin = ana.spec_kinematics

    rth_pt = TH1F()
    rth_pt.Title = "Resonance"
    rth_pt.xData = rkin["energy"]
    rth_pt.Color = "red"

    sth_pt = TH1F()
    sth_pt.Title = "Spectato"
    sth_pt.xData = skin["energy"]
    sth_pt.Color = "blue"

    th = path(TH1F())
    th.Histograms = [rth_pt, sth_pt]
    th.Title = "Energy for Top Children from Top-Philic Mass Injection at " + mp
    th.xTitle = "Energy of Top Children (GeV)"
    th.yTitle = "Density (Arb.) / (10 GeV)"
    th.xBins = 100
    th.xStep = 100
    th.xMin = 0
    th.xMax = 1000
    th.Density = True
    th.Filename = "Figure.3.j"
    th.SaveFigure()


    hist = {}
    for pdgid in sorted(ana.res_pdgid_kinematics):
        title = pdgid_mapping(pdgid)
        if title not in hist: hist[title] = []
        hist[title] += ana.res_pdgid_kinematics[pdgid]["energy"]

    hist = apply(hist)
    th = path(TH1F())
    th.Histograms = hist
    th.Title = "Normalized Energy Distributions of Truth Children from Resonant Top Segmented by Parton Type"
    th.xTitle = "Energy of Top Children (GeV)"
    th.yTitle = "Density (Arb.) / (10 GeV)"
    th.xBins = 100
    th.xStep = 100
    th.xMin = 0
    th.xMax = 1000
    th.Stacked = True
    th.Density = True
    th.Filename = "Figure.3.k"
    th.SaveFigure()

    hist = {}
    for pdgid in sorted(ana.spec_pdgid_kinematics):
        title = pdgid_mapping(pdgid)
        if title not in hist: hist[title] = []
        hist[title] += ana.spec_pdgid_kinematics[pdgid]["energy"]

    hist = apply(hist)
    th = path(TH1F())
    th.Histograms = hist
    th.Title = "Normalized Energy Distributions of Truth Children from Specator Top Segmented by Parton Type"
    th.xTitle = "Energy of Top Children (GeV)"
    th.yTitle = "Density (Arb.) / (10 GeV)"
    th.xBins = 100
    th.xStep = 100
    th.xMin = 0
    th.xMax = 1000
    th.Stacked = True
    th.Density = True
    th.Filename = "Figure.3.l"
    th.SaveFigure()

def kinematics_decay_mode(ana):
    hist_pt = []
    hist_eta = []
    hist_phi = []
    hist_energy = []

    col = colors()
    for mode in [["lep", "leptonic"], ["had", "hadronic"]]:
        mod, title = mode

        th = TH1F()
        th.Color = next(col)
        th.Title = "spectator " + title
        th.xData = ana.spec_decay_mode[mod]["pt"]
        hist_pt.append(th)

    col = colors()
    for mode in [["lep", "leptonic"], ["had", "hadronic"]]:
        mod, title = mode

        th = TH1F()
        th.Color = next(col)
        th.Title = "resonance " + title
        th.xData = ana.res_decay_mode[mod]["pt"]
        hist_pt.append(th)

        th = TH1F()
        th.Color = next(col)
        th.Title = "resonance " + title
        th.xData = ana.res_decay_mode[mod]["eta"]
        hist_eta.append(th)

        th = TH1F()
        th.Color = next(col)
        th.Title = "spectator " + title
        th.xData = ana.spec_decay_mode[mod]["eta"]
        hist_eta.append(th)


        th = TH1F()
        th.Color = next(col)
        th.Title = "resonance " + title
        th.xData = ana.res_decay_mode[mod]["phi"]
        hist_phi.append(th)

        th = TH1F()
        th.Color = next(col)
        th.Title = "spectator " + title
        th.xData = ana.spec_decay_mode[mod]["phi"]
        hist_phi.append(th)


        th = TH1F()
        th.Color = next(col)
        th.Title = "resonance " + title
        th.xData = ana.res_decay_mode[mod]["energy"]
        hist_energy.append(th)

        th = TH1F()
        th.Color = next(col)
        th.Title = "spectator " + title
        th.xData = ana.spec_decay_mode[mod]["energy"]
        hist_energy.append(th)


    th = path(TH1F())
    th.Histograms = hist_pt
    th.Title = "$p_T$ Distributions for Resonant and Spectator Top Children Segmented by Decay Mode"
    th.xTitle = "$p_T$ of Top Children (GeV)"
    th.yTitle = "Entries / (10 GeV)"
    th.Alpha = 0.5
    th.xBins = 60
    th.xStep = 40
    th.xMin = 0
    th.xMax = 600
    #th.Density = True
    th.Filename = "Figure.3.m"
    th.SaveFigure()

    th = path(TH1F())
    th.Histograms = hist_eta
    th.Title = "Normalized Pseudorapidity ($\\eta$) Distributions for Resonant and Spectator Top Children Segmented into Decay Mode"
    th.xTitle = "Pseudorapidity ($\\eta$) (Arb.)"
    th.yTitle = "Density (Arb.) / (0.05)"
    th.xBins = 240
    th.xStep = 1
    th.xMin = -6
    th.xMax =  6
    th.Density = True
    th.Filename = "Figure.3.n"
    th.SaveFigure()

    th = path(TH1F())
    th.Histograms = hist_phi
    th.Title = "Normalized Azimuthal Angle ($\\phi$) Distributions for Resonant and Spectator Top Children Segmented into Decay Mode"
    th.xTitle = "Azimuthal Angle ($\\phi$) (rad)"
    th.yTitle = "Entries (Arb.) / (0.02)"
    th.xBins = 350
    th.xStep = 1
    th.xMin = -3.5
    th.xMax =  3.5
    th.Density = True
    th.Filename = "Figure.3.o"
    th.SaveFigure()

    th = path(TH1F())
    th.Histograms = hist_energy
    th.Title = "Energy Distributions for Resonant and Spectator Top Children Segmented into Decay Mode"
    th.xTitle = "Energy (GeV)"
    th.yTitle = "Entries / (10 GeV)"
    th.Alpha = 0.5
    th.xBins = 100
    th.xStep = 100
    th.xMin = 0
    th.xMax = 1000
    #th.Density = True
    th.Filename = "Figure.3.p"
    th.SaveFigure()

def dr_clustering(ana):
    modes = [
                ["CTRR", "Correct-Top-RR", "green"],
                ["FTRR", "False-Top-RR"  , "green"],
                ["CTSS", "Correct-Top-SS", "blue"],
                ["FTSS", "False-Top-SS"  , "blue"],
                ["FTRS", "False-Top-RS"  , "red"]
    ]

    hist = []
    for mode in modes:
        mod, title, cox = mode

        th = TH1F()
        th.Color = cox
        th.Title = title
        th.xData = ana.dr_clustering[mod]
        hist.append(th)

    th = path(TH1F())
    th.Histograms = hist
    th.Title = "$\\Delta R$ between Truth Children From (Non)-Mutual Top-Quarks"
    th.xTitle = "$\\Delta R$ (Arb.)"
    th.yTitle = "Entries"
    th.xBins = 200
    th.xStep = 0.5
    th.xMin = 0
    th.xMax = 6
    th.Density = True
    th.Filename = "Figure.3.q"
    th.SaveFigure()

def fractional(ana):
    rhad = ana.fractional["rhad-pt"]
    rlep = ana.fractional["rlep-pt"]
    shad = ana.fractional["shad-pt"]
    slep = ana.fractional["slep-pt"]

    hist = {}
    for pdgid in sorted(rhad):
        title = pdgid_mapping(pdgid)
        if "h#" + title not in hist: hist["h#" + title] = []
        hist["h#" + title] += rhad[pdgid]

    for pdgid in sorted(rlep):
        title = pdgid_mapping(pdgid)
        if "l#" + title not in hist: hist["l#" + title] = []
        hist["l#" + title] += rlep[pdgid]*0

    for tl in hist:
        th = TH1F()
        th.Title = tl.split("#")[1] + " (Hadronic Top)" if tl.startswith("h#") else tl.split("#")[1] + " (Leptonic Top)"
        th.xData = hist[tl]
        hist[tl] = th

    th = path(TH1F())
    th.Histograms = list(hist.values())
    th.Title = "Fractional $p_T$ Distribution of Truth Children from Mutual Top-Quark (Resonant)"
    th.xTitle = "Fraction of $p_T$ being Dispersed to Truth Child ($p_{T, \\text{child}} / p_{T, \\text{top}}$)"
    th.yTitle = "Entries / (0.10)"
    th.xBins  = 300
    th.xStep  = 0.3
    th.xMin   = 0
    th.xMax   = 3.0
    th.Stacked = True
    th.Filename = "Figure.3.r"
    th.SaveFigure()

    hist = {}
    for pdgid in sorted(shad):
        title = pdgid_mapping(pdgid)
        if "h#" + title not in hist: hist["h#" + title] = []
        hist["h#" + title] += shad[pdgid]

    for pdgid in sorted(slep):
        title = pdgid_mapping(pdgid)
        if "l#" + title not in hist: hist["l#" + title] = []
        hist["l#" + title] += slep[pdgid]

    for tl in hist:
        th = TH1F()
        th.Title = tl.split("#")[1] + " (Hadronic Top)" if tl.startswith("h#") else tl.split("#")[1] + " (Leptonic Top)"
        th.xData = hist[tl]
        hist[tl] = th

    th = path(TH1F())
    th.Histograms = list(hist.values())
    th.Title = "Fractional $p_T$ Distribution of Truth Children from Mutual Top-Quark (Spectator)"
    th.xTitle = "Fraction of $p_T$ being Dispersed to Truth Child ($p_{T, \\text{child}} / p_{T, \\text{top}}$)"
    th.yTitle = "Entries / $0.10$"
    th.xBins  = 300
    th.xStep  = 0.3
    th.xMin   = 0
    th.xMax   = 3.0
    th.Stacked = True
    th.Filename = "Figure.3.s"
    th.SaveFigure()

    rhad = ana.fractional["rhad-energy"]
    rlep = ana.fractional["rlep-energy"]
    shad = ana.fractional["shad-energy"]
    slep = ana.fractional["slep-energy"]

    hist = {}
    for pdgid in sorted(rhad):
        title = pdgid_mapping(pdgid)
        if "h#" + title not in hist: hist["h#" + title] = []
        hist["h#" + title] += rhad[pdgid]

    for pdgid in sorted(rlep):
        title = pdgid_mapping(pdgid)
        if "l#" + title not in hist: hist["l#" + title] = []
        hist["l#" + title] += rlep[pdgid]

    for tl in hist:
        th = TH1F()
        th.Title = tl.split("#")[1] + " (Hadronic Top)" if tl.startswith("h#") else tl.split("#")[1] + " (Leptonic Top)"
        th.xData = hist[tl]
        hist[tl] = th

    th = path(TH1F())
    th.Histograms = list(hist.values())
    th.Title  = "Fractional Energy Distribution of Truth Children from Mutual Top-Quark (Resonant)"
    th.xTitle = "Fraction of Energy being Dispersed to Truth Child ($E_{\\text{child}} / E_{\\text{top}}$)"
    th.yTitle = "Entries / 0.10"
    th.xBins  = 120
    th.xStep  = 0.2
    th.xMin   = 0
    th.xMax   = 1.2
    th.Stacked  = True
    th.Filename = "Figure.3.t"
    th.SaveFigure()

    hist = {}
    for pdgid in sorted(shad):
        title = pdgid_mapping(pdgid)
        if "h#" + title not in hist: hist["h#" + title] = []
        hist["h#" + title] += shad[pdgid]

    for pdgid in sorted(slep):
        title = pdgid_mapping(pdgid)
        if "l#" + title not in hist: hist["l#" + title] = []
        hist["l#" + title] += slep[pdgid]

    for tl in hist:
        th = TH1F()
        th.Title = tl.split("#")[1] + " (Hadronic Top)" if tl.startswith("h#") else tl.split("#")[1] + " (Leptonic Top)"
        th.xData = hist[tl]
        hist[tl] = th

    th = path(TH1F())
    th.Histograms = list(hist.values())
    th.Title  = "Fractional Energy Distribution of Truth Children from Mutual Top-Quark (Spectator)"
    th.xTitle = "Fraction of Energy being Dispersed to Truth Child ($E_{\\text{child}} / E_{\\text{top}}$)"
    th.yTitle = "Entries / 0.10"
    th.xBins  = 120
    th.xStep  = 0.2
    th.xMin   = 0
    th.xMax   = 1.2
    th.Stacked  = True
    th.Filename = "Figure.3.u"
    th.SaveFigure()

def ChildrenKinematics(ana, mp):
    kinematics_pt(ana, mp)
    kinematics_eta(ana)
    kinematics_phi(ana)
    kinematics_energy(ana, mp)
    kinematics_decay_mode(ana)
    dr_clustering(ana)
    fractional(ana)
