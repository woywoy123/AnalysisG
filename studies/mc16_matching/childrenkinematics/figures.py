from AnalysisG.core.plotting import TH1F, TH2F

global figure_path
global mass_point

def path(hist):
    hist.Style = "ATLAS"
    hist.OutputDirectory = figure_path + "/childrenkinematics/" + mass_point
    hist.Overflow = True
    return hist

def pdgid_mapping(pdgid):
    if "nu" in pdgid: title = "$\\nu_{\\ell}$"
    elif "e" in pdgid or "tau" in pdgid or "mu" in pdgid: title = "$\\ell$"
    elif "u" in pdgid or "c" in pdgid or "d" in pdgid or "s" in pdgid: title = "light-quark"
    else: title = pdgid
    return title

def kinematics_pt(ana):
    rkin = ana.res_kinematics
    skin = ana.spec_kinematics

    rth_pt = TH1F()
    rth_pt.Title = "Resonance"
    rth_pt.xData = rkin["pt"]

    sth_pt = TH1F()
    sth_pt.Title = "Spectator"
    sth_pt.xData = skin["pt"]

    th = path(TH1F())
    th.Histograms = [rth_pt, sth_pt]
    th.Title = "Normalized $p_T$ Distributions for \n Spectator and Resonant Top Children"
    th.xTitle = "$p_T$ of Top Children (GeV)"
    th.yTitle = "Density (Arb.) / ($10$ GeV)"
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

    for k in hist:
        th = TH1F()
        th.Title = k
        th.xData = hist[k]
        hist[k] = th

    th = path(TH1F())
    th.Histograms = list(hist.values())
    th.Title  = "Normalized $p_T$ Distributions for Top Children \n from Resonant Top based on Parton Type"
    th.xTitle = "$p_T$ of Top Children (GeV)"
    th.yTitle = "Density (Arb.) / ($10$ GeV)"
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

    for k in hist:
        th = TH1F()
        th.Title = k
        th.xData = hist[k]
        hist[k] = th

    th = path(TH1F())
    th.Histograms = list(hist.values())
    th.Title  = "Normalized $p_T$ Distribution for Top Children \n from Spectator Top based on Parton Type"
    th.xTitle = "$p_T$ of Top Children (GeV)"
    th.yTitle = "Density (Arb.) / ($10$ GeV)"
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
    sth_pt.Title = "Spectator"
    sth_pt.xData = skin["eta"]

    th = path(TH1F())
    th.Histograms = [rth_pt, sth_pt]
    th.Title = "Normalized Pseudorapidity ($\\eta$) Distribution for \n Spectator and Resonance Top Children"
    th.xTitle = "Pseudorapidity ($\\eta$) (Arb.)"
    th.yTitle = "Density (Arb.) / ($0.5$)"
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

    for k in hist:
        th = TH1F()
        th.Title = k
        th.xData = hist[k]
        hist[k] = th


    th = path(TH1F())
    th.Histograms = list(hist.values())
    th.Title  = "Normalized Pseudorapidity ($\\eta$) Distributions for Top Children \n from Resonant Top based on Parton Type"
    th.xTitle = "Pseudorapidity ($\\eta$) of Top Children (Arb.)"
    th.yTitle = "Density (Arb.) / ($0.5$)"
    th.xBins = 240
    th.xStep = 1
    th.xMin = -6
    th.xMax =  6
    th.Stacked = True
    th.Density = True
#    th.yLogarithmic = True
    th.Filename = "Figure.3.e"
    th.SaveFigure()

    hist = {}
    for pdgid in sorted(ana.spec_pdgid_kinematics):
        title = pdgid_mapping(pdgid)
        if title not in hist: hist[title] = []
        hist[title] += ana.spec_pdgid_kinematics[pdgid]["eta"]

    for k in hist:
        th = TH1F()
        th.Title = k
        th.xData = hist[k]
        hist[k] = th


    th = path(TH1F())
    th.Histograms = list(hist.values())
    th.Title  = "Normalized Pseudorapidity ($\\eta$) Distributions for Top Children \n from Spectator Top based on Parton Type"
    th.xTitle = "Pseudorapidity ($\\eta$) of Top Children (Arb.)"
    th.yTitle = "Density (Arb.) / ($0.5$)"
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
    sth_pt.Title = "Spectator"
    sth_pt.xData = skin["phi"]

    th = path(TH1F())
    th.Histograms = [rth_pt, sth_pt]
    th.Title = "Normalized Azimuthal Angle ($\\phi$) Distributions for \n Spectator and Resonant Top Children"
    th.xTitle = "Azimuthal Angle ($\\phi$) (rad)"
    th.yTitle = "Density (Abr.) / ($0.1$)"
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

    for k in hist:
        th = TH1F()
        th.Title = k
        th.xData = hist[k]
        hist[k] = th

    th = path(TH1F())
    th.Histograms = list(hist.values())
    th.Title = "Normalized Azimuthal Angle ($\\phi$) Distributions for Top Children \n from Resonant Top based on Parton Type"
    th.xTitle = "Azimuthal Angle ($\\phi$) (rad)"
    th.yTitle = "Density (Abr.) / ($0.1$)"
    th.xBins = 350
    th.xStep = 1
    th.xMin = -3.5
    th.xMax =  3.5
    th.Stacked = True
    th.Density = True
    th.Filename = "Figure.3.h"
    th.SaveFigure()

    exit()

    hist = []
    for pdgid in sorted(ana.spec_pdgid_kinematics):
        th = TH1F()
        th.Title = pdgid
        th.xData = ana.spec_pdgid_kinematics[pdgid]["phi"]
        hist.append(th)

    th = path(TH1F())
    th.Histograms = hist
    th.Title = "Azimuthal Angle ($\\phi$) Truth-Children from Spectator Tops"
    th.xTitle = "Azimuthal Angle ($\\phi$) (rad)"
    th.yTitle = "Entries (Arb.)"
    th.xBins = 100
    th.xStep = 1
    th.xMin = -3.5
    th.xMax =  3.5
    th.Stacked = True
    th.Filename = "Figure.3.i"
    th.SaveFigure()

def kinematics_energy(ana):
    rkin = ana.res_kinematics
    skin = ana.spec_kinematics

    rth_pt = TH1F()
    rth_pt.Title = "Resonance"
    rth_pt.xData = rkin["energy"]

    sth_pt = TH1F()
    sth_pt.Title = "Spectator"
    sth_pt.xData = skin["energy"]

    th = path(TH1F())
    th.Histograms = [rth_pt, sth_pt]
    th.Title = "Energy of Truth-Children from Spectator and Resonant Tops"
    th.xTitle = "Energy (GeV)"
    th.yTitle = "Entries (Arb.)"
    th.xBins = 100
    th.xStep = 100
    th.xMin = 0
    th.xMax = 800
    th.yLogarithmic = True
    th.Filename = "Figure.3.j"
    th.SaveFigure()

    hist = []
    for pdgid in sorted(ana.res_pdgid_kinematics):
        th = TH1F()
        th.Title = pdgid
        th.xData = ana.res_pdgid_kinematics[pdgid]["energy"]
        hist.append(th)

    th = path(TH1F())
    th.Histograms = hist
    th.Title = "Energy of Truth-Children from Resonant Tops"
    th.xTitle = "Energy (GeV)"
    th.yTitle = "Entries (Arb.)"
    th.xBins = 100
    th.xStep = 100
    th.xMin = 0
    th.xMax = 800
    th.Stacked = True
    th.yLogarithmic = True
    th.Filename = "Figure.3.k"
    th.SaveFigure()


    hist = []
    for pdgid in sorted(ana.spec_pdgid_kinematics):
        th = TH1F()
        th.Title = pdgid
        th.xData = ana.spec_pdgid_kinematics[pdgid]["energy"]
        hist.append(th)

    th = path(TH1F())
    th.Histograms = hist
    th.Title = "Energy of Truth-Children from Spectator Tops"
    th.xTitle = "Energy (GeV)"
    th.yTitle = "Entries (Arb.)"
    th.xBins = 100
    th.xStep = 100
    th.xMin = 0
    th.xMax = 800
    th.Stacked = True
    th.yLogarithmic = True
    th.Filename = "Figure.3.l"
    th.SaveFigure()

def kinematics_decay_mode(ana):
    hist_pt = []
    hist_eta = []
    hist_phi = []
    hist_energy = []

    for mode in [["lep", "leptonic"], ["had", "hadronic"]]:
        mod, title = mode

        th = TH1F()
        th.Title = "resonance " + title
        th.xData = ana.res_decay_mode[mod]["pt"]
        hist_pt.append(th)

        th = TH1F()
        th.Title = "resonance " + title
        th.xData = ana.res_decay_mode[mod]["eta"]
        hist_eta.append(th)

        th = TH1F()
        th.Title = "resonance " + title
        th.xData = ana.res_decay_mode[mod]["phi"]
        hist_phi.append(th)

        th = TH1F()
        th.Title = "resonance " + title
        th.xData = ana.res_decay_mode[mod]["energy"]
        hist_energy.append(th)


        th = TH1F()
        th.Title = "spectator " + title
        th.xData = ana.spec_decay_mode[mod]["pt"]
        hist_pt.append(th)

        th = TH1F()
        th.Title = "spectator " + title
        th.xData = ana.spec_decay_mode[mod]["eta"]
        hist_eta.append(th)

        th = TH1F()
        th.Title = "spectator " + title
        th.xData = ana.spec_decay_mode[mod]["phi"]
        hist_phi.append(th)

        th = TH1F()
        th.Title = "spectator " + title
        th.xData = ana.spec_decay_mode[mod]["energy"]
        hist_energy.append(th)


    th = path(TH1F())
    th.Histograms = hist_pt
    th.Title = "Transverse Momenta of Resonant and Spectator Top-Quark \n Children Segmented into Decay Channels"
    th.xTitle = "Transverse Momenta (GeV)"
    th.yTitle = "Entries (Arb.)"
    th.xBins = 100
    th.xStep = 100
    th.xMin = 0
    th.xMax = 800
    th.Stacked = True
    th.yLogarithmic = True
    th.Filename = "Figure.3.m"
    th.SaveFigure()

    th = path(TH1F())
    th.Histograms = hist_eta
    th.Title = "Pseudorapidity ($\\eta$) of Resonant and Spectator Top-Quark \n Children Segmented into Decay Channels"
    th.xTitle = "Pseudorapidity ($\\eta$) (Arb.)"
    th.yTitle = "Entries (Arb.)"
    th.xBins = 100
    th.xStep = 1
    th.xMin = -6
    th.xMax =  6
    th.Stacked = True
    th.yLogarithmic = True
    th.Filename = "Figure.3.n"
    th.SaveFigure()

    th = path(TH1F())
    th.Histograms = hist_phi
    th.Title = "Azimuthal Angle ($\\phi$) of Resonant and Spectator Top-Quark \n Children Segmented into Decay Channels"
    th.xTitle = "Azimuthal Angle ($\\phi$) (rad)"
    th.yTitle = "Entries (Arb.)"
    th.xBins = 100
    th.xStep = 1
    th.xMin = -3.5
    th.xMax =  3.5
    th.Stacked = True
    th.Filename = "Figure.3.o"
    th.SaveFigure()

    th = path(TH1F())
    th.Histograms = hist_energy
    th.Title = "Energy of Resonant and Spectator Top-Quark Children Segmented into Decay Channels"
    th.xTitle = "Energy (GeV)"
    th.yTitle = "Entries (Arb.)"
    th.xBins = 100
    th.xStep = 100
    th.xMin = 0
    th.xMax = 800
    th.Stacked = True
    th.yLogarithmic = True
    th.Filename = "Figure.3.p"
    th.SaveFigure()

def dr_clustering(ana):
    modes = [
                ["CTRR", "Correct-Top-RR"],
                ["FTRR", "False-Top-RR"],
                ["CTSS", "Correct-Top-SS"],
                ["FTSS", "False-Top-SS"],
                ["FTRS", "False-Top-RS"]
    ]

    hist = []
    for mode in modes:
        mod, title = mode

        th = TH1F()
        th.Title = title
        th.xData = ana.dr_clustering[mod]
        hist.append(th)

    th = path(TH1F())
    th.Histograms = hist
    th.Title = "$\\Delta$R between Truth Children From (Non)-Mutual Top-Quarks"
    th.xTitle = "$\\Delta$R (Arb.)"
    th.yTitle = "Entries"
    th.xBins = 200
    th.xStep = 0.5
    th.xMin = 0
    th.xMax = 6
    th.Stacked = True
    th.yLogarithmic = True
    th.Filename = "Figure.3.q"
    th.SaveFigure()

    hist = []
    for mode in modes:
        mod, title = mode

        th = TH1F()
        th.Title = title
        th.xData = ana.mass_clustering[mod]
        hist.append(th)

    th = path(TH1F())
    th.Histograms = hist
    th.Title = "Invariant Mass of Summed Truth Children From (Non)-Mutual Top-Quarks (pairs)"
    th.xTitle = "Invariant Mass of Resonance (GeV)"
    th.yTitle = "Entries"
    th.xBins = 500
    th.xStep = 200
    th.xMin = 0
    th.xMax = 1500
    th.Stacked = True
    th.yLogarithmic = True
    th.Filename = "Figure.3.r"
    th.SaveFigure()

    th2 = path(TH2F())
    th2.Title = "Invariant Mass of Summed Truth Children with Respect to $\\Delta$R between Adjacent Truth Children"
    th2.xData = sum(ana.dr_clustering.values(), [])
    th2.yData = sum(ana.mass_clustering.values(), [])

    th2.xTitle = "$\\Delta$R (Arb.)"
    th2.yTitle = "Invariant Mass (GeV)"
    th2.Filename = "Figure.3.s"

    th2.xBins = 500
    th2.yBins = 1000

    th2.xMin = 0
    th2.yMin = 0

    th2.xMax = 6
    th2.yMax = 1500

    th2.SaveFigure()


    th2 = path(TH2F())
    th2.Title = "$\\Delta$R between Truth Children and Top-Quark Transverse Momenta"
    th2.xData = sum(ana.dr_clustering.values(), [])
    th2.yData = sum(ana.top_pt_clustering.values(), [])

    th2.xTitle = "$\\Delta$R (Arb.)"
    th2.yTitle = "Top-Quark Tranverse Momenta (GeV)"
    th2.Filename = "Figure.3.t"

    th2.xBins = 500
    th2.yBins = 1000

    th2.xMin = 0
    th2.yMin = 0

    th2.xMax = 6
    th2.yMax = 1000

    th2.SaveFigure()


    th2 = path(TH2F())
    th2.Title = "$\\Delta$R between Adjacent Truth Children and Top-Quark Energy"
    th2.xData = sum(ana.dr_clustering.values(), [])
    th2.yData = sum(ana.top_energy_clustering.values(), [])

    th2.xTitle = "$\\Delta$R (Arb.)"
    th2.yTitle = "Top-Quark Energy (GeV)"
    th2.Filename = "Figure.3.u"

    th2.xBins = 500
    th2.yBins = 1000

    th2.xMin = 0
    th2.yMin = 0

    th2.xMax = 6
    th2.yMax = 1000

    th2.SaveFigure()

    modes = [
                ["rlep", "Resonant Leptonic"],
                ["rhad", "Resonant Hadronic"],
                ["slep", "Spectator Leptonic"],
                ["shad", "Spectator Hadronic"],
    ]

    hist = []
    for mode in modes:
        mod, title = mode

        th = TH1F()
        th.Title = title
        th.xData = ana.top_children_dr[mod]
        hist.append(th)

    th = path(TH1F())
    th.Histograms = hist
    th.Title = "$\\Delta$R between Truth-Top and Associated Mutal Children"
    th.xTitle = "$\\Delta$R (Arb.)"
    th.yTitle = "Entries"
    th.xBins = 200
    th.xStep = 0.5
    th.xMin = 0
    th.xMax = 6
    th.Stacked = True
    th.yLogarithmic = True
    th.Filename = "Figure.3.v"
    th.SaveFigure()

def fractional(ana):
    rhad = ana.fractional["rhad-pt"]
    rlep = ana.fractional["rlep-pt"]
    shad = ana.fractional["shad-pt"]
    slep = ana.fractional["slep-pt"]

    hist = []
    for mode in rhad:
        th = TH1F()
        th.Title = mode + " (Hadronic)"
        th.xData = rhad[mode]
        hist.append(th)

    for mode in rlep:
        th = TH1F()
        th.Title = mode + " (Leptonic)"
        th.xData = rlep[mode]
        hist.append(th)

    th = path(TH1F())
    th.Histograms = hist
    th.Title = "Fractional Tranverse Momenta Distribution of Truth Children from Mutual Top-Quark \n (Resonant)"
    th.xTitle = "Fraction (Arb.)"
    th.yTitle = "Entries (Arb.)"
    th.xBins = 1000
    th.xStep = 0.5
    th.xMin = 0
    th.xMax = 5
    th.Stacked = True
    th.yLogarithmic = True
    th.Filename = "Figure.3.w"
    th.SaveFigure()

    hist = []
    for mode in shad:
        th = TH1F()
        th.Title = mode + " (Hadronic)"
        th.xData = shad[mode]
        hist.append(th)

    for mode in slep:
        th = TH1F()
        th.Title = mode + " (Leptonic)"
        th.xData = slep[mode]
        hist.append(th)

    th = path(TH1F())
    th.Histograms = hist
    th.Title = "Fractional Tranverse Momenta Distribution of Truth Children from Mutual Top-Quark \n (Spectator)"
    th.xTitle = "Fraction (Arb.)"
    th.yTitle = "Entries (Arb.)"
    th.xBins = 1000
    th.xStep = 0.5
    th.xMin = 0
    th.xMax = 5
    th.Stacked = True
    th.yLogarithmic = True
    th.Filename = "Figure.3.x"
    th.SaveFigure()

    rhad = ana.fractional["rhad-energy"]
    rlep = ana.fractional["rlep-energy"]
    shad = ana.fractional["shad-energy"]
    slep = ana.fractional["slep-energy"]

    hist = []
    for mode in rhad:
        th = TH1F()
        th.Title = mode + " (Hadronic)"
        th.xData = rhad[mode]
        hist.append(th)

    for mode in rlep:
        th = TH1F()
        th.Title = mode + " (Leptonic)"
        th.xData = rlep[mode]
        hist.append(th)

    th = path(TH1F())
    th.Histograms = hist
    th.Title = "Fractional Energy Distribution of Truth Children from Mutual Top-Quark (Resonant)"
    th.xTitle = "Fraction (Arb.)"
    th.yTitle = "Entries"
    th.xBins = 100
    th.xStep = 0.2
    th.xMin = 0
    th.xMax = 1
    th.Stacked = True
    th.yLogarithmic = True
    th.Filename = "Figure.3.y"
    th.SaveFigure()

    hist = []
    for mode in shad:
        th = TH1F()
        th.Title = mode + " (Hadronic)"
        th.xData = shad[mode]
        hist.append(th)

    for mode in slep:
        th = TH1F()
        th.Title = mode + " (Leptonic)"
        th.xData = slep[mode]
        hist.append(th)

    th = path(TH1F())
    th.Histograms = hist
    th.Title = "Fractional Energy Distribution of Truth Children from Mutual Top-Quark (Spectator)"
    th.xTitle = "Fraction (Arb.)"
    th.yTitle = "Entries"
    th.xBins = 100
    th.xStep = 0.1
    th.xMin = 0
    th.xMax = 1
    th.Stacked = True
    th.yLogarithmic = True
    th.Filename = "Figure.3.z"
    th.SaveFigure()

def ChildrenKinematics(ana):
    kinematics_pt(ana)
    kinematics_eta(ana)
    kinematics_phi(ana)
    #kinematics_energy(ana)
    #kinematics_decay_mode(ana)
    #dr_clustering(ana)
    #fractional(ana)
