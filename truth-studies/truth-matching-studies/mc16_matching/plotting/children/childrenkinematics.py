from AnalysisG.Plotting import TH1F, TH2F
global figure_path

def settings():
    settings = {
            "Style" : "ROOT",
            "OutputDirectory" : figure_path + "children-kinematics/figures",
            "Histograms" : [],
            "Histogram" : None,
            "LegendLoc" : "upper right"
    }
    return settings

def kinematics_pt(ana):
    rkin = ana.res_kinematics
    skin = ana.spec_kinematics

    rth_pt = TH1F()
    rth_pt.Title = "Resonance"
    rth_pt.xData = rkin["pt"]

    sth_pt = TH1F()
    sth_pt.Title = "Spectator"
    sth_pt.xData = skin["pt"]

    sett = settings()
    th = TH1F(**sett)
    th.Histograms = [rth_pt, sth_pt]
    th.Title = "Transverse Momenta of Truth-Children from Spectator and Resonant Tops"
    th.xTitle = "Transverse Momenta (GeV)"
    th.yTitle = "Entries <unit>"
    th.xBins = 100
    th.xStep = 100
    th.xMin = 0
    th.xMax = 800
    th.yLogarithmic = True
    th.Filename = "Figure.5.a"
    th.SaveFigure()

    hist = []
    for pdgid in sorted(ana.res_pdgid_kinematics):
        th = TH1F()
        th.Title = pdgid
        th.xData = ana.res_pdgid_kinematics[pdgid]["pt"]
        hist.append(th)

    sett = settings()
    th = TH1F(**sett)
    th.Histograms = hist
    th.Title = "Transverse Momenta of Truth-Children from Resonant Tops"
    th.xTitle = "Transverse Momenta (GeV)"
    th.yTitle = "Entries <unit>"
    th.xBins = 100
    th.xStep = 100
    th.xMin = 0
    th.xMax = 800
    th.yScaling = 10
    th.xScaling = 20
    th.FontSize = 20
    th.LabelSize = 20
    th.OverFlow = True
    th.Stack = True
    th.yLogarithmic = True
    th.Filename = "Figure.5.b"
    th.SaveFigure()


    hist = []
    for pdgid in sorted(ana.spec_pdgid_kinematics):
        th = TH1F()
        th.Title = pdgid
        th.xData = ana.spec_pdgid_kinematics[pdgid]["pt"]
        hist.append(th)

    sett = settings()
    th = TH1F(**sett)
    th.Histograms = hist
    th.Title = "Transverse Momenta of Truth-Children from Spectator Tops"
    th.xTitle = "Transverse Momenta (GeV)"
    th.yTitle = "Entries <unit>"
    th.xBins = 100
    th.xStep = 100
    th.xMin = 0
    th.xMax = 800
    th.yScaling = 10
    th.xScaling = 20
    th.FontSize = 20
    th.LabelSize = 20
    th.OverFlow = True
    th.Stack = True
    th.yLogarithmic = True
    th.Filename = "Figure.5.c"
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

    sett = settings()
    th = TH1F(**sett)
    th.Histograms = [rth_pt, sth_pt]
    th.Title = "Pseudorapidity ($\\eta$) Truth-Children from Spectator and Resonant Tops"
    th.xTitle = "Pseudorapidity ($\\eta$) (arb.)"
    th.yTitle = "Entries"
    th.xBins = 100
    th.xStep = 1
    th.xMin = -6
    th.xMax =  6
    th.yLogarithmic = True
    th.Filename = "Figure.5.d"
    th.SaveFigure()

    hist = []
    for pdgid in sorted(ana.res_pdgid_kinematics):
        th = TH1F()
        th.Title = pdgid
        th.xData = ana.res_pdgid_kinematics[pdgid]["eta"]
        hist.append(th)

    sett = settings()
    th = TH1F(**sett)
    th.Histograms = hist
    th.Title = "Pseudorapidity ($\\eta$) Truth-Children from Resonant Tops"
    th.xTitle = "Pseudorapidity ($\\eta$) (arb.)"
    th.yTitle = "Entries"
    th.xBins = 100
    th.xStep = 1
    th.xMin = -6
    th.xMax =  6
    th.yScaling = 10
    th.xScaling = 20
    th.FontSize = 20
    th.LabelSize = 20
    th.Stack = True
    th.OverFlow = True
    th.yLogarithmic = True
    th.Filename = "Figure.5.e"
    th.SaveFigure()


    hist = []
    for pdgid in sorted(ana.spec_pdgid_kinematics):
        th = TH1F()
        th.Title = pdgid
        th.xData = ana.spec_pdgid_kinematics[pdgid]["eta"]
        hist.append(th)

    sett = settings()
    th = TH1F(**sett)
    th.Histograms = hist
    th.Title = "Pseudorapidity ($\\eta$) Truth-Children from Spectator Tops"
    th.xTitle = "Pseudorapidity ($\\eta$) (arb.)"
    th.yTitle = "Entries"
    th.xBins = 100
    th.xStep = 1
    th.xMin = -6
    th.xMax =  6
    th.yScaling = 10
    th.xScaling = 20
    th.FontSize = 20
    th.LabelSize = 20
    th.Stack = True
    th.OverFlow = True
    th.yLogarithmic = True
    th.Filename = "Figure.5.f"
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

    sett = settings()
    th = TH1F(**sett)
    th.Histograms = [rth_pt, sth_pt]
    th.Title = "Azimuthal Angle ($\\phi$) Truth-Children from Spectator and Resonant Tops"
    th.xTitle = "Azimuthal Angle ($\\phi$) (rad)"
    th.yTitle = "Entries"
    th.xBins = 100
    th.xStep = 1
    th.xMin = -3.5
    th.xMax =  3.5
    th.Filename = "Figure.5.g"
    th.SaveFigure()

    hist = []
    for pdgid in sorted(ana.res_pdgid_kinematics):
        th = TH1F()
        th.Title = pdgid
        th.xData = ana.res_pdgid_kinematics[pdgid]["phi"]
        hist.append(th)

    sett = settings()
    th = TH1F(**sett)
    th.Histograms = hist
    th.Title = "Azimuthal Angle ($\\phi$) Truth-Children from Resonant Tops"
    th.xTitle = "Azimuthal Angle ($\\phi$) (rad)"
    th.yTitle = "Entries"
    th.xBins = 100
    th.xStep = 1
    th.xMin = -3.5
    th.xMax =  3.5
    th.yScaling = 10
    th.xScaling = 20
    th.FontSize = 20
    th.LabelSize = 20
    th.Stack = True
    th.OverFlow = True
    th.Filename = "Figure.5.h"
    th.SaveFigure()


    hist = []
    for pdgid in sorted(ana.spec_pdgid_kinematics):
        th = TH1F()
        th.Title = pdgid
        th.xData = ana.spec_pdgid_kinematics[pdgid]["phi"]
        hist.append(th)

    sett = settings()
    th = TH1F(**sett)
    th.Histograms = hist
    th.Title = "Azimuthal Angle ($\\phi$) Truth-Children from Spectator Tops"
    th.xTitle = "Azimuthal Angle ($\\phi$) (rad)"
    th.yTitle = "Entries"
    th.xBins = 100
    th.xStep = 1
    th.xMin = -3.5
    th.xMax =  3.5
    th.yScaling = 10
    th.xScaling = 20
    th.FontSize = 20
    th.LabelSize = 20
    th.Stack = True
    th.OverFlow = True
    th.Filename = "Figure.5.i"
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

    sett = settings()
    th = TH1F(**sett)
    th.Histograms = [rth_pt, sth_pt]
    th.Title = "Energy of Truth-Children from Spectator and Resonant Tops"
    th.xTitle = "Energy (GeV)"
    th.yTitle = "Entries <unit>"
    th.xBins = 100
    th.xStep = 100
    th.xMin = 0
    th.xMax = 800
    th.yLogarithmic = True
    th.Filename = "Figure.5.j"
    th.SaveFigure()

    hist = []
    for pdgid in sorted(ana.res_pdgid_kinematics):
        th = TH1F()
        th.Title = pdgid
        th.xData = ana.res_pdgid_kinematics[pdgid]["energy"]
        hist.append(th)

    sett = settings()
    th = TH1F(**sett)
    th.Histograms = hist
    th.Title = "Energy of Truth-Children from Resonant Tops"
    th.xTitle = "Energy (GeV)"
    th.yTitle = "Entries <unit>"
    th.xBins = 100
    th.xStep = 100
    th.xMin = 0
    th.xMax = 800
    th.yScaling = 10
    th.xScaling = 20
    th.FontSize = 20
    th.LabelSize = 20
    th.OverFlow = True
    th.Stack = True
    th.yLogarithmic = True
    th.Filename = "Figure.5.k"
    th.SaveFigure()


    hist = []
    for pdgid in sorted(ana.spec_pdgid_kinematics):
        th = TH1F()
        th.Title = pdgid
        th.xData = ana.spec_pdgid_kinematics[pdgid]["energy"]
        hist.append(th)

    sett = settings()
    th = TH1F(**sett)
    th.Histograms = hist
    th.Title = "Energy of Truth-Children from Spectator Tops"
    th.xTitle = "Energy (GeV)"
    th.yTitle = "Entries <unit>"
    th.xBins = 100
    th.xStep = 100
    th.xMin = 0
    th.xMax = 800
    th.yScaling = 10
    th.xScaling = 20
    th.FontSize = 20
    th.LabelSize = 20
    th.OverFlow = True
    th.Stack = True
    th.yLogarithmic = True
    th.Filename = "Figure.5.l"
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


    sett = settings()
    th = TH1F(**sett)
    th.Histograms = hist_pt
    th.Title = "Transverse Momenta of Resonant and Spectator Top-Quark Children Segmented into Decay Channels"
    th.xTitle = "Transverse Momenta (GeV)"
    th.yTitle = "Entries <unit>"
    th.xBins = 100
    th.xStep = 100
    th.xMin = 0
    th.xMax = 800
    th.yScaling = 10
    th.xScaling = 20
    th.FontSize = 20
    th.LabelSize = 20
    th.Stack = True
    th.OverFlow = True
    th.yLogarithmic = True
    th.Filename = "Figure.5.m"
    th.SaveFigure()

    sett = settings()
    th = TH1F(**sett)
    th.Histograms = hist_eta
    th.Title = "Pseudorapidity ($\\eta$) of Resonant and Spectator Top-Quark Children Segmented into Decay Channels"
    th.xTitle = "Pseudorapidity ($\\eta$) (arb.)"
    th.yTitle = "Entries"
    th.xBins = 100
    th.xStep = 1
    th.xMin = -6
    th.xMax =  6
    th.yScaling = 10
    th.xScaling = 20
    th.FontSize = 20
    th.LabelSize = 20
    th.Stack = True
    th.OverFlow = True
    th.yLogarithmic = True
    th.Filename = "Figure.5.n"
    th.SaveFigure()

    sett = settings()
    th = TH1F(**sett)
    th.Histograms = hist_phi
    th.Title = "Azimuthal Angle ($\\phi$) of Resonant and Spectator Top-Quark Children Segmented into Decay Channels"
    th.xTitle = "Azimuthal Angle ($\\phi$) (rad)"
    th.yTitle = "Entries"
    th.xBins = 100
    th.xStep = 1
    th.xMin = -3.5
    th.xMax =  3.5
    th.yScaling = 10
    th.xScaling = 20
    th.FontSize = 20
    th.LabelSize = 20
    th.Stack = True
    th.OverFlow = True
    th.Filename = "Figure.5.o"
    th.SaveFigure()

    sett = settings()
    th = TH1F(**sett)
    th.Histograms = hist_energy
    th.Title = "Energy of Resonant and Spectator Top-Quark Children Segmented into Decay Channels"
    th.xTitle = "Energy (GeV)"
    th.yTitle = "Entries <unit>"
    th.xBins = 100
    th.xStep = 100
    th.xMin = 0
    th.xMax = 800
    th.yScaling = 10
    th.xScaling = 20
    th.FontSize = 20
    th.LabelSize = 20
    th.OverFlow = True
    th.Stack = True
    th.yLogarithmic = True
    th.Filename = "Figure.5.p"
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

    sett = settings()
    th = TH1F(**sett)
    th.Histograms = hist
    th.Title = "$\\Delta$R between Truth Children From (Non)-Mutual Top-Quarks"
    th.xTitle = "$\\Delta$R (arb.)"
    th.yTitle = "Entries"
    th.xBins = 200
    th.xStep = 0.5
    th.xMin = 0
    th.xMax = 6
    th.yScaling = 10
    th.xScaling = 20
    th.FontSize = 20
    th.LabelSize = 20
    th.Stack = True
    th.yLogarithmic = True
    th.Filename = "Figure.5.q"
    th.SaveFigure()

    hist = []
    for mode in modes:
        mod, title = mode

        th = TH1F()
        th.Title = title
        th.xData = ana.mass_clustering[mod]
        hist.append(th)

    sett = settings()
    th = TH1F(**sett)
    th.Histograms = hist
    th.Title = "Invariant Mass of Summed Truth Children From (Non)-Mutual Top-Quarks (pairs)"
    th.xTitle = "Invariant Mass of Resonance (GeV)"
    th.yTitle = "Entries"
    th.xBins = 500
    th.xStep = 200
    th.xMin = 0
    th.xMax = 1500
    th.yScaling = 10
    th.xScaling = 20
    th.FontSize = 20
    th.LabelSize = 20
    th.Stack = True
    th.yLogarithmic = True
    th.Filename = "Figure.5.r"
    th.SaveFigure()

    th2 = TH2F()
    th2.Title = "Invariant Mass of Summed Truth Children with Respect to $\\Delta$R between Adjacent Truth Children"
    th2.xData = sum(ana.dr_clustering.values(), [])
    th2.yData = sum(ana.mass_clustering.values(), [])

    th2.xTitle = "$\\Delta$R (arb.)"
    th2.yTitle = "Invariant Mass (GeV)"
    th2.Filename = "Figure.5.s"

    th2.xBins = 500
    th2.yBins = 1000

    th2.xMin = 0
    th2.yMin = 0

    th2.xMax = 6
    th2.yMax = 1500
    th2.yOverFlow = True
    th2.xOverFlow = True

    th2.OutputDirectory = figure_path + "children-kinematics/figures"
    th2.SaveFigure()


    th2 = TH2F()
    th2.Title = "$\\Delta$R between Truth Children and Top-Quark Transverse Momenta"
    th2.xData = sum(ana.dr_clustering.values(), [])
    th2.yData = sum(ana.top_pt_clustering.values(), [])

    th2.xTitle = "$\\Delta$R (arb.)"
    th2.yTitle = "Top-Quark Tranverse Momenta (GeV)"
    th2.Filename = "Figure.5.t"

    th2.xBins = 500
    th2.yBins = 1000

    th2.xMin = 0
    th2.yMin = 0

    th2.xMax = 6
    th2.yMax = 1000
    th2.yOverFlow = True
    th2.xOverFlow = True

    th2.OutputDirectory = figure_path + "children-kinematics/figures"
    th2.SaveFigure()


    th2 = TH2F()
    th2.Title = "$\\Delta$R between Adjacent Truth Children and Top-Quark Energy"
    th2.xData = sum(ana.dr_clustering.values(), [])
    th2.yData = sum(ana.top_energy_clustering.values(), [])

    th2.xTitle = "$\\Delta$R (arb.)"
    th2.yTitle = "Top-Quark Energy (GeV)"
    th2.Filename = "Figure.5.u"

    th2.xBins = 500
    th2.yBins = 1000

    th2.xMin = 0
    th2.yMin = 0

    th2.xMax = 6
    th2.yMax = 1000
    th2.yOverFlow = True
    th2.xOverFlow = True

    th2.OutputDirectory = figure_path + "children-kinematics/figures"
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

    sett = settings()
    th = TH1F(**sett)
    th.Histograms = hist
    th.Title = "$\\Delta$R between Truth-Top and Associated Mutal Children"
    th.xTitle = "$\\Delta$R (arb.)"
    th.yTitle = "Entries"
    th.xBins = 200
    th.xStep = 0.5
    th.xMin = 0
    th.xMax = 6
    th.yScaling = 10
    th.xScaling = 20
    th.FontSize = 20
    th.LabelSize = 20
    th.Stack = True
    th.yLogarithmic = True
    th.Filename = "Figure.5.v"
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

    sett = settings()
    th = TH1F(**sett)
    th.Histograms = hist
    th.Title = "Fractional Tranverse Momenta Distribution of Truth Children from Mutual Top-Quark (Resonant)"
    th.xTitle = "Fraction (arb.)"
    th.yTitle = "Entries"
    th.xBins = 100
    th.xStep = 0.1
    th.xMin = 0
    th.xMax = 5
    th.yScaling = 10
    th.xScaling = 20
    th.FontSize = 20
    th.LabelSize = 20
    th.Stack = True
    th.yLogarithmic = True
    th.Filename = "Figure.5.w"
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

    sett = settings()
    th = TH1F(**sett)
    th.Histograms = hist
    th.Title = "Fractional Tranverse Momenta Distribution of Truth Children from Mutual Top-Quark (Spectator)"
    th.xTitle = "Fraction (arb.)"
    th.yTitle = "Entries"
    th.xBins = 100
    th.xStep = 0.2
    th.xMin = 0
    th.xMax = 5
    th.yScaling = 10
    th.xScaling = 20
    th.FontSize = 20
    th.LabelSize = 20
    th.Stack = True
    th.yLogarithmic = True
    th.Filename = "Figure.5.x"
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

    sett = settings()
    th = TH1F(**sett)
    th.Histograms = hist
    th.Title = "Fractional Energy Distribution of Truth Children from Mutual Top-Quark (Resonant)"
    th.xTitle = "Fraction (arb.)"
    th.yTitle = "Entries"
    th.xBins = 100
    th.xStep = 0.2
    th.xMin = 0
    th.xMax = 1
    th.yScaling = 10
    th.xScaling = 20
    th.FontSize = 20
    th.LabelSize = 20
    th.Stack = True
    th.yLogarithmic = True
    th.Filename = "Figure.5.y"
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

    sett = settings()
    th = TH1F(**sett)
    th.Histograms = hist
    th.Title = "Fractional Energy Distribution of Truth Children from Mutual Top-Quark (Spectator)"
    th.xTitle = "Fraction (arb.)"
    th.yTitle = "Entries"
    th.xBins = 100
    th.xStep = 0.1
    th.xMin = 0
    th.xMax = 1
    th.yScaling = 10
    th.xScaling = 20
    th.FontSize = 20
    th.LabelSize = 20
    th.Stack = True
    th.yLogarithmic = True
    th.Filename = "Figure.5.z"
    th.SaveFigure()

def ChildrenKinematics(ana):
#    kinematics_pt(ana)
#    kinematics_eta(ana)
#    kinematics_phi(ana)
#    kinematics_energy(ana)
#    kinematics_decay_mode(ana)
    dr_clustering(ana)
#    fractional(ana)
