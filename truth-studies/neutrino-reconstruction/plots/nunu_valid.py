from AnalysisG.Plotting import TH1F, TH2F
global figure_path

def settings():
    settings = {
            "Style" : "ROOT",
            "OutputDirectory" : figure_path + "neutrino-studies/double-neutrino/figures/",
            "Histograms" : [],
            "Histogram" : None,
            "FontSize"  : 22,
            "LabelSize" : 20,
            "xScaling"  : 10,
            "yScaling"  : 12,
            "LegendLoc" : "upper right"
    }
    return settings

def settings_th2f():
    settings = {
            "Style" : "ROOT",
            "OutputDirectory" : figure_path + "neutrino-studies/double-neutrino/figures/",
            "FontSize"  : 22,
            "LabelSize" : 20,
            "xScaling"  : 10,
            "yScaling"  : 12
    }
    return settings



def momentum_template(fign, figures_, pairs, titles, datax, datay):
    for i in range(len(figures_)):
        title = "$\\Delta P_{" + pairs[i][0] + "," + pairs[i][1] + "}$ "
        title += "Momenta Components between \n Truth and Reconstructed Neutrino Pair (" + titles[i] + ")"
        sett = settings_th2f()
        th = TH2F(**sett)
        th.Title = title
        th.xBins = 500
        th.yBins = 500

        th.xMin = -1000
        th.yMin = -1000

        th.xMax = 1000
        th.yMax = 1000

        th.yOverFlow = True
        th.xOverFlow = True

        th.xData = datax[i]
        th.xTitle = "$\\Delta P_{" + pairs[i][0] + "}$-Momenta of truth and pyc neutrino (GeV)"

        th.yData = datay[i]
        th.yTitle = "$\\Delta P_{" + pairs[i][1] + "}$-Momenta of truth and pyc neutrino (GeV)"

        th.Filename = "Figure." + fign + figures_[i]
        th.Color = "tab20c"
        th.SaveFigure()


def neutrino_kinematic_pyc(delta, fign):
    titles = ["Neutrino-1", "Neutrino-2", "all"]

    figures_ = [".1.a", ".2.a", ".3.a"]
    pairs = [("x", "y"), ("x", "y"), ("x", "y")]
    datax = [delta["nu1"]["px"], delta["nu2"]["px"], delta["nu1"]["px"] + delta["nu2"]["px"]]
    datay = [delta["nu1"]["py"], delta["nu2"]["py"], delta["nu1"]["py"] + delta["nu2"]["py"]]
    momentum_template(fign, figures_, pairs, titles, datax, datay)

    figures_ = [".1.b", ".2.b", ".3.b"]
    pairs = [("x", "z"), ("x", "z"), ("x", "z")]
    datax = [delta["nu1"]["px"], delta["nu2"]["px"], delta["nu1"]["px"] + delta["nu2"]["px"]]
    datay = [delta["nu1"]["pz"], delta["nu2"]["pz"], delta["nu1"]["pz"] + delta["nu2"]["pz"]]
    momentum_template(fign, figures_, pairs, titles, datax, datay)


    figures_ = [".1.c", ".2.c", ".3.c"]
    pairs = [("y", "z"), ("y", "z"), ("y", "z")]
    datax = [delta["nu1"]["py"], delta["nu2"]["py"], delta["nu1"]["py"] + delta["nu2"]["py"]]
    datay = [delta["nu1"]["pz"], delta["nu2"]["pz"], delta["nu1"]["pz"] + delta["nu2"]["pz"]]
    momentum_template(fign, figures_, pairs, titles, datax, datay)


def neutrino_kinematic_ref(delta, fign):
    titles = ["Neutrino-1", "Neutrino-2", "all"]

    figures_ = [".1.d", ".2.d", ".3.d"]
    pairs = [("x", "y"), ("x", "y"), ("x", "y")]
    datax = [delta["nu1"]["px"], delta["nu2"]["px"], delta["nu1"]["px"] + delta["nu2"]["px"]]
    datay = [delta["nu1"]["py"], delta["nu2"]["py"], delta["nu1"]["py"] + delta["nu2"]["py"]]
    momentum_template(fign, figures_, pairs, titles, datax, datay)

    figures_ = [".1.e", ".2.e", ".3.e"]
    pairs = [("x", "z"), ("x", "z"), ("x", "z")]
    datax = [delta["nu1"]["px"], delta["nu2"]["px"], delta["nu1"]["px"] + delta["nu2"]["px"]]
    datay = [delta["nu1"]["pz"], delta["nu2"]["pz"], delta["nu1"]["pz"] + delta["nu2"]["pz"]]
    momentum_template(fign, figures_, pairs, titles, datax, datay)

    figures_ = [".1.f", ".2.f", ".3.f"]
    pairs = [("y", "z"), ("y", "z"), ("y", "z")]
    datax = [delta["nu1"]["py"], delta["nu2"]["py"], delta["nu1"]["py"] + delta["nu2"]["py"]]
    datay = [delta["nu1"]["pz"], delta["nu2"]["pz"], delta["nu1"]["pz"] + delta["nu2"]["pz"]]
    momentum_template(fign, figures_, pairs, titles, datax, datay)


def projection_plots(data_pyc, data_ref, fign):
    for j, i, k in zip(["px", "py", "pz", "e"], ["g", "h", "i", "j"], ["$P_x$", "$P_y$", "$P_z$", "Energy"]):
        sett = settings()
        th1 = TH1F()
        th1.Title = "pyc"
        th1.xData = data_pyc["nu1"][j] + data_pyc["nu2"][j]

        th2 = TH1F()
        th2.Title = "reference"
        th2.xData = data_ref["nu1"][j] + data_ref["nu2"][j]

        sett["Histograms"] += [th1, th2]

        th_ = TH1F(**sett)
        th_.Title = "Kinematic Differential between Truth and PYC/Reference \n Implementation for: " + k
        th_.xTitle = "$\\Delta$-" + k + " (GeV)"
        th_.yTitle = "Entries <unit>"
        th_.Filename = "Figure."+ fign + "." + i
        th_.xBins = 200
        th_.xMin  = -1000
        th_.xMax  = 1000
        th_.xStep = 200
        th_.SaveFigure()

def top_mass(data, fign):
    sett = settings()
    th1 = TH1F()
    th1.Title = "pyc"
    th1.xData = data["nu1"]["pyc"] + data["nu2"]["pyc"]

    th2 = TH1F()
    th2.Title = "reference"
    th2.xData = data["nu1"]["reference"] + data["nu2"]["reference"]

    th3 = TH1F()
    th3.Title = "truth"
    th3.xData = data["nu1"]["truth"] + data["nu2"]["truth"]
    th3.Alpha = 0.1
    sett["Histograms"] += [th1, th2, th3]

    th_ = TH1F(**sett)
    th_.Title = "Top-Mass using Truth and PYC/Reference Implementation"
    th_.xTitle = "Invariant Top-Mass (GeV)"
    th_.yTitle = "Entries <unit>"
    th_.Filename = "Figure."+ fign + ".k"
    th_.xBins = 400
    th_.xMin  = 0
    th_.xMax  = 400
    th_.xStep = 40
    th_.SaveFigure()




def DoubleNeutrinoReconstruction(ana):
    neutrino_kinematic_pyc(ana.children_kinematic_delta_pyc, "1")
    neutrino_kinematic_ref(ana.children_kinematic_delta_ref, "1")
    projection_plots(ana.children_kinematic_delta_pyc, ana.children_kinematic_delta_ref, "1")
    top_mass(ana.children_top_mass_diff, "1")

    neutrino_kinematic_pyc(ana.truthjet_kinematic_delta_pyc, "2")
    neutrino_kinematic_ref(ana.truthjet_kinematic_delta_ref, "2")
    projection_plots(ana.truthjet_kinematic_delta_pyc, ana.truthjet_kinematic_delta_ref, "2")
    top_mass(ana.truthjet_top_mass_diff, "2")

    neutrino_kinematic_pyc(ana.jet_kinematic_delta_pyc, "3")
    neutrino_kinematic_ref(ana.jet_kinematic_delta_ref, "3")
    projection_plots(ana.jet_kinematic_delta_pyc, ana.jet_kinematic_delta_ref, "3")
    top_mass(ana.jet_top_mass_diff, "3")

    neutrino_kinematic_pyc(ana.reco_lep_jet_kinematic_delta_pyc, "4")
    neutrino_kinematic_ref(ana.reco_lep_jet_kinematic_delta_ref, "4")
    projection_plots(ana.reco_lep_jet_kinematic_delta_pyc, ana.reco_lep_jet_kinematic_delta_ref, "4")
    top_mass(ana.reco_lep_jet_top_mass_diff, "4")
