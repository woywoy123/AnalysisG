from AnalysisG.Plotting import TH1F, TH2F
global figure_path

def settings():
    settings = {
            "Style" : "ROOT",
            "OutputDirectory" : figure_path + "neutrino-studies/single-neutrino/figures/",
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
            "OutputDirectory" : figure_path + "neutrino-studies/single-neutrino/figures/",
            "FontSize"  : 22,
            "LabelSize" : 20,
            "xScaling"  : 10,
            "yScaling"  : 12
    }
    return settings

def neutrino_kinematic_pyc(delta, fign):
    sett = settings_th2f()
    th = TH2F(**sett)
    th.Title = "$\\Delta P_{x, y}$ Momenta Components between \n Truth and Reconstructed Neutrinos"
    th.xBins = 500
    th.yBins = 500

    th.xMin = -1000
    th.yMin = -1000

    th.xMax = 1000
    th.yMax = 1000

    th.yOverFlow = True
    th.xOverFlow = True

    th.xData = delta["px"]
    th.xTitle = "$\\Delta P_{x}$-Momenta of truth and pyc neutrino (GeV)"

    th.yData = delta["py"]
    th.yTitle = "$\\Delta P_{y}$-Momenta of truth and pyc neutrino (GeV)"

    th.Filename = "Figure." + fign + ".a"
    th.Color = "tab20c"
    th.SaveFigure()


    sett = settings_th2f()
    th = TH2F(**sett)
    th.Title = "$\\Delta P_{x, z}$ Momenta Components between \n Truth and Reconstructed Neutrinos"
    th.xBins = 500
    th.yBins = 500

    th.xMin = -1000
    th.yMin = -1000

    th.xMax = 1000
    th.yMax = 1000

    th.yOverFlow = True
    th.xOverFlow = True

    th.xData = delta["px"]
    th.xTitle = "$\\Delta P_{x}$-Momenta of truth and pyc neutrino (GeV)"

    th.yData = delta["pz"]
    th.yTitle = "$\\Delta P_{z}$-Momenta of truth and pyc neutrino (GeV)"

    th.Filename = "Figure." + fign + ".b"
    th.Color = "tab20c"
    th.SaveFigure()


    sett = settings_th2f()
    th = TH2F(**sett)
    th.Title = "$\\Delta P_{y, z}$ Momenta Components between \n Truth and Reconstructed Neutrinos"
    th.xBins = 500
    th.yBins = 500

    th.xMin = -1000
    th.yMin = -1000

    th.xMax = 1000
    th.yMax = 1000

    th.yOverFlow = True
    th.xOverFlow = True

    th.xData = delta["py"]
    th.xTitle = "$\\Delta P_{y}$-Momenta of truth and pyc neutrino (GeV)"

    th.yData = delta["pz"]
    th.yTitle = "$\\Delta P_{z}$-Momenta of truth and pyc neutrino (GeV)"

    th.Filename = "Figure." + fign + ".c"
    th.Color = "tab20c"
    th.SaveFigure()


def neutrino_kinematic_ref(delta, fign):
    sett = settings_th2f()
    th = TH2F(**sett)
    th.Title = "$\\Delta P_{x, y}$ Momenta Components between \n Truth and Reconstructed Neutrinos"
    th.xBins = 500
    th.yBins = 500

    th.xMin = -1000
    th.yMin = -1000

    th.xMax = 1000
    th.yMax = 1000

    th.yOverFlow = True
    th.xOverFlow = True

    th.xData = delta["px"]
    th.xTitle = "$\\Delta P_{x}$-Momenta of truth and pyc neutrino (GeV)"

    th.yData = delta["py"]
    th.yTitle = "$\\Delta P_{y}$-Momenta of truth and pyc neutrino (GeV)"

    th.Filename = "Figure." + fign + ".d"
    th.Color = "tab20c"
    th.SaveFigure()


    sett = settings_th2f()
    th = TH2F(**sett)
    th.Title = "$\\Delta P_{x, z}$ Momenta Components between \n Truth and Reconstructed Neutrinos"
    th.xBins = 500
    th.yBins = 500

    th.xMin = -1000
    th.yMin = -1000

    th.xMax = 1000
    th.yMax = 1000

    th.yOverFlow = True
    th.xOverFlow = True

    th.xData = delta["px"]
    th.xTitle = "$\\Delta P_{x}$-Momenta of truth and pyc neutrino (GeV)"

    th.yData = delta["pz"]
    th.yTitle = "$\\Delta P_{z}$-Momenta of truth and pyc neutrino (GeV)"

    th.Filename = "Figure." + fign + ".e"
    th.Color = "tab20c"
    th.SaveFigure()


    sett = settings_th2f()
    th = TH2F(**sett)
    th.Title = "$\\Delta P_{y, z}$ Momenta Components between \n Truth and Reconstructed Neutrinos"
    th.xBins = 500
    th.yBins = 500

    th.xMin = -1000
    th.yMin = -1000

    th.xMax = 1000
    th.yMax = 1000

    th.yOverFlow = True
    th.xOverFlow = True

    th.xData = delta["py"]
    th.xTitle = "$\\Delta P_{y}$-Momenta of truth and pyc neutrino (GeV)"

    th.yData = delta["pz"]
    th.yTitle = "$\\Delta P_{z}$-Momenta of truth and pyc neutrino (GeV)"

    th.Filename = "Figure." + fign + ".f"
    th.Color = "tab20c"
    th.SaveFigure()

def projection_plots(data_pyc, data_ref, fign):
    for j, i, k in zip(["px", "py", "pz", "e"], ["g", "h", "i", "j"], ["$P_x$", "$P_y$", "$P_z$", "Energy"]):
        sett = settings()
        th1 = TH1F()
        th1.Title = "pyc"
        th1.xData = data_pyc[j]

        th2 = TH1F()
        th2.Title = "reference"
        th2.xData = data_ref[j]

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
    th1.xData = data["pyc"]

    th2 = TH1F()
    th2.Title = "reference"
    th2.xData = data["reference"]

    th3 = TH1F()
    th3.Title = "truth"
    th3.xData = data["truth"]
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

def s_matrix_bruteforce(ana):
    sett = settings_th2f()
    th = TH2F(**sett)
    th.Title = "S-Matrix Parameter Values Bruteforced for \n lowest $\\chi$ Value using PYC"
    th.xBins = 100
    th.yBins = 100

    th.xMin = 0
    th.yMin = 0

    th.xMax = 11000
    th.yMax = 11000

    th.xOverFlow = True
    th.yOverFlow = True

    th.xData = ana.S_matrix_delta_bruteforce_pyc["ii"]
    th.xTitle = "$S_{ii}$ - diagonal matrix elements (GeV)"

    th.yData = ana.S_matrix_delta_bruteforce_pyc["ij"]
    th.yTitle = "$S_{ij}$ - non diagonal matrix elements (GeV)"

    th.Filename = "Figure.5.a"
    th.Color = "tab20c"
    th.SaveFigure()

    sett = settings_th2f()
    th = TH2F(**sett)
    th.Title = "S-Matrix Parameter Values Bruteforced for \n lowest $\\chi$ Value using Reference"
    th.xBins = 100
    th.yBins = 100

    th.xMin = 0
    th.yMin = 0
    th.xOverFlow = True
    th.yOverFlow = True

    th.xMax = 11000
    th.yMax = 11000

    th.xData = ana.S_matrix_delta_bruteforce_ref["ii"]
    th.xTitle = "$S_{ii}$ - diagonal matrix elements (GeV)"

    th.yData = ana.S_matrix_delta_bruteforce_ref["ij"]
    th.yTitle = "$S_{ij}$ - non diagonal matrix elements (GeV)"

    th.Filename = "Figure.5.b"
    th.Color = "tab20c"
    th.SaveFigure()

    for j, i, k in zip(["px", "py", "pz", "chi2"], ["c", "d", "e", "f"], ["$P_x$", "$P_y$", "$P_z$", "$\\chi$"]):
        sett = settings()
        th1 = TH1F()
        th1.Title = "pyc"
        th1.xData = ana.S_matrix_delta_bruteforce_pyc[j]

        th2 = TH1F()
        th2.Title = "reference"
        th2.xData = ana.S_matrix_delta_bruteforce_ref[j]

        sett["Histograms"] += [th1, th2]

        th_ = TH1F(**sett)
        th_.Title = "Kinematic difference between Truth and PYC/Reference \n for: " + k
        if j == "chi2":
            th_.xTitle = k
            th_.yTitle = "Entries"
            th_.xMin = 0
        else:
            th_.xTitle = "$\\Delta$-" + k + " (GeV)"
            th_.yTitle = "Entries <unit>"
            th_.xMin  = -1000

        th_.Filename = "Figure.5." + i
        th_.xBins = 200
        th_.xMax  = 1000
        th_.xStep = 200
        th_.SaveFigure()




def NeutrinoReconstruction(ana):
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

    s_matrix_bruteforce(ana)
