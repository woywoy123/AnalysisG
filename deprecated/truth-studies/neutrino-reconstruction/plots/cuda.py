from AnalysisG.IO import PickleObject, UnpickleObject
from AnalysisG.Plotting import TH1F, TH2F
global figure_path

def settings():
    settings = {
            "Style" : "ROOT",
            "OutputDirectory" : figure_path + "neutrino-studies/cuda/figures/",
            "Histograms" : [],
            "Histogram" : None,
            "FontSize"  : 20,
            "LabelSize" : 20,
            "xScaling"  : 10,
            "yScaling"  : 12,
            "LegendLoc" : "upper right"
    }
    return settings

def settings_th2f():
    settings = {
            "Style" : "ROOT",
            "OutputDirectory" : figure_path + "neutrino-studies/cuda/figures/",
            "FontSize"  : 20,
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
        th.Color = "cubehelix"
        th.SaveFigure()


def neutrino_kinematic(delta, key, fign_):
    titles = [key, key, key]
    figures_ = [".a", ".b", ".c"]
    pairs = [("x", "y"), ("x", "z"), ("y", "z")]
    datax = [delta["px"], delta["px"], delta["py"]]
    datay = [delta["py"], delta["pz"], delta["pz"]]
    momentum_template(fign_, figures_, pairs, titles, datax, datay)


def time_performance(ref, pyc_ten, pyc_cuda, key):
    base = ref
    print("")
    print("----", key, "----")
    print(base/pyc_ten, "ref / pyc (CPU)")
    print(base/pyc_cuda, "ref / pyc (CUDA)")
    print(pyc_ten/pyc_cuda, "pyc (CPU) /pyc (CUDA)")


def projection_plots(pyc_ten, pyc_cu, pyc_comb, data_ref, fign):
    for j, i, k in zip(["px", "py", "pz", "e", "mt"], [".d", ".e", ".f", ".g", ".h"], ["$P_x$", "$P_y$", "$P_z$", "Energy", "Invariant Top-Mass"]):
        sett = settings()
        th1 = TH1F()
        th1.Title = "pyc-tensor"
        th1.xData = pyc_ten[j]

        th2 = TH1F()
        th2.Title = "pyc-cuda"
        th2.xData = pyc_cu[j]

        th3 = TH1F()
        th3.Title = "pyc-combinatorial"
        if j == "mt": th3.xData = pyc_comb[j + "_sol"]
        else: th3.xData = pyc_comb[j]

        th4 = TH1F()
        th4.Title = "reference"
        th4.xData = data_ref[j]

        if j == "mt":
            th5 = TH1F()
            th5.Title = "truth"
            th5.xData = pyc_comb["tru_" + j]
            sett["Histograms"] += [th1, th2, th3, th4, th5]
        else:
            sett["Histograms"] += [th1, th2, th3, th4]

        th_ = TH1F(**sett)
        th_.Title = "Reconstruction Performance of Double Neutrino \n for Different Algorithmic Implementation Modes \n for: " + k
        if j == "mt": th_.xTitle = "Invariant Top-Mass (GeV)"
        else: th_.xTitle = "$\\Delta$-" + k + " (GeV)"
        th_.yTitle = "Entries <unit>"
        th_.Filename = "Figure."+ fign + i
        th_.xBins = 200
        th_.xMin  = -1000
        th_.xMax  = 1000
        th_.yLogarithmic = True
        th_.xStep = 200
        th_.SaveFigure()

def combinatorial_masses(data, key, fign):
    sett = settings()
    th1 = TH1F()
    th1.Title = "pyc-combinatorial"
    th1.xData = data["mt"]

    th2 = TH1F()
    th2.Title = "truth"
    th2.xData = data["tru_mt"]

    sett["Histograms"] = [th1, th2]
    th_ = TH1F(**sett)
    th_.Title = "Reconstructed Invariant Top Mass using " + key
    th_.xTitle = "Invariant Mass (GeV)"
    th_.yTitle = "Entries <unit>"
    th_.Filename = "Figure."+ fign + ".i"
    th_.xBins = 200
    th_.xMin  = 100
    th_.xMax  = 300
    th_.xStep = 20
    th_.SaveFigure()

    sett = settings()
    th1 = TH1F()
    th1.Title = "pyc-combinatorial"
    th1.xData = data["mw"]

    th2 = TH1F()
    th2.Title = "truth"
    th2.xData = data["tru_wt"]
    sett["Histograms"] = [th1, th2]

    th_ = TH1F(**sett)
    th_.Title = "Reconstructed Invariant W-Boson Mass using " + key
    th_.xTitle = "Invariant Mass (GeV)"
    th_.yTitle = "Entries <unit>"
    th_.Filename = "Figure."+ fign + ".j"
    th_.xBins = 100
    th_.xMin  = 50
    th_.xMax  = 150
    th_.xStep = 20
    th_.SaveFigure()

def combinatorial():
    res = UnpickleObject("results.pkl")
    reference = res["reference"]
    pyc_ten = res["pyc_tensor"]
    pyc_cuda = res["pyc_cuda"]
    pyc_comb = res["pyc_combinatorial"]
    num = 1
    for i in ["children", "truthjet", "jets", "detector"]:
        time_performance(reference[i]["time"], pyc_ten[i]["time"], pyc_cuda[i]["time"], i)
        neutrino_kinematic(pyc_comb[i], i, str(num))
        projection_plots(pyc_ten[i], pyc_cuda[i], pyc_comb[i], reference[i], str(num))
        combinatorial_masses(pyc_comb[i], i, str(num))
        num += 1


