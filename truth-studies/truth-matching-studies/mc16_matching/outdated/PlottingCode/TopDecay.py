
    Plots = PlotTemplate(x)
    Plots["Title"] = "Reconstructed Invariant Top Mass from Immediate Decay Products"
    Plots["xTitle"] = "Invariant Top Mass (GeV)"
    Plots["xMin"] = 120
    Plots["xStep"] = 10
    Plots["xMax"] = 240
    Plots["xBins"] = 1000
    Plots["Filename"] = "Figure_2.1b"
    Plots["Histograms"] = []

    for i in x.TopMassTC:
        _Plots = {}
        _Plots["Title"] = i
        _Plots["xData"] = x.TopMassTC[i]
        Plots["Histograms"].append(TH1F(**_Plots))
    plt = TH1F(**Plots)
    plt.SaveFigure()


