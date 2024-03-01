
    Plots = PlotTemplate(nevents, lumi)
    Plots["Title"] = "Missing Transverse Energy"
    Plots["xTitle"] = "MET (GeV)"
    Plots["xBins"] = 200
    Plots["xMin"] = 0
    Plots["xMax"] = 1000
    Plots["Filename"] = "MET_withRad"
    Plots["Histograms"] = []

    for i in MissingET:
        _Plots = {}
        _Plots["Title"] = i
        _Plots["xData"] = MissingET[i]
        Plots["Histograms"].append(TH1F(**_Plots))
    
    X = CombineTH1F(**Plots)
    X.SaveFigure()

    Plots = PlotTemplate(nevents, lumi)
    Plots["Title"] = "Missing Transverse Energy Difference"
    Plots["xTitle"] = "MET calculated - MET from ntuples (GeV)"
    Plots["xBins"] = 200
    Plots["xMin"] = -500
    Plots["xMax"] = 500
    Plots["Filename"] = "METDiff_withRad"
    Plots["Histograms"] = []

    for i in MissingETDiff:
        _Plots = {}
        _Plots["Title"] = i
        _Plots["xData"] = MissingETDiff[i]
        Plots["Histograms"].append(TH1F(**_Plots))
    
    X = CombineTH1F(**Plots)
    X.SaveFigure()

