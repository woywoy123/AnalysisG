from AnalysisTopGNN.Plotting import TH1F, TH2F, CombineTH1F

def TemplatePlotsTH1F(x):
    Plots = {
                "NEvents" : x.NEvents, 
                "ATLASLumi" : x.Luminosity, 
                "Style" : "ATLAS", 
                "OutputDirectory" : "./Figures/EventNeutrino", 
                "yTitle" : "Entries (a.u.)", 
                "yMin" : 0, "xMin" : 0
            }
    return Plots

def TemplatePlotsTH2F(x):
    Plots = {
                "NEvents" : x.NEvents, 
                "ATLASLumi" : x.Luminosity, 
                "Style" : "ATLAS", 
                "OutputDirectory" : "./Figures/EventNeutrino", 
                "yMin" : 0, "xMin" : 0
            }
    return Plots



def EventNuNuSolutions(x):
    Plots_ = TemplatePlotsTH1F(x)
    Plots_["Title"] = "Number of Neutrino Reconstruction Solutions"
    Plots_["xTitle"] = "Number of Solutions"
    Plots_["Histograms"] = []
    for i in x.NuNuSolutions:
        Plots = {}
        Plots["Title"] = i
        Plots["xData"] = x.NuNuSolutions[i]
        thc = TH1F(**Plots)
        Plots_["Histograms"].append(thc)
    Plots_["xStep"] = 1
    Plots_["xBinCentering"] = True
    Plots_["Filename"] = "Number-of-solutions"
    tc = CombineTH1F(**Plots_)
    tc.SaveFigure()
    
    exit()

    # Njets vs Ntruthjets
    Plots = TemplatePlotsTH2F(x)
    Plots["Title"] = "Number of Truth-Jets vs Reconstructed Jets"
    Plots["xTitle"] = "n-TruthJets"
    Plots["yTitle"] = "n-Jets"
    Plots["xStep"] = 1
    Plots["yStep"] = 1
    Plots["xScaling"] = 2
    Plots["yScaling"] = 2
    Plots["xData"] = x.TruthJets
    Plots["yData"] = x.Jets

    Plots["Filename"] = "N-TruthJets_n-Jets"
    th = TH2F(**Plots)
    th.SaveFigure()


