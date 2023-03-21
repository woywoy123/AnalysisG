from AnalysisTopGNN.Plotting import TH1F, TH2F, CombineTH1F

def TemplatePlotsTH1F(x):
    Plots = {
                "NEvents" : x.NEvents, 
                "ATLASLumi" : x.Luminosity, 
                "Style" : "ATLAS", 
                "OutputDirectory" : "./Figures/TruthEvent", 
                "yTitle" : "Entries (a.u.)", 
                "yMin" : 0, "xMin" : 0
            }
    return Plots

def TemplatePlotsTH2F(x):
    Plots = {
                "NEvents" : x.NEvents, 
                "ATLASLumi" : x.Luminosity, 
                "Style" : "ATLAS", 
                "OutputDirectory" : "./Figures/TruthEvent", 
                "yMin" : 0, "xMin" : 0
            }
    return Plots

def EventNTruthJetAndJets(x):

    Plots = TemplatePlotsTH2F(x)
    Plots["xTitle"] = "n-TruthJets"
    Plots["yTitle"] = "n-Jets"
    Plots["xStep"] = 1
    Plots["yStep"] = 1
    Plots["xBinCentering"] = True 
    Plots["yBinCentering"] = True 
    Plots["xData"] = list(x.TruthJets)
    Plots["yData"] = list(x.Jets)
    #/// Continue here


    Plots["Filename"] = "N-TruthJets_n-Jets"
    th = TH2F(**Plots)







