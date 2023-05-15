from AnalysisG.Plotting import TH1F, CombineTH1F

def PlotTemplate(x):
    Plots = {
                "yTitle" : "Entries (a.u.)",
                "xMin" : 0, 
                "yMin" : 0, 
                "xMax" : None,
                "xBins" : None,
                "OutputDirectory" : "./Figures/SingleLepton", 
                "Style" : "ATLAS",
                "ATLASLumi" : x.Luminosity,
                "NEvents" : x.NEvents
            }
    return Plots

def SingleLepton(inpt):

    plt = PlotTemplate(inpt)
    plt["Title"] = "Reconstructed Invariant Mass of Resonance at \n Different b-tagging Working Points"
    plt["xTitle"] = "Invariant Mass (GeV)" 
    plt["xBins"] = 1000
    plt["xMax"] = 2000
    plt["xStep"] = 100
    plt["Filename"] = "Z-Prime"
    plt["Histograms"] = []   
     
    for wp in inpt.ZMass:
        p_ = {}
        p_["Title"] = wp + " b-tagging" if wp != "Truth" else wp
        p_["xData"] = inpt.ZMass[wp] 
        plt["Histograms"].append(TH1F(**p_))
    
    com = CombineTH1F(**plt)
    com.SaveFigure() 

    plt = PlotTemplate(inpt)
    plt["Title"] = "CutFlow Output"
    plt["xStep"] = 1
    plt["xBinCentering"] = True
    plt["xTickLabels"] = [i for i in inpt.CutFlow if "::" not in i]
    plt["xWeights"] = [inpt.CutFlow[i] for i in inpt.CutFlow if "::" not in i]
    plt["Filename"] = "CutFlowStatus"
    t = TH1F(**plt)
    t.SaveFigure()





