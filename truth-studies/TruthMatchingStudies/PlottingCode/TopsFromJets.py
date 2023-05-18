from AnalysisG.Plotting import TH1F, CombineTH1F, TH2F

def PlotTemplate(x):
    Plots = {
                "yTitle" : "Entries (a.u.)",
                "xMin" : 0, 
                "yMin" : 0, 
                "xMax" : None,
                "xBins" : None,
                "OutputDirectory" : "./Figures/TopsFromJets", 
                "Style" : "ATLAS",
                "ATLASLumi" : x.Luminosity,
                "NEvents" : x.NEvents
            }
    return Plots

def PlotTemplateTH2F(x):
    Plots = {
                "xMin" : 0, 
                "yMin" : 0, 
                "OutputDirectory" : "./Figures/TopsFromJets", 
                "Style" : "ATLAS",
                "ATLASLumi" : x.Luminosity,
                "NEvents" : x.NEvents
            }
    return Plots

def TopMassJets(inpt):
    hist = PlotTemplate(inpt)
    hist["Title"] = "Reconstructed Invariant Mass of Top From (Truth) Jets \n Partitioned into Decay Topology"
    hist["xTitle"] = "Invariant Mass (GeV)"
    hist["xStep"] = 20
    hist["xMax"] = 360
    hist["xBins"] = 360
    hist["IncludeOverflow"] = True
    hist["Filename"] = "Figure_4.1a"
    hist["Histograms"] = []

    cols = iter(["Blue", "Red"])
    for mode in inpt.TopMassJet:
        c = next(cols)
        hist_m = {}
        hist_m["Title"] = mode + "(Jets)"
        hist_m["xData"] = inpt.TopMassJet[mode] 
        hist_m["Color"] = c
        hist["Histograms"].append(TH1F(**hist_m))        
        
        hist_m = {}
        hist_m["Title"] = mode + "(Truth Jets)"
        hist_m["xData"] = inpt.TopMassTruthJet[mode] 
        hist_m["Color"] = c
        hist_m["Texture"] = "/" 
        hist["Histograms"].append(TH1F(**hist_m))        
 
    com = CombineTH1F(**hist)
    #com.SaveFigure() 


    hist = PlotTemplate(inpt)
    hist["Title"] = "Number of (Truth) Jets Contributing to a Top \n Partitioned into Decay Topology"
    hist["xTitle"] = "Number of (Truth) Jets"
    hist["xStep"] = 1
    hist["Filename"] = "Figure_4.1b"
    hist["xBinCentering"] = True
    hist["Histograms"] = []
    cols = iter(["Blue", "Red"])
    for mode in inpt.NJets:
        c = next(cols)
        hist_m = {}
        hist_m["Title"] = mode + "(Jets)"
        hist_m["xData"] = inpt.NJets[mode] 
        hist_m["Color"] = c
        hist["Histograms"].append(TH1F(**hist_m))        
        
        hist_m = {}
        hist_m["Title"] = mode + "(Truth Jets)"
        hist_m["xData"] = inpt.NTruthJets[mode] 
        hist_m["Color"] = c
        hist_m["Texture"] = "/"
        hist["Histograms"].append(TH1F(**hist_m))        
        
    com = CombineTH1F(**hist)
    #com.SaveFigure() 
    
    xdata = {}
    for mode in inpt.TopMassJet:
        mass, nj = inpt.TopMassJet[mode], inpt.NJets[mode]
        for n, m in zip(nj, mass):
            if str(n) + "-Jets (" + mode + ")" not in xdata: xdata[str(n) + "-Jets (" + mode + ")"] = []
            xdata[str(n) + "-Jets (" + mode + ")"].append(m) 
            
    hist = PlotTemplate(inpt)
    hist["Title"] = "Reconstructed Invariant Top Mass using Jets \n Partitioned by Number of Jets"
    hist["xTitle"] = "Invariant Mass (GeV)"
    hist["xStep"] = 20
    hist["xMax"] = 360
    hist["xBins"] = 180
    hist["Filename"] = "Figure_4.1c"
    hist["Stack"] = True
    hist["IncludeOverflow"] = True
    hist["Histograms"] = []
 
    for n in xdata:
        hist_m = {}
        hist_m["Title"] = n
        hist_m["xData"] = xdata[n] 
        hist["Histograms"].append(TH1F(**hist_m))        
        
    com = CombineTH1F(**hist)
    #com.SaveFigure() 


    hist = PlotTemplate(inpt)
    hist["Title"] = "Reconstructed Invariant Top Mass using Jets \n and Detector Based Leptons with Truth Neutrinos"
    hist["xTitle"] = "Invariant Mass (GeV)"
    hist["xStep"] = 20
    hist["xMax"] = 360
    hist["xBins"] = 180
    hist["IncludeOverflow"] = True
    hist["Filename"] = "Figure_4.1d"
    hist["Histograms"] = []
 
    for mode in inpt.TopMassJetDetectorLep:
        if mode == "Had": continue
        hist_m = {}
        hist_m["Title"] = mode + " (Jets + Detector Leptons)"
        hist_m["xData"] = inpt.TopMassJetDetectorLep[mode] 
        hist["Histograms"].append(TH1F(**hist_m))        
        
        hist_m = {}
        hist_m["Title"] = mode + " (Jets + Truth Children Leptons)"
        hist_m["xData"] = inpt.TopMassJet[mode] 
        hist["Histograms"].append(TH1F(**hist_m))        

    com = CombineTH1F(**hist)
    #com.SaveFigure() 

    masshad = inpt.TopMassJet["Had"]
    njethad = inpt.NJets["Had"]
    xdata = {}
    for nj, m in zip(njethad, masshad):
        if nj not in xdata: xdata[nj] = []
        xdata[nj].append(m)

    hist = PlotTemplate(inpt)
    hist["Title"] = "Reconstructed Invariant Top Mass from Jets \n via Hadronic Decay Topology"
    hist["xTitle"] = "Invariant Mass (GeV)"
    hist["xStep"] = 20
    hist["xMax"] = 360
    hist["xBins"] = 180
    hist["IncludeOverflow"] = True
    hist["Filename"] = "Figure_4.1e"
    hist["Histograms"] = []
 
    for mode in xdata:
        hist_m = {}
        hist_m["Title"] = str(mode) + "-Jets"
        hist_m["xData"] = xdata[mode] 
        hist["Histograms"].append(TH1F(**hist_m))        
        
    com = CombineTH1F(**hist)
    com.SaveFigure() 


    hist_2 = PlotTemplateTH2F(inpt)
    hist_2["Title"] = "Average $\Delta$ R Clustering of Truth Matched Particles \n With Respect to Invariant Mass (Hadronic)"
    hist_2["xTitle"] = "Invariant Mass (GeV)"
    hist_2["yTitle"] = "Average $\Delta$R"
    hist_2["xBins"] = 180
    hist_2["xMax"] = 360
    hist_2["yBins"] = 180
    hist_2["yMax"] = 4
    hist_2["Filename"] = "Figure_4.1f"
    hist_2["xData"] = inpt.TopMassJet["Had"]
    hist_2["yData"] = inpt.DeltaRJets["Had"]
    th = TH2F(**hist_2)
    th.SaveFigure()

    hist_2 = PlotTemplateTH2F(inpt)
    hist_2["Title"] = "Average $\Delta$ R Clustering of Truth Matched Particles \n With Respect to Invariant Mass (Leptonic)"
    hist_2["xTitle"] = "Invariant Mass (GeV)"
    hist_2["yTitle"] = "Average $\Delta$R"
    hist_2["xBins"] = 180
    hist_2["xMax"] = 360
    hist_2["yBins"] = 180
    hist_2["yMax"] = 4
    hist_2["Filename"] = "Figure_4.2f"
    hist_2["xData"] = inpt.TopMassJet["Lep"]
    hist_2["yData"] = inpt.DeltaRJets["Lep"]
    th = TH2F(**hist_2)
    th.SaveFigure()

    hist_2 = PlotTemplateTH2F(inpt)
    hist_2["Title"] = "Average $\Delta$ R Clustering of Truth Matched Particles \n With Respect to Invariant Mass (All)"
    hist_2["xTitle"] = "Invariant Mass (GeV)"
    hist_2["yTitle"] = "Average $\Delta$R"
    hist_2["xBins"] = 180
    hist_2["xMax"] = 360
    hist_2["yBins"] = 180
    hist_2["yMax"] = 4
    hist_2["Filename"] = "Figure_4.3f"
    hist_2["xData"] = inpt.TopMassJet["Lep"] + inpt.TopMassJet["Had"]
    hist_2["yData"] = inpt.DeltaRJets["Lep"] + inpt.DeltaRJets["Had"]
    th = TH2F(**hist_2)
    th.SaveFigure() 
