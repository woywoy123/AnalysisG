from AnalysisG.Plotting import TH1F, CombineTH1F, TH2F

def PlotTemplate(x):
    Plots = {
                "yTitle" : "Entries (a.u.)",
                "xMin" : 0, 
                "yMin" : 0, 
                "xMax" : None,
                "xBins" : None,
                "OutputDirectory" : "./Figures/" + x.__class__.__name__, 
                "Style" : "ATLAS",
                "ATLASLumi" : x.Luminosity,
                "NEvents" : x.NEvents
            }
    return Plots



def TopTruthJetsKinematics(inpt):
    

    xdata = {}
    for dr, sym in zip(inpt.TopTruthJet_parton["Parton-dR"], inpt.TopTruthJet_parton["Parton-symbol"]):
        if sym not in xdata: xdata[sym] = []
        xdata[sym].append(dr)
   
    hist = PlotTemplate(inpt)
    hist["Title"] = "$\Delta R$ Of Top Matched Truth Jet Ghost Parton Composition \n Partitioned into PDGID Symbol"
    hist["xTitle"] = "$\Delta R$"
    hist["Filename"] = "Figure_3.1i"
    hist["xMax"] = 1
    hist["xBins"] = 250
    hist["Logarithmic"] = True
    hist["yMin"] = 1
    hist["xStep"] = 0.1
    hist["Histograms"] = []

    for sym in xdata:
        plt = {}
        plt["Title"] = sym
        plt["xData"] = xdata[sym]
        plt["xBins"] = 250
        hist["Histograms"].append(TH1F(**plt))
    
    com = CombineTH1F(**hist)
    com.SaveFigure()

    hist2D = PlotTemplateTH2F(inpt)
    hist2D.Title = "Truth Jet Ghost Matched Parton's $\Delta R$ \n as a Function of $\eta$"
    hist2D.xTitle = "Pseudo-Rapiditiy $\eta$"
    hist2D.yTitle = "$\Delta R$"
    hist2D.yMax = 1.0
    hist2D.xMax = 4.0
    hist2D.xMin = -4.0
    hist2D.yBins = 200
    hist2D.xBins = 200
    hist2D.xStep = 0.4
    hist2D.yStep = 0.1
    hist2D.yData = [i for m in inpt.TopTruthJet_parton for i in inpt.TopTruthJet_parton[m] if "dR" in m]
    hist2D.xData = [i for m in inpt.TopTruthJet_parton for i in inpt.TopTruthJet_parton[m] if "eta" in m]
    hist2D.Filename = "Figure_3.2i"
    hist2D.SaveFigure()


    hist2D = PlotTemplateTH2F(inpt)
    hist2D.Title = "Truth Jet Ghost Matched Parton's $\Delta R$ \n as a Function of $\phi$"
    hist2D.xTitle = "Azimuthal Angle - $\phi$ (Radians)"
    hist2D.yTitle = "$\Delta R$"
    hist2D.yMax = 1.0
    hist2D.xMax = 3.2
    hist2D.xMin = -3.2
    hist2D.yBins = 200
    hist2D.xBins = 200
    hist2D.xStep = 0.4
    hist2D.yStep = 0.1
    hist2D.yData = [i for m in inpt.TopTruthJet_parton for i in inpt.TopTruthJet_parton[m] if "dR" in m]
    hist2D.xData = [i for m in inpt.TopTruthJet_parton for i in inpt.TopTruthJet_parton[m] if "phi" in m]
    hist2D.Filename = "Figure_3.3i"
    hist2D.SaveFigure()

    xdata = {}
    for dr, sym in zip(inpt.TopTruthJet_parton["Parton-Energy-Frac"], inpt.TopTruthJet_parton["Parton-symbol"]):
        if sym not in xdata: xdata[sym] = []
        xdata[sym].append(dr)

    hist = PlotTemplate(inpt)
    hist["Title"] = "Fractional Energy Contribution of Ghost Matched Parton to Truth Jet"
    hist["xTitle"] = r"Fractional Contribution to Truth Jet ($\frac{Parton}{TruthJet}$) (a.u.)"
    hist["Filename"] = "Figure_3.1j"
    hist["xMax"] = 10
    hist["xBins"] = 500
    hist["Logarithmic"] = True
    hist["yMin"] = 1
    hist["xStep"] = 0.5
    hist["Histograms"] = []

    for sym in xdata:
        plt = {}
        plt["Title"] = sym
        plt["xData"] = xdata[sym]
        plt["xBins"] = 250
        hist["Histograms"].append(TH1F(**plt))
    
    com = CombineTH1F(**hist)
    com.SaveFigure()

    hist = PlotTemplate(inpt)
    hist["Title"] = "Reconstructed Invariant Top Mass from Truth Jets \n With and Without Gluon Only Jets (Leptonic Tops Not Considered)"
    hist["xTitle"] = "Invariant Mass (GeV)"
    hist["Filename"] = "Figure_3.1k"
    hist["xMax"] = 400
    hist["xBins"] = 500
    hist["Logarithmic"] = True
    hist["IncludeOverflow"] = True
    hist["yMin"] = 1
    hist["xStep"] = 40
    hist["Histograms"] = []

    for sym in inpt.TopMass:
        plt = {}
        plt["Title"] = sym
        plt["xData"] = inpt.TopMass[sym]
        hist["Histograms"].append(TH1F(**plt))
    com = CombineTH1F(**hist)
    com.SaveFigure()

    hist = PlotTemplate(inpt)
    hist["Title"] = "Truth Jet Invariant Mass Partitioned into n-Top Contributions"
    hist["xTitle"] = "Invariant Mass (GeV)"
    hist["Filename"] = "Figure_3.1l"
    hist["xMax"] = 300
    hist["xBins"] = 500
    hist["Logarithmic"] = True
    hist["IncludeOverflow"] = True
    hist["yMin"] = 1
    hist["xStep"] = 20
    hist["Histograms"] = []

    for sym in inpt.JetMassNTop:
        plt = {}
        plt["Title"] = str(sym) + "-Tops"
        plt["xData"] = inpt.JetMassNTop[sym]
        hist["Histograms"].append(TH1F(**plt))
    com = CombineTH1F(**hist)
    com.SaveFigure()

def MergedTopsTruthJets(inpt):
    
    hist = PlotTemplate(inpt)
    hist["Title"] = "Truth Jet Parton Transverse Momentum of \n Top Merged Truth Jets (Partitioned into PDGID)"
    hist["xTitle"] = "Transverse Momentum (GeV)" 
    hist["IncludeOverflow"] = True 
    hist["Logarithmic"] = True
    hist["xStep"] = 50
    hist["xMax"] = 300
    hist["yMin"] = 1
    hist["xBins"] = 250
    hist["Filename"] = "Figure_3.1a"
    hist["Histograms"] = []
 
    for sym in inpt.PartonPT:
        plt = {}
        plt["Title"] = str(sym)
        plt["xData"] = inpt.PartonPT[sym]
        hist["Histograms"].append(TH1F(**plt))
    com = CombineTH1F(**hist)
    com.SaveFigure()

    hist = PlotTemplate(inpt)
    hist["Title"] = "Top Merged Truth Jet Parton Energy \n (Partitioned into PDGID)"
    hist["xTitle"] = "Energy (GeV)" 
    hist["IncludeOverflow"] = True 
    hist["Logarithmic"] = True
    hist["yMin"] = 1
    hist["xStep"] = 50
    hist["xMax"] = 500
    hist["xBins"] = 250
    hist["Filename"] = "Figure_3.2a"
    hist["Histograms"] = []
 
    for sym in inpt.PartonEnergy:
        plt = {}
        plt["Title"] = str(sym)
        plt["xData"] = inpt.PartonEnergy[sym]
        hist["Histograms"].append(TH1F(**plt))
    com = CombineTH1F(**hist)
    com.SaveFigure()

    hist = PlotTemplate(inpt)
    hist["Title"] = "$\Delta$R Between Jet Axis and Partons contained in \n Top Merged Truth Jets (Partitioned into PDGID)"
    hist["xTitle"] = "$\Delta$R Between Jet Axis and Partons" 
    hist["IncludeOverflow"] = True 
    hist["xStep"] = 0.1
    hist["xMax"] = 1.0
    hist["xBins"] = 250
    hist["Filename"] = "Figure_3.3a"
    hist["Histograms"] = []
 
    for sym in inpt.PartonDr:
        plt = {}
        plt["Title"] = str(sym)
        plt["xData"] = inpt.PartonDr[sym]
        hist["Histograms"].append(TH1F(**plt))
    com = CombineTH1F(**hist)
    com.SaveFigure()

    hist2D = PlotTemplateTH2F(inpt)
    hist2D.Title = "$\Delta$R Between Truth Jet Axis and Contributing Partons \n as a Function of Parton's Energy (Only Gluons)"
    hist2D.xTitle = "Parton Energy (GeV)"
    hist2D.yTitle = "$\Delta R$"
    hist2D.IncludeOverflow = True
    hist2D.yMax = 0.6
    hist2D.xMax = 500
    hist2D.xMin = 0
    hist2D.yBins = 250
    hist2D.xBins = 250
    hist2D.xStep = 100
    hist2D.yStep = 0.1
    hist2D.yData = [i for m in inpt.PartonDr for i in inpt.PartonDr[m] if m == "g"]
    hist2D.xData = [i for m in inpt.PartonEnergy for i in inpt.PartonEnergy[m] if m == "g"]
    hist2D.Filename = "Figure_3.4a"
    hist2D.SaveFigure()
    
    hist2D = PlotTemplateTH2F(inpt)
    hist2D.Title = "$\Delta$R Between Truth Jet Axis and Contributing Partons \n as a Function of Parton's Energy (Without Gluons)"
    hist2D.xTitle = "Parton Energy (GeV)"
    hist2D.yTitle = "$\Delta R$"
    hist2D.IncludeOverflow = True
    hist2D.yMax = 0.6
    hist2D.xMax = 500
    hist2D.xMin = 0
    hist2D.yBins = 250
    hist2D.xBins = 250
    hist2D.xStep = 100
    hist2D.yStep = 0.1
    hist2D.yData = [i for m in inpt.PartonDr for i in inpt.PartonDr[m] if m != "g"]
    hist2D.xData = [i for m in inpt.PartonEnergy for i in inpt.PartonEnergy[m] if m != "g"]
    hist2D.Filename = "Figure_3.5a"
    hist2D.SaveFigure()

    hist = PlotTemplate(inpt)
    hist["Title"] = "Transverse Momentum of Truth Children Matched to \n Truth Jet Contributing Parton (Partitioned into PDGID)"
    hist["xTitle"] = "Transverse Momentum (GeV)" 
    hist["IncludeOverflow"] = True 
    hist["xStep"] = 100
    hist["xMax"] = 800
    hist["xBins"] = 250
    hist["Filename"] = "Figure_3.1b"
    hist["Histograms"] = []
 
    for sym in inpt.ChildPartonPT:
        plt = {}
        plt["Title"] = str(sym)
        plt["xData"] = inpt.ChildPartonPT[sym]
        hist["Histograms"].append(TH1F(**plt))
    com = CombineTH1F(**hist)
    com.SaveFigure()

    hist = PlotTemplate(inpt)
    hist["Title"] = "Energy of Truth Children Matched to \n Truth Jet Contributing Parton (Partitioned into PDGID)"
    hist["xTitle"] = "Energy (GeV)" 
    hist["IncludeOverflow"] = True 
    hist["xStep"] = 100
    hist["xMax"] = 800
    hist["xBins"] = 250
    hist["Filename"] = "Figure_3.2b"
    hist["Histograms"] = []
 
    for sym in inpt.ChildPartonEnergy:
        plt = {}
        plt["Title"] = str(sym)
        plt["xData"] = inpt.ChildPartonEnergy[sym]
        hist["Histograms"].append(TH1F(**plt))
    com = CombineTH1F(**hist)
    com.SaveFigure()

    hist = PlotTemplate(inpt)
    hist["Title"] = "$\Delta$R Between Parton and Truth Child (Partitioned into Parton PDGID)"
    hist["xTitle"] = "$\Delta$R Between Parton and Truth Child" 
    hist["IncludeOverflow"] = True 
    hist["Logarithmic"] = True
    hist["xStep"] = 0.1
    hist["yMin"] = 1
    hist["xMax"] = 1.0
    hist["xBins"] = 250
    hist["Filename"] = "Figure_3.3b"
    hist["Histograms"] = []
 
    for sym in inpt.ChildPartonDr:
        plt = {}
        plt["Title"] = str(sym)
        plt["xData"] = inpt.ChildPartonDr[sym]
        hist["Histograms"].append(TH1F(**plt))
    com = CombineTH1F(**hist)
    com.SaveFigure()

    hist = PlotTemplate(inpt)
    hist["Title"] = "$\Delta$R Between Truth Jet Axis and Truth Child \n (Partitioned into Parton PDGID)"
    hist["xTitle"] = "$\Delta$R Between Truth Jet Axis and Truth Child" 
    hist["IncludeOverflow"] = True 
    hist["Logarithmic"] = True
    hist["xStep"] = 0.2
    hist["xMax"] = 3.0
    hist["yMin"] = 1
    hist["xBins"] = 400
    hist["Filename"] = "Figure_3.4b"
    hist["Histograms"] = []
 
    for sym in inpt.dRChildPartonJetAxis:
        plt = {}
        plt["Title"] = str(sym)
        plt["xData"] = inpt.dRChildPartonJetAxis[sym]
        hist["Histograms"].append(TH1F(**plt))
    com = CombineTH1F(**hist)
    com.SaveFigure()

    hist2D = PlotTemplateTH2F(inpt)
    hist2D.Title = "$\Delta$R Between Contributing Parton and Truth Child as a \n Function of Child's Energy (Only Gluons)"
    hist2D.xTitle = "Child Energy (GeV)"
    hist2D.yTitle = "$\Delta R$ Between Parton and Truth Child"
    hist2D.yMax = 1.0
    hist2D.xMax = 1000
    hist2D.xMin = 0
    hist2D.yBins = 250
    hist2D.xBins = 250
    hist2D.xStep = 100
    hist2D.yStep = 0.1
    hist2D.yData = [i for m in inpt.ChildPartonDr for i in inpt.ChildPartonDr[m] if m == "g"]
    hist2D.xData = [i for m in inpt.ChildPartonEnergy for i in inpt.ChildPartonEnergy[m] if m == "g"]
    hist2D.Filename = "Figure_3.5b"
    hist2D.SaveFigure()
    
    hist2D = PlotTemplateTH2F(inpt)
    hist2D.Title = "$\Delta$R Between Contributing Parton and Truth Child as a \n Function of Child's Energy (Without Gluons)"
    hist2D.xTitle = "Child Energy (GeV)"
    hist2D.yTitle = "$\Delta R$ Between Parton and Truth Child"
    hist2D.yMax = 1.0
    hist2D.xMax = 600
    hist2D.IncludeOverflow = True
    hist2D.xMin = 0
    hist2D.yBins = 100
    hist2D.xBins = 100
    hist2D.xStep = 50
    hist2D.yStep = 0.1
    hist2D.yData = [i for m in inpt.ChildPartonDr for i in inpt.ChildPartonDr[m] if m != "g"]
    hist2D.xData = [i for m in inpt.ChildPartonEnergy for i in inpt.ChildPartonEnergy[m] if m != "g"]
    hist2D.Filename = "Figure_3.6b"
    hist2D.SaveFigure()
    
    hist = PlotTemplate(inpt)
    hist["Title"] = "Parton Type Contribution Frequency Found in Top Merged Truth Jets"
    hist["xTitle"] = "Frequency of Parton Symbol in Truth Jets"
    hist["IncludeOverflow"] = True 
    hist["xStep"] = 1
    hist["xMax"] = 20
    hist["xBins"] = 20
    hist["xBinCentering"] = True
    hist["Filename"] = "Figure_3.1c"
    hist["Histograms"] = []
 
    for sym in inpt.NumberOfConstituentsInJet:
        plt = {}
        plt["Title"] = str(sym)
        plt["xData"] = inpt.NumberOfConstituentsInJet[sym]
        hist["Histograms"].append(TH1F(**plt))
    com = CombineTH1F(**hist)
    com.SaveFigure()

    hist = PlotTemplate(inpt)
    hist["Title"] = "Fractional Contribution of a given Top's Parton to a Truth Jet"
    hist["xTitle"] = "Fractional Energy Contribution of Partons from a Given Top in Truth Jet"
    hist["IncludeOverflow"] = True 
    hist["xStep"] = 0.1
    hist["xMax"] = 1.1
    hist["xBins"] = 100
    hist["yMin"] = 1
    hist["Logarithmic"] = True
    hist["Filename"] = "Figure_3.4c"
    hist["Histograms"] = []
 
    for sym in inpt.TopsTruthJetsMerged:
        plt = {}
        plt["Title"] = str(sym) + "-Tops"
        plt["xData"] = inpt.TopsTruthJetsMerged[sym]
        hist["Histograms"].append(TH1F(**plt))
    com = CombineTH1F(**hist)
    com.SaveFigure()

    hist = PlotTemplate(inpt)
    hist["Title"] = "Reconstructed Invariant Top Mass using the Hadronic \n Decay Topology from Truth Jets using different Energy Fraction Cuts"
    hist["xTitle"] = "Invariant Mass (GeV)"
    hist["IncludeOverflow"] = True 
    hist["xStep"] = 20
    hist["xMax"] = 400
    hist["xBins"] = 200
    hist["Alpha"] = 0.1
    hist["Filename"] = "Figure_3.5c"
    hist["Histograms"] = []
 
    for sym in inpt.TopsTruthJetsCut:
        plt = {}
        plt["Title"] = str(sym) + "-Cut"
        plt["xData"] = inpt.TopsTruthJetsCut[sym]
        hist["Histograms"].append(TH1F(**plt))
    com = CombineTH1F(**hist)
    com.SaveFigure()


