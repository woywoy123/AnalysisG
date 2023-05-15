from AnalysisG.Plotting import TH1F, CombineTH1F, TH2F

def PlotTemplate(x):
    Plots = {
                "yTitle" : "Entries (a.u.)",
                "xMin" : 0, 
                "yMin" : 0, 
                "xMax" : None,
                "xBins" : None,
                "OutputDirectory" : "./Figures/TopsFromTruthJets", 
                "Style" : "ATLAS",
                "ATLASLumi" : x.Luminosity,
                "NEvents" : x.NEvents
            }
    return Plots

def PlotTemplateTH2F(x):
    th2 = TH2F()
    th2.ATLASLumi = x.Luminosity
    th2.Style = "ATLAS"
    th2.NEvents = x.NEvents
    th2.xMin = 0
    th2.yMin = 0
    th2.OutputDirectory = "./Figures/TopsFromTruthJets"
    return th2

def TopMassTruthJets(inpt):
    hist = PlotTemplate(inpt)
    hist["Title"] = "Reconstructed Invariant Mass of Top-Matched Truth Jets \n Compared to Original Truth Top"
    hist["xTitle"] = "Invariant Mass (GeV)"
    hist["IncludeOverflow"] = True
    hist["Logarithmic"] = True
    hist["xStep"] = 50
    hist["xMax"] = 400
    hist["xBins"] = 400
    hist["yMin"] = 1
    hist["Filename"] = "Figure_3.1a" 
    hist["Histograms"] = []
 
    for mode in inpt.TopMass:
        m_hist = {}
        m_hist["Title"] = "TruthJet-" + mode  
        m_hist["xData"] = inpt.TopMass[mode]
        hist["Histograms"].append(TH1F(**m_hist))

        m_hist = {}
        m_hist["Title"] = "TruthTops-" + mode  
        m_hist["xData"] = inpt.TruthTopMass[mode]
        hist["Histograms"].append(TH1F(**m_hist))

    com = CombineTH1F(**hist)
    com.SaveFigure() 


    hist = PlotTemplate(inpt)
    hist["Title"] = "Reconstructed Invariant Mass of Truth Jet derived Tops \n Partitioned into n-TruthJet Contributions (Leptonic Decay)"
    hist["xTitle"] = "Invariant Mass (GeV)"
    hist["IncludeOverflow"] = True
    hist["Logarithmic"] = True
    hist["xStep"] = 50
    hist["xMax"] = 400
    hist["xBins"] = 400
    hist["yMin"] = 1
    hist["Filename"] = "Figure_3.1b" 
    hist["Histograms"] = []
 
    for mode in inpt.TopMassNjets:
        if "Had" in mode: continue
        m_hist = {}
        m_hist["Title"] = mode.replace("Lep-", "") +  "-truthjets" 
        m_hist["xData"] = inpt.TopMassNjets[mode]
        hist["Histograms"].append(TH1F(**m_hist))

    com = CombineTH1F(**hist)
    com.SaveFigure() 

    hist = PlotTemplate(inpt)
    hist["Title"] = "Reconstructed Invariant Mass of Truth Jet derived Tops \n Partitioned into n-TruthJet Contributions (Hadronic Decay)"
    hist["xTitle"] = "Invariant Mass (GeV)"
    hist["IncludeOverflow"] = True
    hist["Logarithmic"] = True
    hist["xStep"] = 50
    hist["xMax"] = 400
    hist["yMin"] = 1
    hist["xBins"] = 400
    hist["Filename"] = "Figure_3.2b" 
    hist["Histograms"] = []
 
    for mode in inpt.TopMassNjets:
        if "Lep" in mode: continue
        m_hist = {}
        m_hist["Title"] = mode.replace("Had-", "") + "-truthjets"
        m_hist["xData"] = inpt.TopMassNjets[mode]
        hist["Histograms"].append(TH1F(**m_hist))

    com = CombineTH1F(**hist)
    com.SaveFigure() 

    
    xdata = [i for mode in inpt.TopMassNjets for i in inpt.TopMassNjets[mode]]
    ydata = [int(mode.split("-")[-1]) for mode in inpt.TopMassNjets for i in inpt.TopMassNjets[mode]]
    m = PlotTemplateTH2F(inpt) 
    m.xData = xdata
    m.yData = ydata
    m.Title = "Reconstructed Invariant Mass of Truth Jet derived Tops"
    m.xTitle = "Invariant Mass (GeV)"
    m.yTitle = "n-Truth Jets"
    m.yStep = 1
    m.xStep = 50
    m.xBins = 200
    m.xMax = 250
    m.xMin = 100
    m.Filename = "Figure_3.1c" 
    m.SaveFigure()


    hist = PlotTemplate(inpt)
    hist["Title"] = "Reconstructed Invariant Mass of Tops from Truth Jets \n Partitioned Into n-Tops Contributing to Any Matched Truth Jets"
    hist["xTitle"] = "Invariant Mass (GeV)"
    hist["IncludeOverflow"] = True
    hist["Logarithmic"] = True
    hist["xStep"] = 50
    hist["xMax"] = 400
    hist["yMin"] = 1
    hist["xBins"] = 400
    hist["Filename"] = "Figure_3.1d" 
    hist["Histograms"] = []
 
    for mode in inpt.TopMassMerged:
        m_hist = {}
        m_hist["Title"] = str(mode) + "-Tops"
        m_hist["xData"] = inpt.TopMassMerged[mode]
        hist["Histograms"].append(TH1F(**m_hist))

    com = CombineTH1F(**hist)
    com.SaveFigure() 

def TopTruthJetsKinematics(inpt):
    
    hist = PlotTemplate(inpt)
    hist["Title"] = "$\Delta R$ Between Truth Jets Matched to a Mutual \n Top Compared to Background"
    hist["xTitle"] = "$\Delta R$"
    hist["Histograms"] = []
    hist["Filename"] = "Figure_3.1f"
    hist["xMax"] = 4
    hist["xStep"] = 0.25
    hist["xBins"] = 250
    hist["Logarithmic"] = True 
    hist["yMin"] = 1
    for dr in inpt.DeltaRTJ_:
        if "dR" not in dr: continue
        plt = {}
        plt["Title"] = dr.replace("-dR", "")
        plt["xData"] = inpt.DeltaRTJ_[dr]
        plt["xBins"] = 250
        hist["Histograms"].append(TH1F(**plt))
    
    com = CombineTH1F(**hist) 
    com.SaveFigure() 

    hist2D = PlotTemplateTH2F(inpt)
    hist2D.Title = "$\Delta R$ as a Function of the Truth Top Transverse Momenta" 
    hist2D.xTitle = "Transverse Momenta (GeV)"
    hist2D.yTitle = "$\Delta R$"
    hist2D.yMax = 4
    hist2D.xMax = 1500
    hist2D.yBins = 500
    hist2D.xBins = 500
    hist2D.xStep = 100
    hist2D.yStep = 0.4
    hist2D.yData = [i for m in inpt.DeltaRTJ_ for i in inpt.DeltaRTJ_[m] if "dR" in m]
    hist2D.xData = [i for m in inpt.DeltaRTJ_ for i in inpt.DeltaRTJ_[m] if "PT" in m]
    hist2D.Filename = "Figure_3.1g"
    hist2D.SaveFigure()

    hist2D = PlotTemplateTH2F(inpt)
    hist2D.Title = "$\Delta R$ as a Function of the Truth Top Energy" 
    hist2D.xTitle = "Energy (GeV)"
    hist2D.yTitle = "$\Delta R$"
    hist2D.yMax = 4
    hist2D.xMax = 1500
    hist2D.yBins = 500
    hist2D.xBins = 500
    hist2D.xStep = 100
    hist2D.yStep = 0.4
    hist2D.yData = [i for m in inpt.DeltaRTJ_ for i in inpt.DeltaRTJ_[m] if "dR" in m]
    hist2D.xData = [i for m in inpt.DeltaRTJ_ for i in inpt.DeltaRTJ_[m] if "Energy" in m]
    hist2D.Filename = "Figure_3.1h"
    hist2D.SaveFigure()


    xdata = {}
    for dr, sym in zip(inpt.TopTruthJet_parton["Parton-dR"], inpt.TopTruthJet_parton["Parton-symbol"]):
        if sym not in xdata: xdata[sym] = []
        xdata[sym].append(dr)
   
    hist = PlotTemplate(inpt)
    hist["Title"] = "$\Delta R$ Of Top Matched Truth Jet Ghost Parton Composition \n Partitioned into PDGID Symbol"
    hist["xTitle"] = "$\Delta R$"
    hist["Filename"] = "Figure_3.1i"
    hist["xMax"] = 4
    hist["xBins"] = 250
    hist["Logarithmic"] = True
    hist["yMin"] = 1
    hist["xStep"] = 0.25
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
    hist2D.yMax = 3.6
    hist2D.xMax = 5
    hist2D.xMin = -5
    hist2D.yBins = 250
    hist2D.xBins = 250
    hist2D.xStep = 0.5
    hist2D.yStep = 0.4
    hist2D.yData = [i for m in inpt.TopTruthJet_parton for i in inpt.TopTruthJet_parton[m] if "dR" in m]
    hist2D.xData = [i for m in inpt.TopTruthJet_parton for i in inpt.TopTruthJet_parton[m] if "eta" in m]
    hist2D.Filename = "Figure_3.2i"
    hist2D.SaveFigure()


    hist2D = PlotTemplateTH2F(inpt)
    hist2D.Title = "Truth Jet Ghost Matched Parton's $\Delta R$ \n as a Function of $\phi$"
    hist2D.xTitle = "Azimuthal Angle - $\phi$ (Radians)"
    hist2D.yTitle = "$\Delta R$"
    hist2D.yMax = 3.6
    hist2D.xMax = 3.2
    hist2D.xMin = -3.2
    hist2D.yBins = 250
    hist2D.xBins = 250
    hist2D.xStep = 0.4
    hist2D.yStep = 0.4
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
    hist["xMax"] = 400
    hist["xBins"] = 500
    hist["Logarithmic"] = True
    hist["IncludeOverflow"] = True
    hist["yMin"] = 1
    hist["xStep"] = 40
    hist["Histograms"] = []
    for sym in inpt.JetMassNTop:
        plt = {}
        plt["Title"] = str(sym) + "-Tops"
        plt["xData"] = inpt.JetMassNTop[sym]
        hist["Histograms"].append(TH1F(**plt))
    com = CombineTH1F(**hist)
    com.SaveFigure()

