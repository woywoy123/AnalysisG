from AnalysisTopGNN.Plotting import TH1F, CombineTH1F, TH2F
from copy import copy

PDGID = { 1 : "d"        ,  2 : "u"             ,  3 : "s", 
          4 : "c"        ,  5 : "b"             , 11 : "e", 
         12 : "$\\nu_e$" , 13 : "$\mu$"         , 14 : "$\\nu_{\mu}$", 
         15 : "$\\tau$"  , 16 : "$\\nu_{\\tau}$", 22 : "g"}

CounterPDGID = {"d"            : 0, "u"       : 0, "s"              : 0, "c"    : 0, 
                "b"            : 0, "e"       : 0, "$\\nu_e$"       : 0, "$\mu$": 0, 
                "$\\nu_{\mu}$" : 0, "$\\tau$" : 0, "$\\nu_{\\tau}$" : 0, "g"    : 0}
 
_leptons = [11, 12, 13, 14, 15, 16]


def PlotTemplate(nevents, lumi):
    Plots = {
                "yTitle" : "Entries (a.u.)",
                "xMin" : 0, 
                "yMin" : 0, 
                "xMax" : None,
                "xBins" : None,
                "OutputDirectory" : "./Figures/TopChildren", 
                "Style" : "ATLAS",
                "ATLASLumi" : lumi,
                "NEvents" : nevents
            }
    return Plots

def TopChildrenPDGID(Ana):
    TopChildrenPDGID = copy(CounterPDGID)
    
    nevents = 0
    lumi = 0
    tops = 0
    for ev in Ana:
        event = ev.Trees["nominal"]
        nevents += 1
        lumi += event.Lumi
        for t in event.Tops:
            tops += 1
            for c in t.Children:
                pdg = PDGID[abs(c.pdgid)]
                TopChildrenPDGID[pdg] += 1
     
    Plots = PlotTemplate(nevents, lumi)
    Plots["Title"] = "Fermionic Decay Products of Tops" 
    Plots["xTitle"] = "Fermion Symbol"
    Plots["xData"] = [i for i in range(len(TopChildrenPDGID))]
    Plots["xTickLabels"] = list(TopChildrenPDGID)
    Plots["xBinCentering"] = True 
    Plots["xStep"] = 1
    Plots["xWeights"] = [float(i / tops) for i in TopChildrenPDGID.values()]
    Plots["Filename"] = "Figure_2.1a"
    Plots["Normalize"] = False
    Plots["yTitle"] = "Fraction of Times Top Decays Into PDGID"
    x = TH1F(**Plots)
    x.SaveFigure()

def ReconstructedMassFromChildren(Ana):
    TopMass = {"Had" : [], "Lep" : []}
    ResonanceMass = {"Had-Had" : [], "Had-Lep" : [], "Lep-Lep" : []}

    nevents = 0
    lumi = 0
    tops = 0
    for ev in Ana:
        event = ev.Trees["nominal"]
        nevents += 1
        lumi += event.Lumi
        
        stringR = {"Had" : [], "Lep" : []}
        for t in event.Tops:

            lp = "Lep" if sum([1 for c in t.Children if abs(c.pdgid) in _leptons]) > 0 else "Had"
            top = sum(t.Children)
            if top == 0:
                continue
            TopMass[lp].append(top.CalculateMass()) 
            if t.FromRes == 1:
                stringR[lp].append(top)
        
        res = sum([t for l in stringR for t in stringR[l]])
        ResonanceMass["-".join([k for k in stringR for p in stringR[k]])] += [res.CalculateMass()]

    Plots = PlotTemplate(nevents, lumi)
    Plots["Title"] = "Reconstructed Invariant Top Mass from Immediate Decay Products"
    Plots["xTitle"] = "Invariant Top Mass (GeV)"
    Plots["xStep"] = 10
    Plots["Filename"] = "Figure_2.1b"
    Plots["Histograms"] = []

    for i in TopMass:
        _Plots = {}
        _Plots["Title"] = i
        _Plots["xData"] = TopMass[i]
        Plots["Histograms"].append(TH1F(**_Plots))
    
    X = CombineTH1F(**Plots)
    X.SaveFigure()

    Plots = PlotTemplate(nevents, lumi)
    Plots["Title"] = "Reconstructed Invariant Scalar H Mass from Top Decay Products"
    Plots["xTitle"] = "Invariant Scalar H Mass (GeV)"
    Plots["xStep"] = 100
    Plots["Filename"] = "Figure_2.1c"
    Plots["Histograms"] = []

    for i in ResonanceMass:
        _Plots = {}
        _Plots["Title"] = i
        _Plots["xData"] = ResonanceMass[i]
        Plots["Histograms"].append(TH1F(**_Plots))
    
    X = CombineTH1F(**Plots)
    X.SaveFigure()


def DeltaRChildren(Ana):
    ChildrenCluster = {"Had" : [], "Lep" : []}
    TopChildrenCluster = {"Had" : [], "Lep" : []}
    ChildrenClusterPT = {"DelR" : [], "PT" : []} 


    nevents = 0
    lumi = 0
    tops = 0
    for ev in Ana:
        event = ev.Trees["nominal"]
        nevents += 1
        lumi += event.Lumi
        
        for t in event.Tops:
            lp = "Lep" if sum([1 for c in t.Children if abs(c.pdgid) in _leptons]) > 0 else "Had"
            com = []
            for c in t.Children:
                for c2 in t.Children:
                    if c2 == c or c2 in com:
                        continue
                    ChildrenCluster[lp] += [c2.DeltaR(c)] 
                    ChildrenClusterPT["DelR"] += [c2.DeltaR(c)]
                    ChildrenClusterPT["PT"] += [t.pt /1000]

                com.append(c)
            TopChildrenCluster[lp] += [t.DeltaR(c) for c in t.Children]          

    Plots = PlotTemplate(nevents, lumi)
    Plots["Title"] = "$\Delta$R Between Decay Products of Mutual Top"
    Plots["xTitle"] = "$\Delta$R"
    Plots["xStep"] = 0.2
    Plots["Filename"] = "Figure_2.1d"
    Plots["xScaling"] = 2.5
    Plots["Histograms"] = []
    
    for i in ChildrenCluster:
        _Plots = {}
        _Plots["Title"] = i
        _Plots["xData"] = ChildrenCluster[i]
        Plots["Histograms"] += [TH1F(**_Plots)]
    
    X = CombineTH1F(**Plots)
    X.SaveFigure()


    Plots = PlotTemplate(nevents, lumi)
    Plots["Title"] = "$\Delta$R Between Parent Top and Decay Products"
    Plots["xTitle"] = "$\Delta$R"
    Plots["xStep"] = 0.2
    Plots["Filename"] = "Figure_2.1e"
    Plots["xScaling"] = 2.5
    Plots["Histograms"] = []
    
    for i in TopChildrenCluster:
        _Plots = {}
        _Plots["Title"] = i
        _Plots["xData"] = TopChildrenCluster[i]
        Plots["Histograms"] += [TH1F(**_Plots)]
    
    X = CombineTH1F(**Plots)
    X.SaveFigure()


    Plots = PlotTemplate(nevents, lumi)
    Plots["Title"] = "$\Delta$R Between Decay Products of Mutual Top \n as a Function of Parent Top Transverse Momenta"
    Plots["yTitle"] = "$\Delta$R"
    Plots["xTitle"] = "Transverse Momentum (GeV)"
    Plots["yStep"] = 0.2
    Plots["xStep"] = 25
    Plots["xScaling"] = 2.5
    Plots["yScaling"] = 2
    Plots["xMin"] = 0
    Plots["yMin"] = 0
    Plots["yData"] = ChildrenClusterPT["DelR"]
    Plots["xData"] = ChildrenClusterPT["PT"]
    Plots["Filename"] = "Figure_2.1g"
    X = TH2F(**Plots)
    X.SaveFigure()













