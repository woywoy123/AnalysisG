
def DeltaRChildren(x):
    Plots = PlotTemplate(x)
    Plots["Title"] = "$\Delta$R Between Truth Children Originating from Mutual Top"
    Plots["xTitle"] = "$\Delta$R"
    Plots["xStep"] = 0.2
    Plots["xMax"] = 5
    Plots["xBins"] = 100
    Plots["Filename"] = "Figure_2.1d"
    Plots["Histograms"] = []
    
    for i in ["Res-DelR", "Spec-DelR"]:
        _Plots = {}
        _Plots["Title"] = i.split("-")[0]
        _Plots["xData"] = x.ChildrenCluster[i]
        Plots["Histograms"] += [TH1F(**_Plots)]
    
    plt = CombineTH1F(**Plots)
    plt.SaveFigure()
       
    Plots = PlotTemplate(x)
    Plots["Title"] = "$\Delta$R Between Parent Top and Hadronic-Leptonic Children."
    Plots["xTitle"] = "$\Delta$R"
    Plots["xStep"] = 0.2
    Plots["xMax"] = 4
    Plots["xBins"] = 100
    Plots["Filename"] = "Figure_2.1e"
    Plots["Histograms"] = []
    
    for i in ["Lep", "Had"]:
        _Plots = {}
        _Plots["Title"] = i
        _Plots["xData"] = x.TopChildrenCluster[i]
        Plots["Histograms"] += [TH1F(**_Plots)]
    
    X = CombineTH1F(**Plots)
    X.SaveFigure()

    Plots = PlotTemplate(x)
    Plots["Title"] = "$\Delta$R Clustering of Children Pair"
    Plots["xTitle"] = "$\Delta$R"
    Plots["xStep"] = 0.2
    Plots["xBins"] = 100
    Plots["xMax"] = 3
    Plots["Filename"] = "Figure_2.1f"
    Plots["Stack"] = True
    Plots["yTitle"] = "Fraction of Cases (%)"
    Plots["Normalize"] = "%"
    Plots["Histograms"] = []
   
    s = [
            "Correct-Top-Res-Res", "False-Top-Res-Res", 
            "Correct-Top-Spec-Spec", "False-Top-Res-Spec", 
            "False-Top-Spec-Spec"
    ]
    for i in s:
        _Plots = {}
        _Plots["Title"] = i.replace("-Top", "")
        _Plots["xData"] = x.MatchedChildren[i]
        Plots["Histograms"] += [TH1F(**_Plots)]
    
    X = CombineTH1F(**Plots)
    X.SaveFigure()

    Plots = PlotTemplate(x)
    Plots["Title"] = "$\Delta$R Between Decay Products of Mutual Top \n as a Function of Parent Top Transverse Momenta (Hadronic)"
    Plots["yTitle"] = "$\Delta$R"
    Plots["xTitle"] = "Transverse Momentum (GeV)"
    Plots["yStep"] = 0.25
    Plots["xStep"] = 100
    Plots["xBins"] = 500
    Plots["yBins"] = 500
    Plots["yMin"] = 0
    Plots["yMax"] = 5.0
    Plots["xMin"] = 0
    Plots["xMax"] = 1250
    Plots["yData"] = x.ChildrenCluster["Had-DelR"]
    Plots["xData"] = x.ChildrenCluster["Had-top-PT"]
    Plots["Filename"] = "Figure_2.1g"
    X = TH2F(**Plots)
    X.SaveFigure()

    Plots = PlotTemplate(x)
    Plots["Title"] = "$\Delta$R Between Decay Products of Mutual Top \n as a Function of Parent Top Transverse Momenta (Leptonic)"
    Plots["yTitle"] = "$\Delta$R"
    Plots["xTitle"] = "Transverse Momentum (GeV)"
    Plots["yStep"] = 0.25
    Plots["xStep"] = 100
    Plots["xBins"] = 500
    Plots["yBins"] = 500
    Plots["yMin"] = 0
    Plots["yMax"] = 5
    Plots["xMin"] = 0
    Plots["xMax"] = 1250
    Plots["yData"] = x.ChildrenCluster["Lep-DelR"]
    Plots["xData"] = x.ChildrenCluster["Lep-top-PT"]
    Plots["Filename"] = "Figure_2.1h"
    X = TH2F(**Plots)
    X.SaveFigure()

def TruthChildrenKinematics(x):
    Plots = PlotTemplate(x)
    Plots["Title"] = "Fractional Transverse Momenta Transferred\n to Decay Products from Parent Top"
    Plots["xTitle"] = "Fraction"
    Plots["xStep"] = 0.1
    Plots["xMax"] = 2
    Plots["xBins"] = 100
    Plots["yTitle"] = "Percentage of all Tops (%)"
    Plots["Filename"] = "Figure_2.1i"
    Plots["Stack"] = True
    Plots["Normalize"] = "%"
    Plots["Histograms"] = []
    
    for i in x.FractionalPT:
        _Plots = {}
        _Plots["Title"] = i
        _Plots["xData"] = x.FractionalPT[i]
        Plots["Histograms"] += [TH1F(**_Plots)]
    
    X = CombineTH1F(**Plots)
    X.SaveFigure()

    Plots = PlotTemplate(x)
    Plots["Title"] = "Fraction of Top Energy Transferred to Decay Products"
    Plots["xTitle"] = "Fraction"
    Plots["xStep"] = 0.1
    Plots["xMax"] = 2
    Plots["xBins"] = 100
    Plots["yTitle"] = "Percentage of all Tops (%)"
    Plots["Filename"] = "Figure_2.1j"
    Plots["Stack"] = True
    Plots["Normalize"] = "%"
    Plots["Histograms"] = []
    
    for i in x.FractionalEnergy:
        _Plots = {}
        _Plots["Title"] = i
        _Plots["xData"] = x.FractionalEnergy[i]
        Plots["Histograms"] += [TH1F(**_Plots)]
    
    X = CombineTH1F(**Plots)
    X.SaveFigure()

PDGID = { 1 : "d"        ,  2 : "u"             ,  3 : "s", 
          4 : "c"        ,  5 : "b"             , 11 : "e", 
         12 : "$\\nu_e$" , 13 : "$\mu$"         , 14 : "$\\nu_{\mu}$", 
         15 : "$\\tau$"  , 16 : "$\\nu_{\\tau}$", 21 : "g", 
         22 : "$\\gamma$"}

CounterPDGID = {"d"            : 0, "u"       : 0, "s"              : 0, "c"    : 0, 
                "b"            : 0, "e"       : 0, "$\\nu_e$"       : 0, "$\mu$": 0, 
                "$\\nu_{\mu}$" : 0, "$\\tau$" : 0, "$\\nu_{\\tau}$" : 0, "g"    : 0, 
                "$\\gamma$"    : 0}
 
_leptons = [11, 12, 13, 14, 15, 16]
_charged_leptons = [11, 13, 15]


def DeltaRLepB(Ana):

    nevents = 0
    lumi = 0
    DeltaR_lepB = {"sameTop": [], "differentTop": []}
    
    for ev in Ana:
        event = ev.Trees["nominal"]
        nevents += 1
        lumi += event.Lumi

        bquarks = [c for t in event.Tops for c in t.Children if PDGID[abs(c.pdgid)] == "b"]

        for t in event.Tops:
            
            lepton_thisTop = [c for c in t.Children if abs(c.pdgid) in _charged_leptons]
            if len(lepton_thisTop) != 1: continue
            for ib,b in enumerate(bquarks):
                if b in t.Children:
                    DeltaR_lepB["sameTop"].append(lepton_thisTop[0].DeltaR(b))
                else:
                    DeltaR_lepB["differentTop"].append(lepton_thisTop[0].DeltaR(b)) 

    Plots = PlotTemplate(nevents, lumi)
    Plots["Title"] = "$\Delta$R Between Lepton and B-quark"
    Plots["xTitle"] = "$\Delta$R"
    Plots["xStep"] = 0.2
    Plots["Filename"] = "Figure_2.1i"
    Plots["xScaling"] = 2.5
    Plots["Histograms"] = []
    
    for i in DeltaR_lepB:
        _Plots = {}
        _Plots["Title"] = i
        _Plots["xData"] = DeltaR_lepB[i]
        Plots["Histograms"] += [TH1F(**_Plots)]
    
    X = CombineTH1F(**Plots)
    X.SaveFigure()


def TopChildrenKinematics(Ana):
    
    nevents = 0
    lumi = 0
    kinB = {"pt": {"fromRes": [], "fromSpec": []}, "eta": {"fromRes": [], "fromSpec": []}, "phi": {"fromRes": [], "fromSpec": []}}
    kinL = {"pt": {"fromRes": [], "fromSpec": []}, "eta": {"fromRes": [], "fromSpec": []}, "phi": {"fromRes": [], "fromSpec": []}}

    for ev in Ana:
        event = ev.Trees["nominal"]
        nevents += 1
        lumi += event.Lumi
        for t in event.Tops:
            if t.FromRes: res = "fromRes"
            else: res = "fromSpec"
            for c in t.Children:
                if PDGID[abs(c.pdgid)] == "b":
                    kinB["pt"][res].append(c.pt/1000.)
                    kinB["eta"][res].append(c.eta)
                    kinB["phi"][res].append(c.phi)
                elif abs(c.pdgid) in _charged_leptons:
                    kinL["pt"][res].append(c.pt/1000.)
                    kinL["eta"][res].append(c.eta)
                    kinL["phi"][res].append(c.phi)
     
    Plots = PlotTemplate(nevents, lumi)
    Plots["Title"] = "Transverse momentum of b-quarks" 
    Plots["xTitle"] = "Transverse Momentum (GeV)"
    Plots["xBins"] = 100
    Plots["xMin"] = 0
    Plots["xMax"] = 1000
    Plots["Filename"] = "Figure_2.1k"
    Plots["Histograms"] = []
        
    for i in kinB["pt"]:
        _Plots = {}
        _Plots["Title"] = i
        _Plots["xData"] = kinB["pt"][i]
        Plots["Histograms"] += [TH1F(**_Plots)]
    X = CombineTH1F(**Plots)
    X.SaveFigure()

    Plots = PlotTemplate(nevents, lumi)
    Plots["Title"] = "Pseudorapidity of b-quarks" 
    Plots["xTitle"] = "Eta"
    Plots["xBins"] = 50
    Plots["xMin"] = -5
    Plots["xMax"] = 5
    Plots["Filename"] = "Figure_2.1l"
    Plots["Histograms"] = []
        
    for i in kinB["eta"]:
        _Plots = {}
        _Plots["Title"] = i
        _Plots["xData"] = kinB["eta"][i]
        Plots["Histograms"] += [TH1F(**_Plots)]
    X = CombineTH1F(**Plots)
    X.SaveFigure()

    Plots = PlotTemplate(nevents, lumi)
    Plots["Title"] = "Azimuth of b-quarks" 
    Plots["xTitle"] = "Phi"
    Plots["xBins"] = 70
    Plots["xMin"] = -3.5
    Plots["xMax"] = 3.5
    Plots["Filename"] = "Figure_2.1m"
    Plots["Histograms"] = []
        
    for i in kinB["phi"]:
        _Plots = {}
        _Plots["Title"] = i
        _Plots["xData"] = kinB["phi"][i]
        Plots["Histograms"] += [TH1F(**_Plots)]
    X = CombineTH1F(**Plots)
    X.SaveFigure()

    Plots = PlotTemplate(nevents, lumi)
    Plots["Title"] = "Transverse momentum of leptons" 
    Plots["xTitle"] = "Transverse Momentum (GeV)"
    Plots["xBins"] = 100
    Plots["xMin"] = 0
    Plots["xMax"] = 1000
    Plots["Filename"] = "Figure_2.1n"
    Plots["Histograms"] = []
        
    for i in kinL["pt"]:
        _Plots = {}
        _Plots["Title"] = i
        _Plots["xData"] = kinL["pt"][i]
        Plots["Histograms"] += [TH1F(**_Plots)]
    X = CombineTH1F(**Plots)
    X.SaveFigure()

    Plots = PlotTemplate(nevents, lumi)
    Plots["Title"] = "Pseudorapidity of leptons" 
    Plots["xTitle"] = "Eta"
    Plots["xBins"] = 50
    Plots["xMin"] = -5
    Plots["xMax"] = 5
    Plots["Filename"] = "Figure_2.1o"
    Plots["Histograms"] = []
        
    for i in kinL["eta"]:
        _Plots = {}
        _Plots["Title"] = i
        _Plots["xData"] = kinL["eta"][i]
        Plots["Histograms"] += [TH1F(**_Plots)]
    X = CombineTH1F(**Plots)
    X.SaveFigure()

    Plots = PlotTemplate(nevents, lumi)
    Plots["Title"] = "Azimuth of leptons" 
    Plots["xTitle"] = "Phi"
    Plots["xBins"] = 70
    Plots["xMin"] = -3.5
    Plots["xMax"] = 3.5
    Plots["Filename"] = "Figure_2.1p"
    Plots["Histograms"] = []
        
    for i in kinL["phi"]:
        _Plots = {}
        _Plots["Title"] = i
        _Plots["xData"] = kinL["phi"][i]
        Plots["Histograms"] += [TH1F(**_Plots)]
    X = CombineTH1F(**Plots)
    X.SaveFigure()

   



