from AnalysisTopGNN.Plotting import TH1F, CombineTH1F, TH2F
from copy import copy

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

def PlotTemplate(nevents, lumi):
    Plots = {
                "yTitle" : "Entries (a.u.)",
                "xMin" : 0, 
                "yMin" : 0, 
                "xMax" : None,
                "xBins" : None,
                "OutputDirectory" : "./Figures/TruthJet", 
                "Style" : "ATLAS",
                "ATLASLumi" : lumi,
                "NEvents" : nevents
            }
    return Plots


def TruthJetPartons(Ana):
    TopTruthJetPartons = copy(CounterPDGID)
    PTFraction = { i : [] for i in CounterPDGID}
    DeltaRPartonTJ = { i : [] for i in CounterPDGID}
    TruthJetPT = { "Top" : [], "Background" : [] }

    nevents = 0
    lumi = 0
    for ev in Ana:
        event = ev.Trees["nominal"]
        nevents += 1
        lumi += event.Lumi
        
        topTJ = [l for t in event.Tops for l in t.TruthJets]
        for tj in event.TruthJets:
            pt = tj.pt
            for p in tj.Partons:
                TopTruthJetPartons[PDGID[abs(p.pdgid)]] += 1
                PTFraction[PDGID[abs(p.pdgid)]].append(p.pt/pt)
                DeltaRPartonTJ[PDGID[abs(p.pdgid)]].append(tj.DeltaR(p))
            
            if tj in topTJ:
                TruthJetPT["Top"].append(tj.pt/1000)
            else:
                TruthJetPT["Background"].append(tj.pt/1000)

    Plots = PlotTemplate(nevents, lumi)
    Plots["Title"] = "Parton Contributions of Truth Jets (GhostParton)"
    Plots["xTitle"] = "Parton Symbol"
    Plots["xStep"] = 1
    Plots["Filename"] = "Figure_3.1a"
    Plots["xData"] = [i for i  in range(len(TopTruthJetPartons))]
    Plots["xWeights"] = [i for i in TopTruthJetPartons.values()]
    Plots["xTickLabels"] = [i for i in TopTruthJetPartons]
    Plots["xBinCentering"] = True
    x = TH1F(**Plots)
    x.SaveFigure()

    Plots = PlotTemplate(nevents, lumi)
    Plots["Title"] = "Fractional PT Contribution to Truth Jet (Stacked)"
    Plots["xTitle"] = "Transverse Momenta ($PT_{parton}$/$PT_{tj}$)"
    Plots["xStep"] = 10
    Plots["Filename"] = "Figure_3.1b"
    Plots["Histograms"] = []
    Plots["Stack"] = True
    Plots["Logarithmic"] = True
    for i in PTFraction:
        _Plots = {}
        _Plots["Title"] = i
        _Plots["xData"] = PTFraction[i]
        Plots["Histograms"].append(TH1F(**_Plots))
    x = CombineTH1F(**Plots) 
    x.SaveFigure()

    Plots = PlotTemplate(nevents, lumi)
    Plots["Title"] = "$\Delta$R Between Contributing Parton and Truth Jet (Stacked)"
    Plots["xTitle"] = "$\Delta$R Between Parton and Truth Jet"
    Plots["xStep"] = 0.25
    Plots["xScaling"] = 2.5
    Plots["Filename"] = "Figure_3.1c"
    Plots["Histograms"] = []
    Plots["Stack"] = False
    Plots["Logarithmic"] = False
    for i in DeltaRPartonTJ:
        _Plots = {}
        _Plots["Title"] = i
        _Plots["xData"] = DeltaRPartonTJ[i]
        Plots["Histograms"].append(TH1F(**_Plots))
    x = CombineTH1F(**Plots) 
    x.SaveFigure()

    Plots = PlotTemplate(nevents, lumi)
    Plots["Title"] = "Transverse Momenta of Truth Jet Originating \n Matched to Truth Top vs Background (Stacked)"
    Plots["xTitle"] = "Transverse Momenta of Truth Jet (GeV)"
    Plots["xStep"] = 100
    Plots["Filename"] = "Figure_3.1d"
    Plots["Histograms"] = []
    Plots["Stack"] = True
    Plots["Logarithmic"] = True
    for i in TruthJetPT:
        _Plots = {}
        _Plots["Title"] = i
        _Plots["xData"] = TruthJetPT[i]
        Plots["Histograms"].append(TH1F(**_Plots))
    x = CombineTH1F(**Plots) 
    x.SaveFigure()

def PartonToChildTruthJet(Ana):
    MissedChild = {i : 0 for i in CounterPDGID}
    Nchildren = {i : 0 for i in CounterPDGID}

    nevents = 0
    lumi = 0
    for ev in Ana:
        event = ev.Trees["nominal"]
        nevents += 1
        lumi += event.Lumi
        
        ChildrenIndex = {}
        ChildrenTJIndex = {}
        for c in range(len(event.TopChildren)):
            ChildrenIndex[c] = event.TopChildren[c]
            ChildrenTJIndex[c] = []
            Nchildren[PDGID[abs(event.TopChildren[c].pdgid)]] += 1
        
        for c in event.TruthJetPartons:
            for tci in ChildrenIndex:
                tc = ChildrenIndex[tci]
                if tc in c.Parent:
                    ChildrenTJIndex[tci].append(c)
        
        for c in ChildrenTJIndex:
            cpart = ChildrenIndex[c]
            sym = PDGID[abs(cpart.pdgid)]
            c = ChildrenTJIndex[c]
            if len(c) == 0:
                MissedChild[sym] += 1
                continue


    Plots = PlotTemplate(nevents, lumi)
    Plots["Title"] = "Top Child Lost in Truth Jets Contributions"
    Plots["xTitle"] = "Symbol"
    Plots["Filename"] = "Figure_3.1e"
    Plots["xStep"] = 1
    Plots["yTitle"] = "Percentage of lost children (%) by category"
    Plots["xData"] = [i for i in range(len(MissedChild))]
    Plots["xTickLabels"] = [i for i in MissedChild]
    Plots["xWeights"] = [float(MissedChild[i]/Nchildren[i])*100 if Nchildren[i] > 0 else 1 for i in MissedChild]
    Plots["xBinCentering"] = True
    x = TH1F(**Plots) 
    x.SaveFigure()

def ReconstructedTopMassTruthJet(Ana):
    TopMassPartons = {"Had" : [], "Lep" : []} 
    TopMassTruthJet = {"Had" : [], "Lep" : []} 
    ResonanceMass = {"Had-Had" : [], "Had-Lep" : [], "Lep-Lep" : []}

    nevents = 0
    lumi = 0

    resLost = 0
    resTopsLost = 0
    n_resTops = 0

    specTopsLost = 0
    n_specTops = 0
    
    TopsLost = 0
    ntops = 0

    for ev in Ana:
        event = ev.Trees["nominal"]
        nevents += 1
        lumi += event.Lumi

        stringR = {"Had" : [], "Lep" : []}
        for t in event.Tops:

            lp = "Lep" if sum([1 for c in t.Children if abs(c.pdgid) in _leptons]) > 0 else "Had"
           
            topTJ = [x for x in t.TruthJets]
            topTJ += [c for c in t.Children if abs(c.pdgid) in _leptons]

            topTJ = sum(topTJ)
           
            n_resTops += 1 if t.FromRes else 0
            n_specTops += 0 if t.FromRes else 1
            
            ntops += 1
            if topTJ == 0:
                TopsLost += 1
                resTopsLost += 1 if t.FromRes else 0 
                specTopsLost += 0 if t.FromRes else 1
                continue
            
            TopMassTruthJet[lp].append(topTJ.CalculateMass())
            if t.FromRes == 1:
                stringR[lp].append(topTJ)
 
        res = sum([t for l in stringR for t in stringR[l]])
        if len([t for l in stringR for t in stringR[l]]) < 2:
            resLost += 1
            continue
        ResonanceMass["-".join([k for k in stringR for p in stringR[k]])] += [res.CalculateMass()]


    Plots = PlotTemplate(nevents, lumi)
    Plots["Title"] = "Reconstructed Invariant Top Mass from Truth Jets \n (For Leptonic the Neutrinos are Included)"
    Plots["xTitle"] = "Invariant Mass (GeV)"
    Plots["xBins"] = 200
    Plots["xMax"] = 800
    Plots["Filename"] = "Figure_3.1f"
    Plots["Histograms"] = []
    for i in TopMassTruthJet:
        _Plots = {}
        _Plots["Title"] = i
        _Plots["xData"] = TopMassTruthJet[i]
        Plots["Histograms"].append(TH1F(**_Plots))
    x = CombineTH1F(**Plots) 
    x.SaveFigure()

    Plots = PlotTemplate(nevents, lumi)
    Plots["Title"] = "Reconstructed Invariant Scalar H Mass from Truth Jets\n (For Leptonic the Neutrinos are Included)"
    Plots["xTitle"] = "Invariant Mass (GeV)"
    Plots["xStep"] = 100
    Plots["xMax"] = 2000
    Plots["xScaling"] = 2.5
    Plots["Filename"] = "Figure_3.1g"
    Plots["Histograms"] = []
    Plots["Stack"] = True
    for i in ResonanceMass:
        _Plots = {}
        _Plots["Title"] = i
        _Plots["xData"] = ResonanceMass[i]
        Plots["Histograms"].append(TH1F(**_Plots))
    x = CombineTH1F(**Plots) 
    x.SaveFigure()

    print("(Max Efficiency - Resonance Reconstruction) - " +  str(round(float(1 - float(resLost)/nevents)*100, 2)) + "%")
    print("(Max Efficiency - Resonance Tops) - " + str(round(float(1 - float(resTopsLost)/n_resTops)*100, 2)) + "%")
    print("(Max Efficiency - Spectator Tops) - " + str(round(float(1 - float(specTopsLost)/n_specTops)*100, 2)) + "%")
    print("(Max Efficiency - Tops) - " + str(round(float(1 - float(TopsLost)/ntops)*100, 2)) + "%")
    print("(Cross Section) - Resonance) - " + str(float((nevents - resLost)/lumi)*0.000001) + "fb")

    return {"ResEff"  : float(1 - float(resLost)/nevents)*100, 
            "ResTop"  : float(1 - float(resTopsLost)/n_resTops)*100, 
            "SpecTop" : float(1 - float(specTopsLost)/n_specTops)*100, 
            "Tops"    : float(1 - float(TopsLost)/ntops)*100, 
            "x-sec"   : float((nevents - resLost)/lumi)*0.000001}


