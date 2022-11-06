from AnalysisTopGNN.Plotting import TH1F, CombineTH1F

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


def TruthChildrenAll(Ana):
    _leptons = [11, 12, 13, 14, 15, 16]

    NChildrenResTopLep = []
    NChildrenResTopHad = []
    
    NChildrenSpecTopLep = []
    NChildrenSpecTopHad = []

    lumi = 0 
    nevents = 0
    for i in Ana:
        event = i.Trees["nominal"]
        tops = event.TopPostFSR
        
        for t in tops:
            _lep = False
            
            if len([c for c in t.Children if abs(c.pdgid) in _leptons]) > 0:
                _lep = True
            
            if t.FromRes == 1 and _lep:
                NChildrenResTopLep.append(t)

            elif t.FromRes == 1 and _lep == False:
                NChildrenResTopHad.append(t)

            elif t.FromRes == 0 and _lep:
                NChildrenSpecTopLep.append(t)

            elif t.FromRes == 0 and _lep == False:
                NChildrenSpecTopHad.append(t) 

        lumi += event.Lumi
        nevents += 1
   
    NChildren(nevents, lumi, NChildrenResTopLep, NChildrenResTopHad, NChildrenSpecTopLep, NChildrenSpecTopHad, "1")
    Momentum(nevents, lumi, NChildrenResTopLep, NChildrenResTopHad, NChildrenSpecTopLep, NChildrenSpecTopHad, "1")
    InvariantMass(nevents, lumi, NChildrenResTopLep, NChildrenResTopHad, NChildrenSpecTopLep, NChildrenSpecTopHad, "1") 
    DeltaR(nevents, lumi, NChildrenResTopLep, NChildrenResTopHad, NChildrenSpecTopLep, NChildrenSpecTopHad, "1") 
    DeltaRTop(nevents, lumi, NChildrenResTopLep, NChildrenResTopHad, NChildrenSpecTopLep, NChildrenSpecTopHad, "1") 
    
def TruthChildrenAll(Ana):
    _leptons = [11, 12, 13, 14, 15, 16]

    NChildrenResTopLep = []
    NChildrenResTopHad = []
    
    NChildrenSpecTopLep = []
    NChildrenSpecTopHad = []

    lumi = 0 
    nevents = 0
    for i in Ana:
        event = i.Trees["nominal"]
        tops = event.TopPostFSR
        
        for t in tops:
            _lep = False
            
            if len([c for c in t.Children if abs(c.pdgid) in _leptons]) > 0:
                _lep = True
            
            if t.FromRes == 1 and _lep:
                NChildrenResTopLep.append(t)

            elif t.FromRes == 1 and _lep == False:
                NChildrenResTopHad.append(t)

            elif t.FromRes == 0 and _lep:
                NChildrenSpecTopLep.append(t)

            elif t.FromRes == 0 and _lep == False:
                NChildrenSpecTopHad.append(t) 

        lumi += event.Lumi
        nevents += 1
   
    NChildren(nevents, lumi, NChildrenResTopLep, NChildrenResTopHad, NChildrenSpecTopLep, NChildrenSpecTopHad, "1")
    Momentum(nevents, lumi, NChildrenResTopLep, NChildrenResTopHad, NChildrenSpecTopLep, NChildrenSpecTopHad, "1")
    InvariantMass(nevents, lumi, NChildrenResTopLep, NChildrenResTopHad, NChildrenSpecTopLep, NChildrenSpecTopHad, "1") 
    DeltaR(nevents, lumi, NChildrenResTopLep, NChildrenResTopHad, NChildrenSpecTopLep, NChildrenSpecTopHad, "1") 
    DeltaRTop(nevents, lumi, NChildrenResTopLep, NChildrenResTopHad, NChildrenSpecTopLep, NChildrenSpecTopHad, "1") 
 
def TruthChildrenHadron(Ana):
    _leptons = [11, 12, 13, 14, 15, 16]

    NChildrenResTopLep = []
    NChildrenResTopHad = []
    
    NChildrenSpecTopLep = []
    NChildrenSpecTopHad = []

    lumi = 0 
    nevents = 0
    for i in Ana:
        event = i.Trees["nominal"]
        tops = event.TopPostFSR
        
        skip = False
        for t in tops:
            _lep = False
            if len([c for c in t.Children if abs(c.pdgid) in _leptons and t.FromRes == 1]) > 0:
                skip = True
        if skip:
            continue

        for t in tops:
            _lep = False
        
            if len([c for c in t.Children if abs(c.pdgid) in _leptons]) > 0:
                _lep = True
            
            if t.FromRes == 1 and _lep:
                NChildrenResTopLep.append(t)

            elif t.FromRes == 1 and _lep == False:
                NChildrenResTopHad.append(t)

            elif t.FromRes == 0 and _lep:
                NChildrenSpecTopLep.append(t)

            elif t.FromRes == 0 and _lep == False:
                NChildrenSpecTopHad.append(t) 

        lumi += event.Lumi
        nevents += 1
   
    NChildren(nevents, lumi, NChildrenResTopLep, NChildrenResTopHad, NChildrenSpecTopLep, NChildrenSpecTopHad, "2")
    Momentum(nevents, lumi, NChildrenResTopLep, NChildrenResTopHad, NChildrenSpecTopLep, NChildrenSpecTopHad, "2")
    InvariantMass(nevents, lumi, NChildrenResTopLep, NChildrenResTopHad, NChildrenSpecTopLep, NChildrenSpecTopHad, "2") 
    DeltaR(nevents, lumi, NChildrenResTopLep, NChildrenResTopHad, NChildrenSpecTopLep, NChildrenSpecTopHad, "2") 
    DeltaRTop(nevents, lumi, NChildrenResTopLep, NChildrenResTopHad, NChildrenSpecTopLep, NChildrenSpecTopHad, "2") 
 






def NChildren(nevents, lumi, NChildrenResTopLep, NChildrenResTopHad, NChildrenSpecTopLep, NChildrenSpecTopHad, fig):
    Plots = PlotTemplate(nevents, lumi)
    if fig == "1":
        Plots["xTitle"] = "Number of Children"
        Plots["Title"] = "Res-Lep"
        Plots["xData"] = [len(t.Children) for t in NChildrenResTopLep]
        RL = TH1F(**Plots)

    Plots["Title"] = "Res-Had"
    Plots["xData"] = [len(t.Children) for t in NChildrenResTopHad]
    RH = TH1F(**Plots)

    Plots["Title"] = "Spec-Lep"
    Plots["xData"] = [len(t.Children) for t in NChildrenSpecTopLep]
    SL = TH1F(**Plots)

    Plots["Title"] = "Spec-Had"
    Plots["xData"] = [len(t.Children) for t in NChildrenSpecTopHad]
    SH = TH1F(**Plots)

    Plots["Title"] = "Number of Decay Products from Tops"
    Plots["xData"] = []
    Plots["Histograms"] = [RL, RH, SL, SH] if fig == "1" else [RH, SL, SH]
    Plots["Stack"] = True
    Plots["xBinCentering"] = True
    Plots["xStep"] = 1
    Plots["Filename"] = "Figure.2." + fig + "a"
    T2a = CombineTH1F(**Plots)
    T2a.SaveFigure()


def Momentum(nevents, lumi, NChildrenResTopLep, NChildrenResTopHad, NChildrenSpecTopLep, NChildrenSpecTopHad, fig):
    Plots = PlotTemplate(nevents, lumi)
    Plots["xTitle"] = "Transverse Momenta of Child (GeV)"
    Plots["xMax"] = 750
    
    merge = []
    if fig == "1":
        Plots["Title"] = "Res-Lep"
        Plots["xData"] = [c.pt/1000 for t in NChildrenResTopLep for c in t.Children]
        RL = TH1F(**Plots)
        merge.append(RL)

    Plots["Title"] = "Res-Had"
    Plots["xData"] = [c.pt/1000 for t in NChildrenResTopHad for c in t.Children]
    RH = TH1F(**Plots)
    merge.append(RH)

    Plots["Title"] = "Spec-Lep"
    Plots["xData"] = [c.pt/1000 for t in NChildrenSpecTopLep for c in t.Children]
    SL = TH1F(**Plots)
    merge.append(SL)

    Plots["Title"] = "Spec-Had"
    Plots["xData"] = [c.pt/1000 for t in NChildrenSpecTopHad for c in t.Children]
    SH = TH1F(**Plots)
    merge.append(SH)

    Plots["Title"] = "Transverse Momenta Distribution of Top Decay Products \n under Different Decay Modes"
    Plots["xData"] = []
    Plots["Histograms"] = merge
    Plots["Filename"] = "Figure.2." + fig + "b"
    Plots["Stack"] = True
    Plots["xStep"] = 50
    T2b = CombineTH1F(**Plots)
    T2b.SaveFigure()

def InvariantMass(nevents, lumi, NChildrenResTopLep, NChildrenResTopHad, NChildrenSpecTopLep, NChildrenSpecTopHad, fig):
    Plots = PlotTemplate(nevents, lumi)
    Plots["xMin"] = 120
    Plots["xMax"] = 200
    Plots["xTitle"] = "Invariant Mass (GeV)"

    merge = []
    if fig == "1":
        Plots["Title"] = "Res-Lep"
        Plots["xData"] = [sum(t.Children).CalculateMass() for t in NChildrenResTopLep if len(t.Children) != 0]
        RL = TH1F(**Plots)
        merge.append(RL)

    Plots["Title"] = "Res-Had"
    Plots["xData"] = [sum(t.Children).CalculateMass() for t in NChildrenResTopHad if len(t.Children) != 0]
    RH = TH1F(**Plots)
    merge.append(RH)

    Plots["Title"] = "Spec-Lep"
    Plots["xData"] = [sum(t.Children).CalculateMass() for t in NChildrenSpecTopLep if len(t.Children) != 0]
    SL = TH1F(**Plots)
    merge.append(SL)

    Plots["Title"] = "Spec-Had"
    Plots["xData"] = [sum(t.Children).CalculateMass() for t in NChildrenSpecTopHad if len(t.Children) != 0]
    SH = TH1F(**Plots)
    merge.append(SH)

    Plots["Title"] = "Invariant Mass of Reconstructed Top Quark from Decay Products \n under Different Decay Modes"
    Plots["xData"] = []
    Plots["Histograms"] = merge
    Plots["Filename"] = "Figure.2." + fig + "c"
    Plots["Stack"] = False
    Plots["xStep"] = 5
    T2c = CombineTH1F(**Plots)
    T2c.SaveFigure()

def DeltaR(nevents, lumi, NChildrenResTopLep, NChildrenResTopHad, NChildrenSpecTopLep, NChildrenSpecTopHad, fig):
    def dR(t):
        done = []
        out = []
        for c in t.Children:
            for x in t.Children:
                if x == c:
                    continue
                if x in done:
                    continue
                out.append(c.DeltaR(x))
            done.append(c)
        return out
    
    Plots = PlotTemplate(nevents, lumi)
    Plots["xTitle"] = "$\Delta$R Between Children (a.u.)"

    merge = []
    if fig == "1":
        Plots["Title"] = "Res-Lep"
        Plots["xData"] = [dr for t in NChildrenResTopLep for dr in dR(t)]
        RL = TH1F(**Plots)
        merge.append(RL)

    Plots["Title"] = "Res-Had"
    Plots["xData"] = [dr for t in NChildrenResTopHad for dr in dR(t)]
    RH = TH1F(**Plots)
    merge.append(RH)

    Plots["Title"] = "Spec-Lep"
    Plots["xData"] = [dr for t in NChildrenSpecTopLep for dr in dR(t)]
    SL = TH1F(**Plots)
    merge.append(SL)

    Plots["Title"] = "Spec-Had"
    Plots["xData"] = [dr for t in NChildrenSpecTopHad for dr in dR(t)]
    SH = TH1F(**Plots)
    merge.append(SH)

    Plots["Title"] = "$\Delta$R between Children of Common Top \n under Different Decay Modes"
    Plots["xData"] = []
    Plots["Histograms"] = merge
    Plots["Filename"] = "Figure.2." + fig + "d"
    Plots["xStep"] = 0.4
    Plots["Stack"] = True
    T2d = CombineTH1F(**Plots)
    T2d.SaveFigure()

def DeltaRTop(nevents, lumi, NChildrenResTopLep, NChildrenResTopHad, NChildrenSpecTopLep, NChildrenSpecTopHad, fig):
    def dRTop(t):
        out = []
        for c in t.Children:
            out.append(t.DeltaR(c))
        return out

    Plots = PlotTemplate(nevents, lumi)
    Plots["xTitle"] = "$\Delta$R Between Children and Top (a.u.)"

    merge = []
    if fig == "1":
        Plots["Title"] = "Res-Lep"
        Plots["xData"] = [dr for t in NChildrenResTopLep for dr in dRTop(t)]
        RL = TH1F(**Plots)
        merge.append(RL)

    Plots["Title"] = "Res-Had"
    Plots["xData"] = [dr for t in NChildrenResTopHad for dr in dRTop(t)]
    RH = TH1F(**Plots)
    merge.append(RH)

    Plots["Title"] = "Spec-Lep"
    Plots["xData"] = [dr for t in NChildrenSpecTopLep for dr in dRTop(t)]
    SL = TH1F(**Plots)
    merge.append(SL)

    Plots["Title"] = "Spec-Had"
    Plots["xData"] = [dr for t in NChildrenSpecTopHad for dr in dRTop(t)]
    SH = TH1F(**Plots)
    merge.append(SH)

    Plots["Title"] = "$\Delta$R between Children and Top of Origin \n under Different Decay Modes"
    Plots["xData"] = []
    Plots["Histograms"] = merge
    Plots["Filename"] = "Figure.2." + fig + "e"
    Plots["xStep"] = 0.4
    Plots["Stack"] = True
    T1d = CombineTH1F(**Plots)
    T1d.SaveFigure()


