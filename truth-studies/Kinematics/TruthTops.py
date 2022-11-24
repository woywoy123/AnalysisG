from AnalysisTopGNN.Plotting import TH1F, CombineTH1F

def PlotTemplate(nevents, lumi):
    Plots = {
                "Style" : "ATLAS",
                "NEvents" : nevents, 
                "ATLASLumi" : lumi,
                "OutputDirectory" : "./Figures/TruthTops", 
                "yTitle" : "Entries (a.u.)",
                "yMin" : 0,
            }
    return Plots

def TruthTopsAll(Ana):    
    _leptons = [11, 12, 13, 14, 15, 16]
    
    ResDecayMode = []

    TruthTopsRes = [] 
    TruthSpecTops = []
    ZPrimeMass = []

    ResonanceLepton = []
    ResonanceHadron = []
    SpectatorLepton = []
    SpectatorHadron = []
    
    NumTops = []
    lumi = 0
    nevents = 0
    for i in Ana:
        event = i.Trees["nominal"]
        lumi += event.Lumi 
        nevents += 1 
        Zprime = []
        
        skip = False
        for t in event.Tops:
            if t.FromRes == 0:
                continue
            _lep = False
            if sum([1 for c in t.Children if abs(c.pdgid) in _leptons]) > 0:
                skip = True
        ResDecayMode.append(0 if skip else 1)

        for t in event.Tops:

            if t.FromRes == 1:
                TruthTopsRes.append(t.CalculateMass())
                Zprime.append(t)
            else:
                TruthSpecTops.append(t.CalculateMass())
           
            _lep = False
            for c in t.Children:
                if abs(c.pdgid) in _leptons:
                    _lep = True
                    break
            
            if _lep and t.FromRes == 1:
                ResonanceLepton.append(t)

            elif _lep == False and t.FromRes == 1:
                ResonanceHadron.append(t)

            elif _lep and t.FromRes == 0:
                SpectatorLepton.append(t)

            elif _lep == False and t.FromRes == 0:
                SpectatorHadron.append(t)
        ZPrimeMass.append(sum(Zprime).CalculateMass())
        NumTops.append(len(event.Tops))
    
    PlotsInvariantMassTops(nevents, lumi, TruthTopsRes, TruthSpecTops, ZPrimeMass, NumTops, "1")
    PlotsDecayMode(nevents, lumi, ResonanceLepton, ResonanceHadron, SpectatorHadron, SpectatorLepton, "1")
    PlotsResDecayMode(nevents, lumi, ResDecayMode)


def TruthTopsHadron(Ana):
    _leptons = [11, 12, 13, 14, 15, 16]
   
    TruthTopsRes = [] 
    TruthSpecTops = []
    ZPrimeMass = []

    ResonanceLepton = []
    ResonanceHadron = []
    SpectatorLepton = []
    SpectatorHadron = []
    
    NumTops = []
    lumi = 0
    nevents = 0
    for i in Ana:
        event = i.Trees["nominal"]
       
        skip = False
        for t in event.Tops:
            _lep = False
            for c in t.Children:
                if abs(c.pdgid) in _leptons:
                    _lep = True
            if t.FromRes == 1 and _lep:
                skip = True
        if skip:
            continue
        lumi += event.Lumi 
        nevents += 1 
        Zprime = []
 
        for t in event.Tops:

            if t.FromRes == 1:
                TruthTopsRes.append(t.CalculateMass())
                Zprime.append(t)
            else:
                TruthSpecTops.append(t.CalculateMass())
           
            _lep = False
            for c in t.Children:
                if abs(c.pdgid) in _leptons:
                    _lep = True
                    break
            
            if _lep and t.FromRes == 1:
                ResonanceLepton.append(t)

            elif _lep == False and t.FromRes == 1:
                ResonanceHadron.append(t)

            elif _lep and t.FromRes == 0:
                SpectatorLepton.append(t)

            elif _lep == False and t.FromRes == 0:
                SpectatorHadron.append(t)
        ZPrimeMass.append(sum(Zprime).CalculateMass())
        NumTops.append(len(event.Tops))

    PlotsInvariantMassTops(nevents, lumi, TruthTopsRes, TruthSpecTops, ZPrimeMass, NumTops, "2")
    PlotsDecayMode(nevents, lumi, ResonanceLepton, ResonanceHadron, SpectatorHadron, SpectatorLepton, "2")


def PlotsInvariantMassTops(nevents, lumi, TruthTopsRes, TruthSpecTops, ZPrimeMass, NumTops, Fig):
    Plots = PlotTemplate(nevents, lumi)
    Plots["Title"] = "Invariant Mass of Truth Tops Originating\n from the Z' Resonance (1.5 TeV) (Final State Radiation)"
    Plots["xTitle"] = "Invariant Mass (GeV)"
    Plots["xData"] = TruthTopsRes
    Plots["xMin"] = 170
    Plots["xMax"] = 180
    Plots["xBins"] = 100
    Plots["Filename"] = "Figure.1." + Fig + "a"
    T1a = TH1F(**Plots)
    T1a.SaveFigure()

    Plots["Title"] = "Invariant Mass of Spectator Tops"
    Plots["Filename"] = "Figure.1." + Fig + "c"
    Plots["xData"] = TruthSpecTops
    T1c = TH1F(**Plots)
    T1c.SaveFigure()
    
    Plots["Title"] = "Invariant Mass of Z' Resonance (1.5 TeV) derived from Truth Tops"
    Plots["Filename"] = "Figure.1." + Fig + "b"
    Plots["xMin"] = 0
    Plots["xMax"] = 1800
    Plots["xStep"] = 100
    Plots["xData"] = ZPrimeMass
    T1b = TH1F(**Plots)
    T1b.SaveFigure()

    Plots["Title"] = "Number of Tops for Sampled Events"
    Plots["xTitle"] = "Number of Tops"
    Plots["xMin"] = 0
    Plots["xMax"] = None
    Plots["xData"] = NumTops
    Plots["xStep"] = 1
    Plots["xBinCentering"] = True
    Plots["Filename"] = "Figure.1." + Fig + "d"
    T1d = TH1F(**Plots)
    T1d.SaveFigure()


def PlotsDecayMode(nevents, lumi, ResonanceLepton, ResonanceHadron, SpectatorHadron, SpectatorLepton, Fig):
    Plots = PlotTemplate(nevents, lumi)
    Plots["Title"] = "Decay Modes of all Tops"
    Plots["xTickLabels"] = ["Res-Lep (" + str(len(ResonanceLepton)) + ")", 
                           "Res-Had (" + str(len(ResonanceHadron)) + ")", 
                           "Spec-Lep (" + str(len(SpectatorLepton)) + ")", 
                           "Spec-Had (" + str(len(SpectatorHadron)) + ")", 
                           "n-Top delta (" + str(abs(len(ResonanceLepton + ResonanceHadron) - len(SpectatorLepton + SpectatorHadron))) +")"]
    
    Plots["xTitle"] = "Decay Mode of Top (a.u.)"
    Plots["xData"] = [0, 1, 2, 3, 4]
    Plots["xWeights"] = [len(ResonanceLepton), 
                         len(ResonanceHadron), 
                         len(SpectatorLepton), 
                         len(SpectatorHadron), 
                         abs(len(ResonanceLepton + ResonanceHadron) - len(SpectatorLepton + SpectatorHadron))]
    Plots["xMin"] = 0
    Plots["xStep"] = 1
    
    Plots["xBinCentering"] = True
    Plots["Filename"] = "Figure.1." + Fig + "e"
    
    T1e = TH1F(**Plots)
    T1e.SaveFigure()

    Plots = {
                "Title" : None, 
                "xTitle" : "Transverse Momenta (GeV)", 
                "yTitle" : "Entries",
                "xBins" : 200,
                "Style" : "ATLAS",
                "ATLASLumi" : lumi,
                "NEvents" : nevents,
            }
    
    Plots["Title"] = "Res-Lep"
    Plots["xData"] = [t.pt/1000 for t in ResonanceLepton]
    RL = TH1F(**Plots)

    Plots["Title"] = "Res-Had"
    Plots["xData"] = [t.pt/1000 for t in ResonanceHadron]
    RH = TH1F(**Plots)

    Plots["Title"] = "Spec-Lep"
    Plots["xData"] = [t.pt/1000 for t in SpectatorLepton]
    SL = TH1F(**Plots)

    Plots["Title"] = "Spec-Had"
    Plots["xData"] = [t.pt/1000 for t in SpectatorHadron]
    SH = TH1F(**Plots)

    Plots["Title"] = "Transverse Momenta Distribution of Tops \nfrom Respective Decay Modes"
    Plots["xData"] = []
    Plots["OutputDirectory"] = "./Figures/TruthTops"
    Plots["Histograms"] = [RL, RH, SL, SH]
    Plots["Filename"] = "Figure.1." + Fig + "f"
    T1f = CombineTH1F(**Plots)
    T1f.SaveFigure()


def PlotsResDecayMode(nevents, lumi, ResDecayMode):
    Plots = PlotTemplate(nevents, lumi)
    Plots["Title"] = "Resonance Decay Modes"
    Plots["xData"] = [0, 1]
    Plots["xBinCentering"] = True

    Plots["xTickLabels"] = ["Hadronically", "Leptonically"]
    Plots["xWeights"] = [len([i for i in ResDecayMode if i == 1]), len([i for i in ResDecayMode if i == 0])]
    Plots["xStep"] = 1
    Plots["Filename"] = "Figure.1.1g"
    T1g = TH1F(**Plots)
    T1g.SaveFigure()
