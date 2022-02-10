from Functions.IO.IO import File, PickleObject, UnpickleObject
from Functions.Event.EventGenerator import EventGenerator
from Functions.Plotting.Histograms import TH2F, TH1F
from Functions.Particles.Particles import Particle

def TestTopShapes( Compiler = "CustomSignalSample.pkl", sample = ""):
    E = UnpickleObject(Compiler)

    Top_Mass = []
    Top_MassPreFSR = []
    Top_MassPostFSR = []

    Top_FromChildren_Mass = []
    Top_FromChildren_MassPostFSR = []

    Top_FromTruthJets = []
    Top_FromTruthJets_NoLeptons = []

    Top_FromJets_NoLeptons = []

    uniqueParticles = set()
    for i in E.Events:
        event = E.Events[i]["nominal"]
        tt = event.TruthTops
        tprf = event.TopPreFSR
        tpof = event.TopPostFSR

        d = {}
        for k in tt:
            k.CalculateMass()
            Top_Mass.append(k.Mass_GeV)

            k.CalculateMassFromChildren()
            Top_FromChildren_Mass.append(k.Mass_init_GeV)

            d[k.Index+1] = []
        
        for k in tprf:
            k.CalculateMass()
            Top_MassPreFSR.append(k.Mass_GeV)
 
        F = {}
        for k in tpof:
            k.CalculateMass()
            Top_MassPostFSR.append(k.Mass_GeV)

            k.CalculateMassFromChildren()
            Top_FromChildren_MassPostFSR.append(k.Mass_init_GeV)
            
            skip = False
            for j in k.Decay_init:
                uniqueParticles.add(j.pdgid)
                if abs(j.pdgid) in [11, 13, 15]:
                    skip = True
                    break

            if len(k.Decay_init) == 0:
                skip = True
            
            ignore = -1
            if skip == False:
                ignore = k.Index+1
                F[k.Index+1] = []
        
        for k in event.TruthJets:
            for t in k.GhostTruthJetMap:
                if t in d:
                    d[t].append(k)

                if t in F:
                    F[t].append(k)

        for k in d:
            P = Particle(True)
            P.Decay += d[k]
            P.CalculateMassFromChildren()
            Top_FromTruthJets.append(P.Mass_GeV)

        for k in F:
            P = Particle(True)
            P.Decay += F[k]
            P.CalculateMassFromChildren()
            Top_FromTruthJets_NoLeptons.append(P.Mass_GeV)
        
        F = {}
        for k in event.TruthJets:
            if len(k.Decay) == 0:
                continue
            for j in k.GhostTruthJetMap:
                if j == ignore:
                    continue
                if j not in F:
                    F[j] = []
                if k.Decay[0].Type != "jet":
                    continue

                F[j].append(k.Decay[0])
        
        for k in F:
            P = Particle(True)
            if len(F[k]) <= 1:
                continue
            P.Decay += F[k]
            P.CalculateMassFromChildren()
            Top_FromJets_NoLeptons.append(P.Mass_GeV)


    for i in uniqueParticles:
        print("--- FOUND ----> ", i)
 

    # Tops from Truth information figures 
    t = TH1F() 
    t.Title = "Mass of Truth Top using m_truth branch"
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.xBins = 250
    t.xMin = 0
    t.xMax = 250
    t.xData = Top_Mass
    t.Filename = "TruthTops.png"
    t.SaveFigure("Plots/TestCustomAnalysisTop" + sample)

    t = TH1F() 
    t.Title = "Mass of Top Based on Ghost Pre-FSR"
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.xBins = 250
    t.xMin = 0
    t.xMax = 250
    t.xData = Top_MassPreFSR
    t.Filename = "TruthTopsPreFSR.png"
    t.SaveFigure("Plots/TestCustomAnalysisTop" + sample)

    t = TH1F() 
    t.Title = "Mass of Top Based on Ghost Post-FSR"    
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.xBins = 250
    t.xMin = 0
    t.xMax = 250
    t.xData = Top_MassPostFSR
    t.Filename = "TruthTopsPostFSR.png"
    t.SaveFigure("Plots/TestCustomAnalysisTop" + sample)


    t = TH1F() 
    t.Title = "Mass of Truth Top using m_truth branch (Children)"
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.xBins = 250
    t.xMin = 0
    t.xMax = 250
    t.xData = Top_FromChildren_Mass
    t.Filename = "TruthTops_Children.png"
    t.SaveFigure("Plots/TestCustomAnalysisTop" + sample)

    t = TH1F() 
    t.Title = "Mass of Top Based on Ghost Post-FSR (Children)"    
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.xBins = 250
    t.xMin = 0
    t.xMax = 250
    t.xData = Top_FromChildren_MassPostFSR
    t.Filename = "TruthTopsPostFSR_Children.png"
    t.SaveFigure("Plots/TestCustomAnalysisTop" + sample)

    t = TH1F() 
    t.Title = "Mass of Top Based on Ghost Matched Truth Jets\n (Inclusive of Leptonic decaying Top)"    
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.xBins = 500
    t.xMin = 0
    t.xMax = 500
    t.xData = Top_FromTruthJets
    t.Filename = "TruthTops_GhostTruthJets.png"
    t.SaveFigure("Plots/TestCustomAnalysisTop" + sample)

    t = TH1F() 
    t.Title = "Mass of Top Based on Ghost Matched Truth Jets\n (Exclusive of Leptonic decaying Top)"    
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.xBins = 500
    t.xMin = 0
    t.xMax = 500
    t.xData = Top_FromTruthJets_NoLeptons
    t.Filename = "TruthTops_GhostTruthJets_NoLeptons.png"
    t.SaveFigure("Plots/TestCustomAnalysisTop" + sample)

    t = TH1F() 
    t.Title = "Mass of Top Based on Matched Jets\n (Exclusive of Leptonic decaying Top) and NJets > 1"    
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.xBins = 500
    t.xMin = 0
    t.xMax = 500
    t.xData = Top_FromJets_NoLeptons
    t.Filename = "TruthTops_Jets_NoLeptons.png"
    t.SaveFigure("Plots/TestCustomAnalysisTop" + sample)
    return True

def Test_tttt():
    TestTopShapes( "tttt.pkl", "_tttt" )
    return True

def Test_ttbar():
    TestTopShapes( "ttbar.pkl", "_ttbar" )
    return True

def Test_SingleTop():
    TestTopShapes( "SingleTop_S.pkl", "_SingleTop" )
    return True

