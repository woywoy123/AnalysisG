from Functions.IO.IO import File, PickleObject, UnpickleObject
from Functions.Event.EventGenerator import EventGenerator
from Functions.Plotting.Histograms import TH2F, TH1F, CombineHistograms
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

def TestJets(Sample, Name):

    E = UnpickleObject(Sample)

    TopMass_Top = []
    TopMass_Ghost = []

    for i in E.Events:
        tops = E.Events[i]["nominal"].TopPostFSR
        
        for t in tops:
            skip = False
            for c in t.Decay_init:
                if abs(c.pdgid) in [11, 13, 15]:
                    skip = True


            T = Particle(True)
            TJ = Particle(True)

            if len(t.Decay) == 0:
                continue
            for obj in t.Decay:
                T.Decay += obj.Decay
                TJ.Decay.append(obj)

                if len(obj.Decay) != 1:
                    skip = True
            
            if skip:
                continue           

            T.CalculateMassFromChildren()
            if T.Mass_GeV > 0:
                TopMass_Top.append(T.Mass_GeV)
 
            TJ.CalculateMassFromChildren()
            if TJ.Mass_GeV > 0:
                TopMass_Ghost.append(TJ.Mass_GeV)


    t = TH1F() 
    t.Title = "Mass of Top Based on Truth Jets (Non Leptonic)"    
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.xBins = 250
    t.xMin = 0
    t.xMax = 500
    t.xData = TopMass_Ghost
    t.Filename = "TopMass_TruthJet.png"
    t.SaveFigure("Plots/TestJetMatchingScheme" + Name)

    t = TH1F() 
    t.Title = "Mass of Top Based on Truth Jets Matched to Reconstructed Jets (Non Leptonic)"    
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.xBins = 250
    t.xMin = 0
    t.xMax = 500
    t.xData = TopMass_Top
    t.Filename = "TruthTops_Ghost.png"
    t.SaveFigure("Plots/TestJetMatchingScheme" + Name)

def Test_tttt_Jets():
    TestJets("CustomSignalSample.pkl", "_tttt")
    return True

def Test_SimilarityCustomOriginalMethods():
    def CollectTopsMass(tops):
        out = []
        for t in tops:
            t.CalculateMass()
            out.append(t.Mass_GeV)
        return out
    
    def CollectChildrenTop(tops):
        out = []
        for t in tops:
            t.CalculateMassFromChildren()
            out.append(t.Mass_init_GeV)
        return out

    def LeptonInTop(top):
        for c in t.Decay_init:
            if abs(c.pdgid) in [11, 12, 13, 14, 15, 16]:
                return True 
        return False                

    Tops_mTruth = []
    Tops_PreFSR = []
    Tops_PostFSR = []

    Children_mTruth = []
    Children_PostFSR = []

    TruthJet = []
    TruthJet_NoLep = []

    Status_Code = []
    Children_mTruth_Weird = []
    Children_PostFSR_Weird = []
    TruthJet_Weird = []

    TopTruthJetShare_0 = []
    TopTruthJetShare_1 = []
    TopTruthJetShare_2 = []
    TopTruthJetShare_3 = []

    top_energy_nJet_x = []
    top_energy_nJet_y = []

    TruthJetsShared = []
    
    Jets = []
    JetsSharedWithTruthJets_x = []
    JetsSharedWithTruthJets_y = []

    JetsSharedTruthTops_x = []
    JetsSharedTruthTops_y = []

    JetsSharedZPRIME_x = []
    JetsSharedZPRIME_y = []
    
    ZPrimeMass = []

    
    from Functions.IO.Files import Directories 
    d = Directories("CustomSample_Cache")
    Files = d.ListFilesInDir("CustomSample_Cache")
    for F in Files:
        E_C = UnpickleObject("CustomSample_Cache/" + F)
        print(F)
        for i in E_C.Events:
            ev = E_C.Events[i]["nominal"]
            
            skip = False
            for t in ev.TopPreFSR:
                if t.Status != 22:
                    skip = True
            if skip == True:
                Children_mTruth_Weird += CollectChildrenTop(ev.TruthTops)
                Children_PostFSR_Weird += CollectChildrenTop(ev.TopPostFSR)
                for t, j in zip(ev.TopPostFSR, ev.TopPreFSR):
                    TruthJet_Weird.append(t.Mass_GeV)
                    Status_Code.append(j.Status)
                continue 

            Tops_mTruth += CollectTopsMass(ev.TruthTops)
            Tops_PreFSR += CollectTopsMass(ev.TopPreFSR)
            Tops_PostFSR += CollectTopsMass(ev.TopPostFSR)

            Children_mTruth += CollectChildrenTop(ev.TruthTops)
            Children_PostFSR += CollectChildrenTop(ev.TopPostFSR)
            
            # Includes Lepton
            for t in ev.TopPostFSR:
                TruthJet.append(t.Mass_GeV)
            
            # Removes Lepton
            TJets = []
            for t in ev.TopPostFSR:
                for l in t.Decay:
                    if l not in TJets:
                        TJets.append(l)
                if LeptonInTop(t):
                    continue

                TruthJet_NoLep.append(t.Mass_GeV) 
            
            # Find Truth Jets being shared by the same tops
            Reuse = [-1 for l in range(len(TJets))]
            for t in ev.TopPostFSR:
                for j in t.Decay:
                    Reuse[TJets.index(j)] += 1
           
            origin_t_0 = []
            for t in ev.TopPostFSR:
                
                if LeptonInTop(t):
                    continue
                shared = 0
                for j in t.Decay:
                    shared += Reuse[TJets.index(j)]

                if shared == 0:
                    TopTruthJetShare_0.append(t.Mass_GeV)
                if shared == 1:
                    TopTruthJetShare_1.append(t.Mass_GeV)
                if shared == 2:
                    TopTruthJetShare_2.append(t.Mass_GeV)
                if shared == 3:
                    TopTruthJetShare_3.append(t.Mass_GeV)
                    
                top_energy_nJet_x.append(t.e / 1000)
                top_energy_nJet_y.append(shared)

            TruthJetsShared += Reuse
            
            # Do the above with the Jets
            uniquejets = []
            for t in ev.TopPostFSR:
                skip = False
                if LeptonInTop(t):
                    skip = True

                Top = Particle(True)
                for jt in t.Decay:
                    for je in jt.Decay:
                        if not skip:
                            Top.Decay.append(je) 
                        
                        if je not in uniquejets:
                            uniquejets.append(je)
                if not skip: 
                    Top.CalculateMassFromChildren()
                    Jets.append(Top.Mass_GeV)


            Reuse = [-1 for l in range(len(uniquejets))]
            for t in ev.TopPostFSR:
                for jt in t.Decay:
                    for je in jt.Decay:
                        Reuse[uniquejets.index(je)] += 1
            
            JetsSharedWithTruthJets_x += [j.e/1000 for j in uniquejets]
            JetsSharedWithTruthJets_y += Reuse
            
            Z = Particle(True)
            for t in ev.TopPostFSR:

                if t.FromRes == 1:
                    Z.Decay.append(t)

                if LeptonInTop(t):
                    continue 

                share = 0
                for jt in t.Decay:
                    for j in jt.Decay:
                        share += Reuse[uniquejets.index(j)]
                JetsSharedTruthTops_x.append(t.e / 1000)
                JetsSharedTruthTops_y.append(share)

                if t.FromRes == 1:
                    JetsSharedZPRIME_x.append(t.e / 1000)
                    JetsSharedZPRIME_y.append(share)
            
            Z.CalculateMassFromChildren()
            ZPrimeMass.append(Z.Mass_GeV)
            
            E_C.Events[i]["nominal"] = []
   
    BackUp = {}
    BackUp["Tops_mTruth"] = Tops_mTruth                 
    BackUp["Tops_PreFSR"] = Tops_PreFSR
    BackUp["Tops_PostFSR"] = Tops_PostFSR
    BackUp["Children_mTruth"] = Children_mTruth
    BackUp["Children_PostFSR"] = Children_PostFSR
    BackUp["TruthJet"] = TruthJet
    BackUp["TruthJet_NoLep"] = TruthJet_NoLep
    BackUp["Status_Code"] = Status_Code
    BackUp["Children_mTruth_Weird"] = Children_mTruth_Weird
    BackUp["Children_PostFSR_Weird"] = Children_PostFSR_Weird
    BackUp["TruthJet_Weird"] = TruthJet_Weird
    BackUp["TopTruthJetShare_0"] = TopTruthJetShare_0
    BackUp["TopTruthJetShare_1"] = TopTruthJetShare_1
    BackUp["TopTruthJetShare_2"] = TopTruthJetShare_2
    BackUp["TopTruthJetShare_3"] = TopTruthJetShare_3
    BackUp["top_energy_nJet_x"] = top_energy_nJet_x
    BackUp["top_energy_nJet_y"] = top_energy_nJet_y
    BackUp["TruthJetsShared"] = TruthJetsShared
    BackUp["Jets"] = Jets
    BackUp["JetsSharedWithTruthJets_x"] = JetsSharedWithTruthJets_x
    BackUp["JetsSharedWithTruthJets_y"] = JetsSharedWithTruthJets_y
    BackUp["JetsSharedTruthTops_x"] = JetsSharedTruthTops_x
    BackUp["JetsSharedTruthTops_y"] = JetsSharedTruthTops_y
    BackUp["JetsSharedZPRIME_x"] = JetsSharedZPRIME_x
    BackUp["JetsSharedZPRIME_y"] = JetsSharedZPRIME_y
    BackUp["ZPrimeMass"] = ZPrimeMass
    PickleObject(BackUp, "Plot.pkl")

def Test_SimilarityCustomOriginalMethods_Plot():
    BackUp = UnpickleObject("Plot.pkl")
    
    # Illustration of consistent tops 
    T = CombineHistograms()
    T.DefaultDPI = 500
    T.DefaultScaling = 7
    T.LabelSize = 15
    T.FontSize = 10
    T.LegendSize = 10
    T.Title = "Invariant Top Mass Derived From Different Algorithms"
    T.Log = False

    t = TH1F() 
    t.Title = "mTruth"
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.Color = "black"
    t.xBins = 250
    t.xMin = 0
    t.xMax = 200
    t.xData = BackUp["Tops_mTruth"]
    t.Filename = "mTruth.png"
    t.SaveFigure("Plots/MatchingAlgorithm/Raw")
    
    tc = TH1F()
    tc.Title = "PreFSR (Alt-Algo)"
    tc.xTitle = "Mass (GeV)"
    tc.yTitle = "Entries"
    tc.xBins = 250
    tc.xMin = 0
    tc.xMax = 200
    tc.xData = BackUp["Tops_PreFSR"]
    tc.Filename = "TopPreFSR.png"
    tc.SaveFigure("Plots/MatchingAlgorithm/Raw")

    tc_in = TH1F()
    tc_in.Title = "PostFSR (Alt-Algo)"
    tc_in.xTitle = "Mass (GeV)"
    tc_in.yTitle = "Entries"
    tc_in.xBins = 250
    tc_in.xMin = 0
    tc_in.xMax = 200
    tc_in.xData = BackUp["Tops_PostFSR"]
    tc_in.Filename = "TopPostFSR.png"
    tc_in.SaveFigure("Plots/MatchingAlgorithm/Raw")

    T.Histograms = [t, tc, tc_in]
    T.Filename = "CompareTops.png"
    T.Save("Plots/MatchingAlgorithm")
    T = "" 

    # Illustration of consistent top mass based on truth children 
    T = CombineHistograms()
    T.DefaultDPI = 500
    T.DefaultScaling = 7
    T.LabelSize = 15
    T.FontSize = 10
    T.LegendSize = 10
    T.Title = "Invariant Mass of Top Derived from Children"
    T.Log = True

    tc = TH1F()
    tc.Title = "mTruth"
    tc.xTitle = "Mass (GeV)"
    tc.yTitle = "Entries"
    tc.xBins = 250
    tc.xMin = 0
    tc.xMax = 200
    tc.xData = BackUp["Children_mTruth"]
    tc.Filename = "Top_Children_mTruth.png"
    tc.SaveFigure("Plots/MatchingAlgorithm/Raw")

    tc_in = TH1F()
    tc_in.Title = "PostFSR (Alt-Algo)"
    tc_in.xTitle = "Mass (GeV)"
    tc_in.yTitle = "Entries"
    tc_in.xBins = 250
    tc_in.xMin = 0
    tc_in.xMax = 200
    tc_in.xData = BackUp["Children_PostFSR"]
    tc_in.Filename = "Top_Children_PostFSR.png"
    tc_in.SaveFigure("Plots/MatchingAlgorithm/Raw")

    T.Histograms = [tc, tc_in]
    T.Filename = "CompareChildren.png"
    T.Save("Plots/MatchingAlgorithm")
    T = ""
    
    # Overlaying the mass of top based on truth jet and children and Tops
    T = CombineHistograms()
    T.DefaultDPI = 500
    T.DefaultScaling = 7
    T.LabelSize = 15
    T.FontSize = 10
    T.LegendSize = 10
    T.Title = "Invariant Mass of Top from Decay (Proposed Algorithm)"
    T.Log = False

    tc = TH1F()
    tc.Title = "Tops"
    tc.xTitle = "Mass (GeV)"
    tc.yTitle = "Entries"
    tc.xBins = 500
    tc.xMin = 0
    tc.xMax = 500
    tc.xData = BackUp["Tops_PostFSR"]
    tc.Filename = "InvMassTopDecay_Top.png"
    tc.SaveFigure("Plots/MatchingAlgorithm/Raw")

    tc_in = TH1F()
    tc_in.Title = "Children"
    tc_in.xTitle = "Mass (GeV)"
    tc_in.yTitle = "Entries"
    tc_in.xBins = 500
    tc_in.xMin = 0
    tc_in.xMax = 500
    tc_in.xData = BackUp["Children_PostFSR"]
    tc_in.Filename = "InvMassTopDecay_Children.png"
    tc_in.SaveFigure("Plots/MatchingAlgorithm/Raw")

    tc_ = TH1F()
    tc_.Title = "Truth Jet"
    tc_.xTitle = "Mass (GeV)"
    tc_.yTitle = "Entries"
    tc_.xBins = 500
    tc_.xMin = 0
    tc_.xMax = 500
    tc_.xData = BackUp["TruthJet"]
    tc_.Filename = "InvMassTopDecay_TruthJet.png"
    tc_.SaveFigure("Plots/MatchingAlgorithm/Raw")

    tc_NL = TH1F()
    tc_NL.Title = "Truth Jet (No Lepton)"
    tc_NL.xTitle = "Mass (GeV)"
    tc_NL.yTitle = "Entries"
    tc_NL.xBins = 500
    tc_NL.xMin = 0
    tc_NL.xMax = 500
    tc_NL.xData = BackUp["TruthJet_NoLep"]
    tc_NL.Filename = "InvMassTopDecay_TruthJet_NOLEP.png"
    tc_NL.SaveFigure("Plots/MatchingAlgorithm/Raw")

    T.Histograms = [tc, tc_, tc_in, tc_NL]
    T.Filename = "CompareDecayChain.png"
    T.Save("Plots/MatchingAlgorithm")
    T = ""

    # Weird behaviour Children
    T = CombineHistograms()
    T.DefaultDPI = 500
    T.DefaultScaling = 7
    T.LabelSize = 15
    T.FontSize = 10
    T.LegendSize = 10
    T.Title = "Invariant Mass of Top from Decay (Weird Events)"
    T.Log = False

    tc = TH1F()
    tc.Title = "Children (mTruth)"
    tc.xTitle = "Mass (GeV)"
    tc.yTitle = "Entries"
    tc.xBins = 250
    tc.xMin = 0
    tc.xMax = 500
    tc.xData = BackUp["Children_mTruth_Weird"]
    tc.Filename = "InvMassTopDecay_Children_Weird.png"
    tc.SaveFigure("Plots/MatchingAlgorithm/Raw")

    tc_in = TH1F()
    tc_in.Title = "Children (Alt-Algo)"
    tc_in.xTitle = "Mass (GeV)"
    tc_in.yTitle = "Entries"
    tc_in.xBins = 250
    tc_in.xMin = 0
    tc_in.xMax = 500
    tc_in.xData = BackUp["Children_PostFSR_Weird"]
    tc_in.Filename = "InvMassTopDecay_Children_AA_Weird.png"
    tc_in.SaveFigure("Plots/MatchingAlgorithm/Raw")

    tc_ = TH1F()
    tc_.Title = "Truth Jet (Alt-Algo)"
    tc_.xTitle = "Mass (GeV)"
    tc_.yTitle = "Entries"
    tc_.xBins = 250
    tc_.xMin = 0
    tc_.xMax = 500
    tc_.xData = BackUp["TruthJet_Weird"]
    tc_.Filename = "InvMassTopDecay_TruthJet_Weird.png"
    tc_.SaveFigure("Plots/MatchingAlgorithm/Raw")

    T.Histograms = [tc, tc_, tc_in]
    T.Filename = "CompareDecayChainWeird.png"
    T.Save("Plots/MatchingAlgorithm")
    T = ""

    # Random Statistics
    # --- Codes
    tc = TH1F()
    tc.Title = "Status Codes"
    tc.xTitle = "Pythia Status"
    tc.yTitle = "Entries"
    tc.xBins = 100
    tc.xMin = 0
    tc.xMax = 100
    tc.xData = BackUp["Status_Code"]
    tc.Filename = "TopStatusCodes_Weird.png"
    tc.SaveFigure("Plots/MatchingAlgorithm/")

 
    # Overlaying the mass of top based on truth jet and children and Tops
    T = CombineHistograms()
    T.DefaultDPI = 500
    T.DefaultScaling = 7
    T.LabelSize = 15
    T.FontSize = 10
    T.LegendSize = 10
    T.Title = "Invariant Mass of Top From n-Shared Truth Jets (Non Leptonic)"
    T.Log = False

    tc = TH1F()
    tc.Title = "0"
    tc.xTitle = "Mass (GeV)"
    tc.yTitle = "Entries"
    tc.xBins = 500
    tc.xMin = 0
    tc.xMax = 500
    tc.xData = BackUp["TopTruthJetShare_0"]
    tc.Filename = "TruthJetShare_0.png"
    tc.SaveFigure("Plots/MatchingAlgorithm/Raw")

    tc_in = TH1F()
    tc_in.Title = "1"
    tc_in.xTitle = "Mass (GeV)"
    tc_in.yTitle = "Entries"
    tc_in.xBins = 500
    tc_in.xMin = 0
    tc_in.xMax = 500
    tc_in.xData = BackUp["TopTruthJetShare_1"]
    tc_in.Filename = "TruthJetShare_1.png"
    tc_in.SaveFigure("Plots/MatchingAlgorithm/Raw")

    tc_ = TH1F()
    tc_.Title = "2"
    tc_.xTitle = "Mass (GeV)"
    tc_.yTitle = "Entries"
    tc_.xBins = 500
    tc_.xMin = 0
    tc_.xMax = 500
    tc_.xData = BackUp["TopTruthJetShare_2"]
    tc_.Filename = "TruthJetShare_2.png"
    tc_.SaveFigure("Plots/MatchingAlgorithm/Raw")

    tc_NL = TH1F()
    tc_NL.Title = "3"
    tc_NL.xTitle = "Mass (GeV)"
    tc_NL.yTitle = "Entries"
    tc_NL.xBins = 500
    tc_NL.xMin = 0
    tc_NL.xMax = 500
    tc_NL.xData = BackUp["TopTruthJetShare_3"]
    tc_NL.Filename = "TruthJetShare_3.png"
    tc_NL.SaveFigure("Plots/MatchingAlgorithm/Raw")

    T.Histograms = [tc, tc_, tc_in, tc_NL]
    T.Filename = "TruthJetsShared_NoLep.png"
    T.Save("Plots/MatchingAlgorithm")
    T = ""

    tc_ = TH1F()
    tc_.Title = "Shared Truth Jets"
    tc_.xTitle = "n-Shared"
    tc_.yTitle = "Entries"
    tc_.xMin = 0
    tc_.xData = BackUp["TruthJetsShared"]
    tc_.Filename = "truthjetsshared.png"
    tc_.SaveFigure("Plots/MatchingAlgorithm/")

    # Overlaying the mass of top from all decay chain
    T = CombineHistograms()
    T.DefaultDPI = 500
    T.DefaultScaling = 7
    T.LabelSize = 15
    T.FontSize = 10
    T.LegendSize = 10
    T.Title = "Invariant Mass of Top from Decay (Proposed Algorithm - No Lepton)"
    T.Log = True

    tc = TH1F()
    tc.Title = "Tops"
    tc.xTitle = "Mass (GeV)"
    tc.yTitle = "Entries"
    tc.xBins = 500
    tc.xMin = 0
    tc.xMax = 500
    tc.xData = BackUp["Tops_PostFSR"]
    tc.Filename = "InvMassTopDecay_Top.png"
    tc.SaveFigure("Plots/MatchingAlgorithm/Raw")

    tc_in = TH1F()
    tc_in.Title = "Children"
    tc_in.xTitle = "Mass (GeV)"
    tc_in.yTitle = "Entries"
    tc_in.xBins = 500
    tc_in.xMin = 0
    tc_in.xMax = 200
    tc_in.xData = BackUp["Children_PostFSR"]
    tc_in.Filename = "InvMassTopDecay_Children.png"
    tc_in.SaveFigure("Plots/MatchingAlgorithm/Raw")

    tc_NL = TH1F()
    tc_NL.Title = "Truth Jet"
    tc_NL.xTitle = "Mass (GeV)"
    tc_NL.yTitle = "Entries"
    tc_NL.xBins = 500
    tc_NL.xMin = 0
    tc_NL.xMax = 500
    tc_NL.xData = BackUp["TruthJet_NoLep"]
    tc_NL.Filename = "InvMassTopDecay_TruthJet_NOLEP.png"
    tc_NL.SaveFigure("Plots/MatchingAlgorithm/Raw")

    tc_ = TH1F()
    tc_.Title = "Reco Jet"
    tc_.xTitle = "Mass (GeV)"
    tc_.yTitle = "Entries"
    tc_.xBins = 500
    tc_.xMin = 0
    tc_.xMax = 500
    tc_.xData = BackUp["Jets"]
    tc_.Filename = "InvMassTopDecay_RecoJet.png"
    tc_.SaveFigure("Plots/MatchingAlgorithm/Raw")

    T.Histograms = [tc, tc_, tc_in, tc_NL]
    T.Filename = "CompareDecayChain.png"
    T.Save("Plots/MatchingAlgorithm")
    T = ""

    tc_ = TH1F()
    tc_.Title = "Z' Mass"
    tc_.xTitle = "Mass (GeV)"
    tc_.yTitle = "Entries"
    tc_.xBins = 500
    tc_.xMin = 0
    tc_.xMax = 2000
    tc_.xData = BackUp["ZPrimeMass"]
    tc_.Filename = "ZPrimeMass.png"
    tc_.SaveFigure("Plots/MatchingAlgorithm")


    T = TH2F()
    T.Normalize = False
    T.Title = "Truth Jet Energy vs n-Jets Shared"
    T.xTitle = "Energy (GeV)"
    T.yTitle = "n-Jets"
    T.xMin = 0 
    T.xMax = 2500
    T.yMin = 0
    T.yMax = 5
    T.yBin = 6
    T.xBins = 100
    T.Filename = "EnergyTruthJets_vs_nSharedJets.png"
    T.xData = BackUp["JetsSharedWithTruthJets_x"]
    T.yData = BackUp["JetsSharedWithTruthJets_y"] 
    T.SaveFigure("Plots/MatchingAlgorithm/")
    T = ""



    T = TH2F()
    T.Normalize = False
    T.Title = "Truth Top Energy vs n-Jets Shared"
    T.xTitle = "Energy (GeV)"
    T.yTitle = "n-Jets"
    T.xMin = 0 
    T.xMax = 2500
    T.yMin = 0
    T.yMax = 5
    T.xBins = 100
    T.yBins = 6
    T.Filename = "EnergyTruthTops_vs_nSharedJets.png"
    T.xData = BackUp["JetsSharedTruthTops_x"]
    T.yData = BackUp["JetsSharedTruthTops_y"] 
    T.SaveFigure("Plots/MatchingAlgorithm/")
    T = ""


    T = TH2F()
    T.Normalize = False
    T.Title = "Energy of Z' Tops vs nJets Shared"
    T.xTitle = "Energy (GeV)"
    T.yTitle = "n-Jets Shared"
    T.xMin = 0 
    T.yMin = 0
    T.yMax = 6
    T.xBins = 100
    T.yBins = 7
    T.Filename = "ZPrimeTops_vs_nSharedJets.png"
    T.xData = BackUp["JetsSharedZPRIME_x"]
    T.yData = BackUp["JetsSharedZPRIME_y"]
    T.SaveFigure("Plots/MatchingAlgorithm/")
    T = ""


    T = TH2F()
    T.Normalize = False
    T.Title = "Truth Top Energy vs n-Truth Jets Shared"
    T.xTitle = "Energy (GeV)"
    T.yTitle = "n-Truth Jets"
    T.xMin = 0 
    T.yMin = 0
    T.yMax = 5
    T.xBins = 100
    T.yBins = 5
    T.Filename = "TopEnergySharedTruthJets.png"
    T.xData = BackUp["top_energy_nJet_x"]
    T.yData = BackUp["top_energy_nJet_y"]
    T.SaveFigure("Plots/MatchingAlgorithm/")
    T = ""







    return True
    
