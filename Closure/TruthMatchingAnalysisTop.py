from Functions.IO.IO import File, PickleObject, UnpickleObject
from Functions.Event.EventGenerator import EventGenerator
from Functions.Plotting.Histograms import TH2F, TH1F, CombineHistograms
from Functions.Particles.Particles import Particle
from math import sqrt

def Test_SimilarityCustomOriginalMethods(CM_Energy = ""):
    
    def R(p):
        return sqrt(p.eta * p.eta + p.phi * p.phi)
    
    def dR(p, x):
        return sqrt((p.eta - x.eta)**2 + (p.phi - x.phi)**2)




    BackUp = {}
    BackUp["Tops_mTruth"] = []             
    BackUp["Tops_PreFSR"] = []
    BackUp["Tops_PostFSR"] = []

    BackUp["Children_mTruth"] = []
    BackUp["Children_PostFSR"] = []

    BackUp["TruthJet_NoLep"] = []
    BackUp["TruthJet_WithLep"] = []
    BackUp["Jets_NoLep"] = []
    BackUp["Jets_Lep"] = []

    BackUp["Status_Code"] = []
   

    BackUp["TopTruthJetShare_0"] = []
    BackUp["TopTruthJetShare_1"] = [] 
    BackUp["TopTruthJetShare_2"] = [] 
    BackUp["TopTruthJetShare_3"] = [] 

    BackUp["TopJetShare_0"] = []
    BackUp["TopJetShare_1"] = [] 
    BackUp["TopJetShare_2"] = []
    BackUp["TopJetShare_3"] = [] 

    BackUp["Top_1TruthJets"] = []
    BackUp["Top_2TruthJets"] = []
    BackUp["Top_3TruthJets"] = []
    BackUp["Top_4TruthJets"] = []

    BackUp["Top_1Jets"] = []
    BackUp["Top_2Jets"] = []
    BackUp["Top_3Jets"] = []
    BackUp["Top_4Jets"] = []

    BackUp["Single_TruthJet_JetSimilarity_E_TJ"] = []
    BackUp["Single_TruthJet_JetSimilarity_E_J"] = [] 
    BackUp["Single_TruthJet_JetSimilarity_R_TJ"] = [] 
    BackUp["Single_TruthJet_JetSimilarity_R_J"] = []  
    BackUp["Single_TruthJet_JetSimilarity_PT_TJ"] = []
    BackUp["Single_TruthJet_JetSimilarity_PT_J"] = [] 
    
    BackUp["Single_TruthJet_JetsDeltaR"] = []

    BackUp["Two_TruthJet_JetSimilarity_E_TJ"] = []
    BackUp["Two_TruthJet_JetSimilarity_E_J"] = [] 
    BackUp["Two_TruthJet_JetSimilarity_R_TJ"] = [] 
    BackUp["Two_TruthJet_JetSimilarity_R_J"] = []  
    BackUp["Two_TruthJet_JetSimilarity_PT_TJ"] = []
    BackUp["Two_TruthJet_JetSimilarity_PT_J"] = []   

    BackUp["Two_TruthJet_JetsDeltaR"] = []

    from Functions.IO.Files import Directories 
    Files = []
    for i in ["a", "d", "e"]:
        dx = "_Cache/CustomSample_tttt_"+ CM_Energy + "_MC_" + i + "_Cache"
        d = Directories(dx)
        for f in d.ListFilesInDir(dx):
            Files.append(dx + "/" + f)
    #Files = []
    #Files.append("CustomSignalSample.pkl")
    for F in Files:
        print(F)
        E_C = UnpickleObject(F)
        for i in E_C.Events:
            ev = E_C.Events[i]["nominal"]
            
            if len(ev.TopPostFSR) != len(ev.TopPreFSR) != len(ev.TruthTops):
                continue
            for prefsr, postfsr, truth in zip(ev.TopPreFSR, ev.TopPostFSR, ev.TruthTops): 
                prefsr.CalculateMass(), postfsr.CalculateMass(), truth.CalculateMass()
                BackUp["Status_Code"].append(prefsr.Status)
                BackUp["Tops_PreFSR"].append(prefsr.Mass_GeV)
                BackUp["Tops_PostFSR"].append(postfsr.Mass_GeV)
                BackUp["Tops_mTruth"].append(truth.Mass_GeV)

                postfsr.CalculateMassFromChildren()
                truth.CalculateMassFromChildren()
                
                BackUp["Children_PostFSR"].append(postfsr.Mass_init_GeV)
                BackUp["Children_mTruth"].append(truth.Mass_init_GeV)
               
                if len(postfsr.Decay) == 0:
                    continue

                BackUp["TruthJet_WithLep"].append(postfsr.Mass_GeV)


                L = [j for j in postfsr.Decay_init if abs(j.pdgid) in [11, 13, 15]]
                if len(L) != 0 :
                    continue
                
                if len([g for g in postfsr.Decay if g.Type in ["mu", "el"]]) > 0:
                    continue

                BackUp["TruthJet_NoLep"].append(postfsr.Mass_GeV)
                
                nj = []
                for j in postfsr.Decay:
                    nj += j.GhostTruthJetMap
                nj = len(list(set(nj)))-1
                
                if nj <= 3:
                    BackUp["TopTruthJetShare_"+str(nj)].append(postfsr.Mass_GeV)

                nj = len(postfsr.Decay)
                if nj <= 4:
                    BackUp["Top_" + str(nj) + "TruthJets"].append(postfsr.Mass_GeV)
            
            cap = {}
            for j in ev.DetectorParticles:
                if j.Type == "jet":
                    ind = j.JetMapTops
                else:
                    ind = [j.Index]
                
                for k in ind:
                    if k not in cap:
                        cap[k] = []
                    cap[k].append(j)

            
            for k in cap:
                P = Particle(True)
                P.Decay += cap[k]
                P.CalculateMassFromChildren()
                
                BackUp["Jets_Lep"].append(P.Mass_GeV)
                if len([g for g in cap[k] if g.Type in ["mu", "el"]]) > 0:
                    continue
                
                BackUp["Jets_NoLep"].append(P.Mass_GeV)


                nj = []
                for j in P.Decay:
                    nj += j.JetMapTops
                nj = len(list(set(nj)))-1
                
                if nj <= 3:
                    BackUp["TopJetShare_"+str(nj)].append(P.Mass_GeV)

                nj = len(P.Decay)
                if nj <= 4:
                    BackUp["Top_" + str(nj) + "Jets"].append(P.Mass_GeV)
            
            for tj in ev.TruthJets:
                if len(tj.Decay) == 1:
                    j = tj.Decay[0]
                    rj = ( (j.eta)**2 + (j.phi)**2 )**0.5
                    rtj = ( (tj.eta)**2 + (tj.phi)**2 )**0.5
                    dr = ( (j.eta - tj.eta)**2 + (j.phi - tj.phi)**2 )**0.5

                    BackUp["Single_TruthJet_JetSimilarity_E_TJ"].append(tj.e/1000)
                    BackUp["Single_TruthJet_JetSimilarity_E_J"].append(j.e/1000)
                    BackUp["Single_TruthJet_JetSimilarity_R_TJ"].append(rtj)
                    BackUp["Single_TruthJet_JetSimilarity_R_J"].append(rj)
                    BackUp["Single_TruthJet_JetSimilarity_PT_TJ"].append(tj.pt/1000)
                    BackUp["Single_TruthJet_JetSimilarity_PT_J"].append(j.pt/1000)

                    BackUp["Single_TruthJet_JetsDeltaR"].append(dr)


                if len(tj.Decay) == 2:
                    for j in tj.Decay:
                        rj = ( (j.eta)**2 + (j.phi)**2 )**0.5
                        rtj = ( (tj.eta)**2 + (tj.phi)**2 )**0.5
                        dr = ( (j.eta - tj.eta)**2 + (j.phi - tj.phi)**2 )**0.5

                        BackUp["Two_TruthJet_JetSimilarity_E_TJ"].append(tj.e/1000)
                        BackUp["Two_TruthJet_JetSimilarity_E_J"].append(j.e/1000)
                        BackUp["Two_TruthJet_JetSimilarity_R_TJ"].append(rtj)
                        BackUp["Two_TruthJet_JetSimilarity_R_J"].append(rj)
                        BackUp["Two_TruthJet_JetSimilarity_PT_TJ"].append(tj.pt/1000)
                        BackUp["Two_TruthJet_JetSimilarity_PT_J"].append(j.pt/1000)

                        BackUp["Two_TruthJet_JetsDeltaR"].append(dr)

            E_C.Events[i]["nominal"] = ""




    PickleObject(BackUp, "Plot" + CM_Energy + ".pkl")
    return True

def Test_SimilarityCustomOriginalMethods_Plot(Energy):

    E = "Plots/MatchingAlgorithm_tttt_" + Energy
    def Histograms_Template(Title, xTitle, yTitle, bins, Min, Max, Data, FileName, Color = None):
        H = TH1F()
        H.Title = Title
        H.xTitle = xTitle
        H.yTitle = yTitle
        H.xBins = bins
        H.xMin = Min
        H.xMax = Max
        H.xData = Data
        H.Alpha = 0.25
        
        if Color is not None:
            H.Color = Color
        H.Filename = FileName
        H.SaveFigure(E + "/Raw")
        return H

    def Histograms2D_Template(Title, xTitle, yTitle, xBins, yBins, xMin, xMax, yMin, yMax, xData, yData, FileName, Diagonal):
        H = TH2F()
        H.Diagonal = Diagonal
        H.Title = Title
        H.xTitle = xTitle
        H.yTitle = yTitle
        H.xBins = xBins 
        H.yBins = yBins 
        H.xMin = xMin
        H.xMax = xMax
        H.yMin = yMin
        H.yMax = yMax
        H.xData = xData
        H.yData = yData
        
        H.Filename = FileName
        H.SaveFigure(E)
        return H

    #BackUp = UnpickleObject("Plot.pkl")
    BackUp = UnpickleObject("Plot"+Energy+".pkl")

    # Random Statistics
    # --- Codes
    py = Histograms_Template("Status Codes", "Pythia Status", "Entries", 100, 0, 100, BackUp["Status_Code"], "TopStatusCodes.png")

    # Illustration of consistent tops 
    T = CombineHistograms()
    T.DefaultDPI = 500
    T.DefaultScaling = 7
    T.LabelSize = 15
    T.FontSize = 10
    T.LegendSize = 10
    T.Title = "Invariant Top Mass Derived From Different Algorithms"
    T.Log = False
    
    t1 = Histograms_Template("mTruth", "Mass (GeV)", "Entries", 250, 0, 200, BackUp["Tops_mTruth"], "mTruth.png")
    t2 = Histograms_Template("PreFSR (Alt-Algo)", "Mass (GeV)", "Entries", 250, 0, 200, BackUp["Tops_PreFSR"], "TopPreFSR.png")
    t3 = Histograms_Template("PostFSR (Alt-Algo)", "Mass (GeV)", "Entries", 250, 0, 200, BackUp["Tops_PostFSR"], "TopPostFSR.png", "black")

    T.Histograms = [t1, t2, t3]
    T.Filename = "TopMass_TruthTops.png"
    T.Save(E)

    # Illustration of consistent top mass based on truth children 
    T = CombineHistograms()
    T.DefaultDPI = 500
    T.DefaultScaling = 7
    T.LabelSize = 15
    T.FontSize = 10
    T.LegendSize = 10
    T.Title = "Invariant Mass of Top Derived from Children"
    T.Log = True

    tc = Histograms_Template("mTruth", "Mass (GeV)", "Entries", 250, 0, 200, BackUp["Children_mTruth"], "Top_Children_mTruth.png")
    tc_in = Histograms_Template("PostFSR (Alt-Algo)", "Mass (GeV)", "Entries", 250, 0, 200, BackUp["Children_PostFSR"], "Top_Children_PostFSR.png")

    T.Histograms = [tc, tc_in]
    T.Filename = "TopMass_Children.png"
    T.Save(E)
   

    # Overlaying the mass of top based on truth jet and children and Tops
    T = CombineHistograms()
    T.DefaultDPI = 500
    T.DefaultScaling = 7
    T.LabelSize = 15
    T.FontSize = 10
    T.LegendSize = 10
    T.Title = "Invariant Mass of Top from Decay Products (Including Leptonic Top)"
    T.Log = True

    t3.xBins = 500
    t3.Title = "Tops"
    t3.xMax = 500

    tc_in.xBins = 500
    tc_in.Title = "Children"
    tc_in.xMax = 500

    t4 = Histograms_Template("Truth Jets", "Mass (GeV)", "Entries", 500, 0, 500, BackUp["TruthJet_WithLep"], "TruthJet_WithLep.png")

    T.Histograms = [t4, tc_in, t3]
    T.Filename = "TopMass_TruthJet_WithLep.png"
    T.Save(E)

    # Overlaying the mass of top from all decay chain
    T = CombineHistograms()
    T.DefaultDPI = 500
    T.DefaultScaling = 7
    T.LabelSize = 15
    T.FontSize = 10
    T.LegendSize = 10
    T.Title = "Invariant Mass of Top from Decay Products (Without Leptonic Top)"
    T.Log = True

    t5 = Histograms_Template("Truth Jets", "Mass (GeV)", "Entries", 500, 0, 500, BackUp["TruthJet_NoLep"], "TruthJet_NoLep.png")

    T.Histograms = [t5, tc_in, t3]
    T.Filename = "TopMass_TruthJet_NoLep.png"
    T.Save(E)



    # Overlaying the mass of top from all decay chain
    T = CombineHistograms()
    T.DefaultDPI = 500
    T.DefaultScaling = 7
    T.LabelSize = 15
    T.FontSize = 10
    T.LegendSize = 10
    T.Title = "Invariant Mass of Top from Decay Products (Including Leptonic Top)"
    T.Log = True

    t6 = Histograms_Template("Jets", "Mass (GeV)", "Entries", 500, 0, 500, BackUp["Jets_Lep"], "Jet_WithLep.png")

    T.Histograms = [t6, t4, tc_in, t3]
    T.Filename = "TopMass_Jet_WithLep.png"
    T.Save(E)

    # Overlaying the mass of top from all decay chain
    T = CombineHistograms()
    T.DefaultDPI = 500
    T.DefaultScaling = 7
    T.LabelSize = 15
    T.FontSize = 10
    T.LegendSize = 10
    T.Title = "Invariant Mass of Top from Decay Products (Without Leptonic Top)"
    T.Log = True

    t7 = Histograms_Template("Jets", "Mass (GeV)", "Entries", 500, 0, 500, BackUp["Jets_NoLep"], "Jet_NoLep.png")

    T.Histograms = [t7, t5, tc_in, t3]
    T.Filename = "TopMass_Jet_NoLep.png"
    T.Save(E)


    # Splitting TRUTH JET top mass reconstruction according to how many tops contribute to a particular truth jet
    T = CombineHistograms()
    T.DefaultDPI = 500
    T.DefaultScaling = 7
    T.LabelSize = 15
    T.FontSize = 10
    T.LegendSize = 10
    T.Title = "Invariant Top Mass from Truth Jets - Overlapping Tops Contributing to Truth Jet \n (Without Leptonic Top - Included in Counting)"
    T.Log = True

    truj_t0 = Histograms_Template("0-Tops", "Mass (GeV)", "Entries", 500, 0, 500, BackUp["TopTruthJetShare_0"], "0Tops_Sharing_TruthJet.png")
    truj_t1 = Histograms_Template("1-Tops", "Mass (GeV)", "Entries", 500, 0, 500, BackUp["TopTruthJetShare_1"], "1Tops_Sharing_TruthJet.png")
    truj_t2 = Histograms_Template("2-Tops", "Mass (GeV)", "Entries", 500, 0, 500, BackUp["TopTruthJetShare_2"], "2Tops_Sharing_TruthJet.png")
    truj_t3 = Histograms_Template("3-Tops", "Mass (GeV)", "Entries", 500, 0, 500, BackUp["TopTruthJetShare_3"], "3Tops_Sharing_TruthJet.png")

    T.Histograms = [truj_t0, truj_t1, truj_t2, truj_t3]
    T.Filename = "TruthJetsSharedBetweenTops.png"
    T.Save(E)


    # Splitting JET top mass reconstruction according to how many tops contribute to a particular truth jet
    T = CombineHistograms()
    T.DefaultDPI = 500
    T.DefaultScaling = 7
    T.LabelSize = 15
    T.FontSize = 10
    T.LegendSize = 10
    T.Title = "Invariant Top Mass from Jets - Overlapping Tops Contributing to Jet \n (Without Leptonic Top - Included in Counting)"
    T.Log = True

    j_t0 = Histograms_Template("0-Tops", "Mass (GeV)", "Entries", 500, 0, 500, BackUp["TopJetShare_0"], "0Tops_Sharing_Jet.png")
    j_t1 = Histograms_Template("1-Tops", "Mass (GeV)", "Entries", 500, 0, 500, BackUp["TopJetShare_1"], "1Tops_Sharing_Jet.png")
    j_t2 = Histograms_Template("2-Tops", "Mass (GeV)", "Entries", 500, 0, 500, BackUp["TopJetShare_2"], "2Tops_Sharing_Jet.png")
    j_t3 = Histograms_Template("3-Tops", "Mass (GeV)", "Entries", 500, 0, 500, BackUp["TopJetShare_3"], "3Tops_Sharing_Jet.png")

    T.Histograms = [j_t0, j_t1, j_t2, j_t3]
    T.Filename = "JetsSharedBetweenTops.png"
    T.Save(E)



    # Splitting TRUTH JET top mass reconstruction based on number of n-jets contributing to the reconstruction
    T = CombineHistograms()
    T.DefaultDPI = 500
    T.DefaultScaling = 7
    T.LabelSize = 15
    T.FontSize = 10
    T.LegendSize = 10
    T.Title = "Invariant Top Mass from Truth Jets - Number of Jets \n Contributing to Reconstruction (Without Leptonic Top)"
    T.Log = True

    truj_t0 = Histograms_Template("1-Truth Jets", "Mass (GeV)", "Entries", 500, 0, 500, BackUp["Top_1TruthJets"], "1TruthJet.png")
    truj_t1 = Histograms_Template("2-Truth Jets", "Mass (GeV)", "Entries", 500, 0, 500, BackUp["Top_2TruthJets"], "2TruthJet.png")
    truj_t2 = Histograms_Template("3-Truth Jets", "Mass (GeV)", "Entries", 500, 0, 500, BackUp["Top_3TruthJets"], "3TruthJet.png")
    truj_t3 = Histograms_Template("4-Truth Jets", "Mass (GeV)", "Entries", 500, 0, 500, BackUp["Top_4TruthJets"], "4TruthJet.png")

    T.Histograms = [truj_t0, truj_t1, truj_t2, truj_t3]
    T.Filename = "N_TruthJetsTops.png"
    T.Save(E)


    # Splitting TRUTH JET top mass reconstruction based on number of n-jets contributing to the reconstruction
    T = CombineHistograms()
    T.DefaultDPI = 500
    T.DefaultScaling = 7
    T.LabelSize = 15
    T.FontSize = 10
    T.LegendSize = 10
    T.Title = "Invariant Top Mass from Jets - Number of Jets \n Contributing to Reconstruction (Without Leptonic Top)"
    T.Log = True

    truj_t0 = Histograms_Template("1-Jets", "Mass (GeV)", "Entries", 500, 0, 500, BackUp["Top_1Jets"], "1Jet.png")
    truj_t1 = Histograms_Template("2-Jets", "Mass (GeV)", "Entries", 500, 0, 500, BackUp["Top_2Jets"], "2Jet.png")
    truj_t2 = Histograms_Template("3-Jets", "Mass (GeV)", "Entries", 500, 0, 500, BackUp["Top_3Jets"], "3Jet.png")
    truj_t3 = Histograms_Template("4-Jets", "Mass (GeV)", "Entries", 500, 0, 500, BackUp["Top_4Jets"], "4Jet.png")

    T.Histograms = [truj_t0, truj_t1, truj_t2, truj_t3]
    T.Filename = "N_JetsTops.png"
    T.Save(E)


    # DeltaR between Truth Jet and Jet 
    T = CombineHistograms()
    T.DefaultDPI = 500
    T.DefaultScaling = 7
    T.LabelSize = 15
    T.FontSize = 10
    T.LegendSize = 10
    T.Title = r"$\Delta$ R Between Truth Jet and Jet - Single and Double Jet Matching"
    T.Log = True

    r0 = Histograms_Template("Single", r'$\Delta$ R (Arb.)', "Entries", 250, 0, 1, BackUp["Single_TruthJet_JetsDeltaR"], "1R_Jet.png")
    r1 = Histograms_Template("Two", r'$\Delta$ R (Arb.)', "Entries", 250, 0, 1, BackUp["Two_TruthJet_JetsDeltaR"], "2R_Jet.png")

    T.Histograms = [r0, r1]
    T.Filename = "dR_Jets_TruthJets.png"
    T.Save(E)



    # =========================== 2D Histograms ======================================= #
    
    # Single Truth Jet Matched to Jet
    Histograms2D_Template("Energy of Truth Jet vs Matched Jet - Singly Matched", 
            "Truth Jet Energy (GeV)", "Jet Energy (GeV)", 250, 250, 0, 300, 0, 300, 
            BackUp["Single_TruthJet_JetSimilarity_E_J"], BackUp["Single_TruthJet_JetSimilarity_E_TJ"], 
            "JetMatchingSimilarity_Energy_Single.png", True)

    Histograms2D_Template("R (not $\Delta$R) of Truth Jet vs Matched Jet - Singly Matched", 
            "Truth Jet R (Arb.)", "Jet R (Arb.)", 250, 250, 0, 3, 0, 3, 
            BackUp["Single_TruthJet_JetSimilarity_R_J"], BackUp["Single_TruthJet_JetSimilarity_R_TJ"], 
            "JetMatchingSimilarity_R_Single.png", True)

    Histograms2D_Template("$P_T$ of Truth Jet vs Matched Jet - Singly Matched", 
            "Truth Jet $P_T$ (GeV)", "Jet $P_T$ (GeV)", 250, 250, 0, 300, 0, 300, 
            BackUp["Single_TruthJet_JetSimilarity_PT_J"], BackUp["Single_TruthJet_JetSimilarity_PT_TJ"], 
            "JetMatchingSimilarity_PT_Single.png", True)

    # Two Truth Jet Matched to Jet
    Histograms2D_Template("Energy of Truth Jet vs Matched Jet - Two Matched", 
            "Truth Jet Energy (GeV)", "Jet Energy (GeV)", 250, 250, 0, 600, 0, 600, 
            BackUp["Two_TruthJet_JetSimilarity_E_J"], BackUp["Two_TruthJet_JetSimilarity_E_TJ"], 
            "JetMatchingSimilarity_Energy_Two.png", True)

    Histograms2D_Template("R (not $\Delta$R) of Truth Jet vs Matched Jet - Two Matched", 
            "Truth Jet R (Arb.)", "Jet R (Arb.)", 250, 250, 0, 3, 0, 3, 
            BackUp["Two_TruthJet_JetSimilarity_R_J"], BackUp["Two_TruthJet_JetSimilarity_R_TJ"], 
            "JetMatchingSimilarity_R_Two.png", True)

    Histograms2D_Template("$P_T$ of Truth Jet vs Matched Jet - Two Matched", 
            "Truth Jet $P_T$ (GeV)", "Jet $P_T$ (GeV)", 250, 250, 0, 600, 0, 600, 
            BackUp["Two_TruthJet_JetSimilarity_PT_J"], BackUp["Two_TruthJet_JetSimilarity_PT_TJ"], 
            "JetMatchingSimilarity_PT_Two.png", True)









    return True
    
