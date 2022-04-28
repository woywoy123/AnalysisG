from Functions.IO.Files import Directories
from Functions.IO.IO import PickleObject, UnpickleObject
from Functions.Plotting.Histograms import TH2F, TH1F, CombineHistograms


def Histograms_Template(Title, xTitle, yTitle, bins, Min, Max, Data, FileName = None, Dir = None, Color = None, Weight = None):
    H = TH1F()
    H.Title = Title
    H.xTitle = xTitle
    H.yTitle = yTitle
    H.xBins = bins
    H.xMin = Min
    H.xMax = Max
    H.xData = Data
    H.Alpha = 0.25
    H.Weights = Weight
    
    if Color is not None:
        H.Color = Color

    if FileName != None:
        H.Filename = FileName
        H.SaveFigure("Plots/" + Dir + "/Raw")
    return H

def Histograms2D_Template(Title, xTitle, yTitle, xBins, yBins, xMin, xMax, yMin, yMax, xData, yData, FileName, Dir, Diagonal = False, Weight = None):
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
    H.Weights = Weight
    H.ShowBinContent = True

    H.Filename = FileName
    H.SaveFigure("Plots/" + Dir)
    return H


def HistogramCombineTemplate(DPI = 500, Scaling = 7, Size = 10):
    T = CombineHistograms()
    T.DefaultDPI = DPI
    T.DefaultScaling = Scaling
    T.LabelSize = Size + 5
    T.FontSize = Size
    T.LegendSize = Size
    return T


def JetMergingFrequency(FileDir, eta_cut = None, pt_cut = None, Plot = True):
   
    Backup = {}

    Backup["NonTopJets_J_Energy"] = []
    Backup["NonTopJets_J_Eta"] = []
    Backup["NonTopJets_J_Phi"] = []
    Backup["NonTopJets_J_Pt"] = []

    Backup["TopsInJets_J_TN1_Energy"] = []
    Backup["TopsInJets_J_TN1_Eta"] = []
    Backup["TopsInJets_J_TN1_Phi"] = []
    Backup["TopsInJets_J_TN1_Pt"] = []

    Backup["TopsInJets_J_TN2_Energy"] = []
    Backup["TopsInJets_J_TN2_Eta"] = []
    Backup["TopsInJets_J_TN2_Phi"] = []
    Backup["TopsInJets_J_TN2_Pt"] = []

    Backup["TopsInJets_J_TN3_Energy"] = []
    Backup["TopsInJets_J_TN3_Eta"] = []
    Backup["TopsInJets_J_TN3_Phi"] = []
    Backup["TopsInJets_J_TN3_Pt"] = []
       
    Backup["TopsInJets_J_N1_Energy"] = []
    Backup["TopsInJets_J_N1_Eta"] = []
    Backup["TopsInJets_J_N1_Phi"] = []
    Backup["TopsInJets_J_N1_Pt"] = []

    Backup["TopsInJets_J_N2_Energy"] = []
    Backup["TopsInJets_J_N2_Eta"] = []
    Backup["TopsInJets_J_N2_Phi"] = []
    Backup["TopsInJets_J_N2_Pt"] = []

    Backup["TopsInJets_J_N3_Energy"] = []
    Backup["TopsInJets_J_N3_Eta"] = []
    Backup["TopsInJets_J_N3_Phi"] = []
    Backup["TopsInJets_J_N3_Pt"] = []

    Backup["TopsInJets_Energy_Fract1"] = []
    Backup["TopsInJets_Energy_Fract2"] = []
    Backup["TopsInJets_Energy_Fract3"] = []

    Backup["TopsInJets_PT_Fract1"] = []
    Backup["TopsInJets_PT_Fract2"] = []
    Backup["TopsInJets_PT_Fract3"] = []
    
    if isinstance(FileDir, str):
        d = Directories(FileDir).ListFilesInDir(FileDir)
    else: 
        d = FileDir

    nj_sum = 0
    for f in d:
        if isinstance(FileDir, str):
            F = UnpickleObject(FileDir + "/" + f)
        else: 
            F = f
        
        for ev in F.Events:
            event = F.Events[ev]["nominal"]
            
            jets = event.Jets
            tops = event.TopPostFSR
            for j in jets:
                if eta_cut != None and abs(j.eta) > eta_cut:
                    continue
                if pt_cut != None and abs(j.pt/1000) < pt_cut:
                    continue
                
                nj_sum += 1

                if -1 in j.JetMapTops:
                    Backup["NonTopJets_J_Energy"].append(j.e/1000)
                    Backup["NonTopJets_J_Eta"].append(j.eta)
                    Backup["NonTopJets_J_Phi"].append(j.phi)
                    Backup["NonTopJets_J_Pt"].append(j.pt/1000)
                    continue

                if len(j.JetMapTops) > 3:
                    continue

                for t in j.JetMapTops:
                    top = tops[t]
                    Backup["TopsInJets_J_TN" + str(len(j.JetMapTops)) + "_Energy"].append(top.e/1000)
                    Backup["TopsInJets_J_TN" + str(len(j.JetMapTops)) + "_Eta"].append(top.eta)
                    Backup["TopsInJets_J_TN" + str(len(j.JetMapTops)) + "_Phi"].append(top.phi)
                    Backup["TopsInJets_J_TN" + str(len(j.JetMapTops)) + "_Pt"].append(top.pt/1000)

                    Backup["TopsInJets_J_N" + str(len(j.JetMapTops)) + "_Energy"].append(j.e/1000)
                    Backup["TopsInJets_J_N" + str(len(j.JetMapTops)) + "_Eta"].append(j.eta)               
                    Backup["TopsInJets_J_N" + str(len(j.JetMapTops)) + "_Phi"].append(j.phi)
                    Backup["TopsInJets_J_N" + str(len(j.JetMapTops)) + "_Pt"].append(j.pt/1000)
                
                    Backup["TopsInJets_Energy_Fract" + str(len(j.JetMapTops))].append(j.e/top.e)
                    Backup["TopsInJets_PT_Fract" + str(len(j.JetMapTops))].append(j.pt/top.pt)
 
    frac0 = len(Backup["NonTopJets_J_Energy"])/nj_sum
    frac1 = len(Backup["TopsInJets_J_N1_Energy"])/nj_sum
    frac2 = len(Backup["TopsInJets_J_N2_Energy"])/(2*nj_sum)
    frac3 = len(Backup["TopsInJets_J_N3_Energy"])/(3*nj_sum)

     
    if eta_cut != None or pt_cut != None and Plot == False:
        return [frac0,frac1,frac2,frac3]

    HT_1 = Histograms_Template("Fraction of Jets Containing n-Tops", "n-Tops", "Fraction of Jets", 4, 0, 4, [0,1,2,3], "n-Tops_Fractions", "ParticleProperties", Weight = [frac0,frac1,frac2,frac3]) 

    OutputDir = "ParticleProperties/"
    if eta_cut != None:
        OutputDir += "_ETA_CUT_" + str(eta_cut)
    if pt_cut != None:
        OutputDir += "_PT_CUT_" + str(pt_cut)

    T = HistogramCombineTemplate()
    T.Title = "Jet Energy as a function of number of Tops within Given Jet"
    HT_0 = Histograms_Template("0-Tops", "Energy (GeV)", "Sampled Jets", 100, 0, 500, Backup["NonTopJets_J_Energy"], "NonTopJetEnergy", OutputDir)  
    HT_1 = Histograms_Template("1-Tops", "Energy (GeV)", "Sampled Jets", 100, 0, 500, Backup["TopsInJets_J_N1_Energy"], "1TopsJetEnergy", OutputDir)  
    HT_2 = Histograms_Template("2-Tops", "Energy (GeV)", "Sampled Jets", 100, 0, 500, Backup["TopsInJets_J_N2_Energy"], "2TopsJetEnergy", OutputDir) 
    HT_3 = Histograms_Template("3-Tops", "Energy (GeV)", "Sampled Jets", 100, 0, 500, Backup["TopsInJets_J_N3_Energy"], "3TopsJetEnergy", OutputDir) 
    T.Histograms = [HT_0, HT_1, HT_2, HT_3]
    T.Filename = "JetEnergy_VS_N_tops"
    T.Save("Plots/" + OutputDir)

    T = HistogramCombineTemplate()    
    T.Title = "Jet PT as a function of number of Tops within Given Jet"
    HT_0 = Histograms_Template("0-Tops", "Transverse Momentum (GeV)", "Sampled Jets", 100, 0, 500, Backup["NonTopJets_J_Pt"], "NonTopJetPT", OutputDir)  
    HT_1 = Histograms_Template("1-Tops", "Transverse Momentum (GeV)", "Sampled Jets", 100, 0, 500, Backup["TopsInJets_J_N1_Pt"], "1TopsJetPT", OutputDir)  
    HT_2 = Histograms_Template("2-Tops", "Transverse Momentum (GeV)", "Sampled Jets", 100, 0, 500, Backup["TopsInJets_J_N2_Pt"], "2TopsJetPT", OutputDir) 
    HT_3 = Histograms_Template("3-Tops", "Transverse Momentum (GeV)", "Sampled Jets", 100, 0, 500, Backup["TopsInJets_J_N3_Pt"], "3TopsJetPT", OutputDir) 
    T.Histograms = [HT_0, HT_1, HT_2, HT_3]
    T.Filename = "JetPT_VS_N_tops"
    T.Save("Plots/" + OutputDir)

    # Stack plot of Tops in a given jet as a function of the Truth Top's Energy/PT
    T = HistogramCombineTemplate()  
    T.Title = "Stack Plot of n-Tops within a given Jet as a Function of the Truth Top's Energy"
    HT_1 = Histograms_Template("1-Tops", "Energy (GeV)", "Sampled Jets", 100, 150, 1500, Backup["TopsInJets_J_TN1_Energy"], "1Tops_TopEnergy", OutputDir)  
    HT_2 = Histograms_Template("2-Tops", "Energy (GeV)", "Sampled Jets", 100, 150, 1500, Backup["TopsInJets_J_TN2_Energy"], "2Tops_TopEnergy", OutputDir) 
    HT_3 = Histograms_Template("3-Tops", "Energy (GeV)", "Sampled Jets", 100, 150, 1500, Backup["TopsInJets_J_TN3_Energy"], "3Tops_TopEnergy", OutputDir) 
    T.Histograms = [HT_1, HT_2, HT_3]
    T.Filename = "TruthTopEnergy_VS_N_Tops_InJet"
    T.Save("Plots/" + OutputDir)

    T = HistogramCombineTemplate()  
    T.Title = "Stack Plot of n-Tops within a given Jet as a Function of the Truth Top's PT"
    HT_1 = Histograms_Template("1-Tops", "Transverse Momentum (GeV)", "Sampled Jets", 100, 150, 1500, Backup["TopsInJets_J_TN1_Pt"], "1Tops_TopPT", OutputDir)  
    HT_2 = Histograms_Template("2-Tops", "Transverse Momentum (GeV)", "Sampled Jets", 100, 150, 1500, Backup["TopsInJets_J_TN2_Pt"], "2Tops_TopPT", OutputDir) 
    HT_3 = Histograms_Template("3-Tops", "Transverse Momentum (GeV)", "Sampled Jets", 100, 150, 1500, Backup["TopsInJets_J_TN3_Pt"], "3Tops_TopPT", OutputDir) 
    T.Histograms = [HT_1, HT_2, HT_3]
    T.Filename = "TruthTopPT_VS_N_Tops_InJet"
    T.Save("Plots/" + OutputDir)

    T = HistogramCombineTemplate()
    T.Title = "Energy Fraction of Truth-Top Energy within Sampled Jet"
    HT_1 = Histograms_Template("1-Tops", "(Jet)/(Truth-Top) Energy", "Entries", 100, 0, 1.25, Backup["TopsInJets_Energy_Fract1"], "Tops-1_Fract_Energy", OutputDir)  
    HT_2 = Histograms_Template("2-Tops", "(Jet)/(Truth-Top) Energy", "Entries", 100, 0, 1.25, Backup["TopsInJets_Energy_Fract2"], "Tops-2_Fract_Energy", OutputDir) 
    HT_3 = Histograms_Template("3-Tops", "(Jet)/(Truth-Top) Energy", "Entries", 100, 0, 1.25, Backup["TopsInJets_Energy_Fract3"], "Tops-3_Fract_Energy", OutputDir) 
    T.Histograms = [HT_1, HT_2, HT_3]
    T.Filename = "FractionalEnergyContributionsInJets.png"
    T.Save("Plots/" + OutputDir)

    T = HistogramCombineTemplate()
    T.Title = "Energy Fraction of Truth-Top PT within Sampled Jet"
    HT_1 = Histograms_Template("1-Tops", "(Jet)/(Truth-Top) PT", "Entries", 100, 0, 1.25, Backup["TopsInJets_PT_Fract1"], "Tops-1_Fract_PT", OutputDir)  
    HT_2 = Histograms_Template("2-Tops", "(Jet)/(Truth-Top) PT", "Entries", 100, 0, 1.25, Backup["TopsInJets_PT_Fract2"], "Tops-2_Fract_PT", OutputDir) 
    HT_3 = Histograms_Template("3-Tops", "(Jet)/(Truth-Top) PT", "Entries", 100, 0, 1.25, Backup["TopsInJets_PT_Fract3"], "Tops-3_Fract_PT", OutputDir) 
    T.Histograms = [HT_1, HT_2, HT_3]
    T.Filename = "FractionalEnergyContributionsInJets.png"
    T.Save("Plots/" + OutputDir)

    return True

def JetMergingFrequencyFraction(FileDir):

    start_eta = 3
    end_eta = 1
    n_eta = 100
    delta_eta = (abs(end_eta - start_eta)/n_eta)
    Scan_eta = [round(start_eta - delta_eta*j, 3) for j in range(n_eta+1)] 


    start_pt = 10
    end_pt = 100
    n_pt = 9
    delta_pt = (abs(end_pt - start_pt)/n_pt)
    Scan_pt = [start_pt + delta_pt*j for j in range(n_pt+1)] 
    

    d = Directories(FileDir).ListFilesInDir(FileDir)
   
    Loop = [] 
    for i in d:
        Loop.append(UnpickleObject(FileDir + "/" + i))
   
    Backup = {}
    for pt in Scan_pt:
        for eta in Scan_eta:
            print("Scanning ----> "+ str(pt) + " @ " + str(eta))
            Backup[str(pt) + "_" + str(eta)] = JetMergingFrequency(Loop, eta, pt)

        print("Checkpoint ----> "+ str(pt))
        PickleObject(Backup, "Fraction_of_Jets_Tops_" + str(pt) + ".pkl", "_Pickles")
    
    
    return True

def JetMergingFrequencyFractionPlot(Dir):
    F = UnpickleObject(Dir)
    
    PT = []
    ETA = []
    for i in F:
        PT.append(float(i.split("_")[0]))
        ETA.append(float(i.split("_")[1]))
    PT = list(set(PT))
    ETA = list(set(ETA))
    PT.sort()
    ETA.sort()
    
    Weights_0 = []
    Weights_1 = []
    Weights_2 = []
    Weights_3 = []
    for pt in PT:
        W_i_0 = []
        W_i_1 = []
        W_i_2 = []
        W_i_3 = []
        for eta in ETA:
            d = F[str(pt)+ "_" + str(eta)]
            W_i_0.append(d[0])
            W_i_1.append(d[1])
            W_i_2.append(d[2])
            W_i_3.append(d[3])
        Weights_0.append(W_i_0)
        Weights_1.append(W_i_1)
        Weights_2.append(W_i_2)
        Weights_3.append(W_i_3)

    Histograms2D_Template("Fraction of Jets Without Tops After Applying PT and ETA Cuts", "Lowest PT Threshold (GeV)", "Highest Eta Threshold", 
            None, None, min(PT), max(PT), min(ETA), max(ETA), 
            PT, ETA, "Fractions_0-Tops", "JetMergingFrequency/", Weight = Weights_0)

    Histograms2D_Template("Fraction of Jets With 1-Top After Applying PT and ETA Cuts", "Lowest PT Threshold (GeV)", "Highest Eta Threshold", 
            None, None, min(PT), max(PT), min(ETA), max(ETA), 
            PT, ETA, "Fractions_1-Tops", "JetMergingFrequency/", Weight = Weights_1)
    
    Histograms2D_Template("Fraction of Jets With 2-Tops After Applying PT and ETA Cuts", "Lowest PT Threshold (GeV)", "Highest Eta Threshold", 
            None, None, min(PT), max(PT), min(ETA), max(ETA), 
            PT, ETA, "Fractions_2-Tops", "JetMergingFrequency/", Weight = Weights_2)

    Histograms2D_Template("Fraction of Jets With 3-Tops After Applying PT and ETA Cuts", "Lowest PT Threshold (GeV)", "Highest Eta Threshold", 
            None, None, min(PT), max(PT), min(ETA), max(ETA), 
            PT, ETA, "Fractions_3-Tops", "JetMergingFrequency/", Weight = Weights_3)

    return True

def FragmentationOfTriplets(FileDir, eta_cut = None, pt_cut = None, Plot = True):
 
    def Cut(jet):
        if eta_cut != None and abs(j.eta) > eta_cut:
            return True
        if pt_cut != None and abs(j.pt/1000) < pt_cut:
            return True 
        return False

    Backup = {}
    Backup["N"] = 0
    Backup["N_broken"] = 0
    
    Backup["N_signal"] = 0
    Backup["N_broken_signal"] = 0

    Backup["N_spect"] = 0
    Backup["N_broken_spect"] = 0

    Backup["N_jets_Per_top"] = []
    Backup["N_jets_Per_top_Signal"] = []
    Backup["N_jets_Per_top_Spect"] = []

    if isinstance(FileDir, str):
        d = Directories(FileDir).ListFilesInDir(FileDir)
    else: 
        d = FileDir

    for f in d: 
        if isinstance(FileDir, str):
            F = UnpickleObject(FileDir + "/" + f)
        else: 
            F = f

        for e in F.Events:
            event = F.Events[e]["nominal"]

            jets = event.Jets
            tops = event.TopPostFSR
            
            Backup["N"] += len(tops)
            for t in tops:
                t.Decay = []

            for t in jets:
                if -1 in t.JetMapTops:
                    continue
                
                for t_index in t.JetMapTops:
                    tops[t_index].Decay.append(t)

            for t in tops:
                for j in t.Decay:
                    if Cut(j):
                        break
                if Cut(j):
                    Backup["N_broken"] += 1

                if Cut(j) and t.FromRes == 1:
                    Backup["N_broken_signal"] += 1
               
                if Cut(j) and t.FromRes == 0:
                    Backup["N_broken_spect"] += 1

                Backup["N_jets_Per_top"].append(len(t.Decay))
                if t.FromRes == 1:
                    Backup["N_signal"] += 1
                    Backup["N_jets_Per_top_Signal"].append(len(t.Decay))
                else:
                    Backup["N_spect"] += 1
                    Backup["N_jets_Per_top_Spect"].append(len(t.Decay))
    
    eff_all = float(Backup["N_broken"]) / float(Backup["N"])
    eff_signal = float(Backup["N_broken_signal"]) / float(Backup["N_signal"])
    eff_spect = float(Backup["N_broken_spect"]) / float(Backup["N_spect"])

    if eta_cut != None and pt_cut != None:
        return [eff_all, eff_signal, eff_spect]

    OutputDir = "Plots/FragmentationOfTriplets"
    T = HistogramCombineTemplate()
    T.Title = "Number of Jets Associated with Tops"
    H1 = Histograms_Template("All-Tops", "N-Jets", "Sampled Tops", max(Backup["N_jets_Per_top"]), 0, max(Backup["N_jets_Per_top"]), Backup["N_jets_Per_top"], "N-Jets_tops_all", OutputDir)
    H2 = Histograms_Template("All-Signal", "N-Jets", "Sampled Tops", max(Backup["N_jets_Per_top"]), 0, max(Backup["N_jets_Per_top"]), Backup["N_jets_Per_top_Signal"], "N-Jets_tops_Signal", OutputDir)
    H3 = Histograms_Template("All-Spectator", "N-Jets", "Sampled Tops", max(Backup["N_jets_Per_top"]), 0, max(Backup["N_jets_Per_top"]), Backup["N_jets_Per_top_Spect"], "N-Jets_tops_Spect", OutputDir)
    T.Histograms = [H1, H2, H3]
    T.Filename = "N-jets_tops_Combined"
    T.Save(OutputDir)

    print("All Top Triplets Broken due to Cut ", eff_all)
    print("Signal Top Triplets Broken due to Cut", eff_signal)
    print("Spectator Top Triplets Broken due to Cut", eff_spect)


    return True

def FragmentationOfTripletsScanning(FileDir, Pickle = None):

    if Pickle == None:
        start_eta = 3
        end_eta = 1
        n_eta = 100
        delta_eta = (abs(end_eta - start_eta)/n_eta)
        Scan_eta = [round(start_eta - delta_eta*j, 3) for j in range(n_eta+1)] 


        start_pt = 10
        end_pt = 100
        n_pt = 9
        delta_pt = (abs(end_pt - start_pt)/n_pt)
        Scan_pt = [start_pt + delta_pt*j for j in range(n_pt+1)] 
        
        d = Directories(FileDir).ListFilesInDir(FileDir)
   
        Loop = [] 
        for i in d:
            Loop.append(UnpickleObject(FileDir + "/" + i))
   
        Backup = {}
        for pt in Scan_pt:
            for eta in Scan_eta:
                print("Scanning ----> "+ str(pt) + " @ " + str(eta))
                Backup[str(pt) + "_" + str(eta)] = FragmentationOfTriplets(Loop, eta, pt)

            print("Checkpoint ----> "+ str(pt))
            PickleObject(Backup, "Fragmentation_Triplets" + str(pt) + ".pkl", "_Pickles")
            Pickle = "_Pickles/Fragmentation_Triplets" + str(pt) + ".pkl"

    F = UnpickleObject(Pickle)
    
    PT = []
    ETA = []
    for i in F:
        PT.append(float(i.split("_")[0]))
        ETA.append(float(i.split("_")[1]))
    PT = list(set(PT))
    ETA = list(set(ETA))
    PT.sort()
    ETA.sort()
    
    eff_all = []
    eff_signal = []
    eff_spect = []
    for pt in PT:
        W_i_0 = []
        W_i_1 = []
        W_i_2 = []
        for eta in ETA:
            d = F[str(pt)+ "_" + str(eta)]
            W_i_0.append(d[0])
            W_i_1.append(d[1])
            W_i_2.append(d[2])
            
        eff_all.append(W_i_0)
        eff_signal.append(W_i_1)
        eff_spect.append(W_i_2)

    Histograms2D_Template("Fraction of Tops Affected by Cut (All - Signal + Spectator)", "Lowest PT Threshold (GeV)", "Highest Eta Threshold", 
            None, None, min(PT), max(PT), min(ETA), max(ETA), 
            PT, ETA, "All-Tops", "FragmentationOfTriplets/", Weight = eff_all)

    Histograms2D_Template("Fraction of Tops Affected by Cut (Resonance Tops)", "Lowest PT Threshold (GeV)", "Highest Eta Threshold", 
            None, None, min(PT), max(PT), min(ETA), max(ETA), 
            PT, ETA, "Signal-Tops", "FragmentationOfTriplets/", Weight = eff_signal)
    
    Histograms2D_Template("Fraction of Tops Affected by Cut (Spectator Tops)", "Lowest PT Threshold (GeV)", "Highest Eta Threshold", 
            None, None, min(PT), max(PT), min(ETA), max(ETA), 
            PT, ETA, "Spectator-Tops", "FragmentationOfTriplets/", Weight = eff_spect)


    return True


def TopJetTrajectory(FileDir):

    d = Directories(FileDir).ListFilesInDir(FileDir)
    Backup = {}
    Backup["DeltaR_TopJet_All"] = []
    Backup["DeltaR_TopJet_Signal"] = []
    Backup["DeltaR_TopJet_Spectator"] = []
    Backup["DeltaR_TopJet_Leptonic_Signal"] = []
    Backup["DeltaR_TopJet_Leptonic_Spect"] = []

    Backup["DeltaR_JetsOfSameTop_All"] = []
    Backup["DeltaR_JetsOfSameTop_Signal"] = []
    Backup["DeltaR_JetsOfSameTop_Spectator"] = []

    Backup["DeltaR_TopJet_Top_1"] = []
    Backup["DeltaR_TopJet_Top_2"] = []
    Backup["DeltaR_TopJet_Top_3"] = []
    Backup["DeltaR_TopJet_Top_4"] = []
    Backup["DeltaR_TopJet_Top_5"] = []
    Backup["DeltaR_TopJet_Top_21"] = []

    Backup["DeltaR_TopTruthJet_Top_1"] = []
    Backup["DeltaR_TopTruthJet_Top_2"] = []
    Backup["DeltaR_TopTruthJet_Top_3"] = []
    Backup["DeltaR_TopTruthJet_Top_4"] = []
    Backup["DeltaR_TopTruthJet_Top_5"] = []
    Backup["DeltaR_TopTruthJet_Top_21"] = []

    for f in d:
        F = UnpickleObject(FileDir + "/" + f)

        for ev in F.Events:
            event = F.Events[ev]["nominal"]

            jets = event.Jets
            truthjets = event.TruthJets
            tops = event.TopPostFSR
            
            lepto = []
            for t in tops:
                t.Decay = []
                if len(list(set([11, 13, 15]) & set([abs(p.pdgid) for p in t.Decay_init]))) > 0:
                    lepto.append(t.Index)
                t.Decay_init = []

            for j in jets:
                if -1 in j.JetMapTops:
                    continue

                for t in j.JetMapTops:
                    tops[t].Decay.append(j)

            for j in truthjets:
                if -1 in j.GhostTruthJetMap:
                    continue

                for t in j.GhostTruthJetMap:
                    tops[t].Decay_init.append(j)
                


            for t in tops:
                for j in t.Decay_init:
                    if t.FromRes == 1:
                        
                        if abs(j.pdgid) == 1:
                            Backup["DeltaR_TopTruthJet_Top_1"].append(j.DeltaR(t))

                        if abs(j.pdgid) == 2:
                            Backup["DeltaR_TopTruthJet_Top_2"].append(j.DeltaR(t))
                        
                        if abs(j.pdgid) == 3:
                            Backup["DeltaR_TopTruthJet_Top_3"].append(j.DeltaR(t))

                        if abs(j.pdgid) == 4:
                            Backup["DeltaR_TopTruthJet_Top_4"].append(j.DeltaR(t))

                        if abs(j.pdgid) == 5:
                            Backup["DeltaR_TopTruthJet_Top_5"].append(j.DeltaR(t))

                        if abs(j.pdgid) == 21:
                            Backup["DeltaR_TopTruthJet_Top_21"].append(j.DeltaR(t))


                for j in t.Decay:
                    if t.FromRes == 1:
                        Backup["DeltaR_TopJet_Signal"].append(j.DeltaR(t))

                        if len(j.JetMapGhost) != 1 or -1 in j.JetMapGhost:
                            continue
                        
                        if j.truthPartonLabel == 1:
                            Backup["DeltaR_TopJet_Top_1"].append(j.DeltaR(t))

                        if j.truthPartonLabel == 2:
                            Backup["DeltaR_TopJet_Top_2"].append(j.DeltaR(t))
                        
                        if j.truthPartonLabel == 3:
                            Backup["DeltaR_TopJet_Top_3"].append(j.DeltaR(t))

                        if j.truthPartonLabel == 4:
                            Backup["DeltaR_TopJet_Top_4"].append(j.DeltaR(t))

                        if j.truthPartonLabel == 5:
                            Backup["DeltaR_TopJet_Top_5"].append(j.DeltaR(t))

                        if j.truthPartonLabel == 21:
                            Backup["DeltaR_TopJet_Top_21"].append(j.DeltaR(t))

                    
                    if t.FromRes == 0:
                        Backup["DeltaR_TopJet_Spectator"].append(j.DeltaR(t))
                    
                    if t.Index in lepto and t.FromRes == 1:
                        Backup["DeltaR_TopJet_Leptonic_Signal"].append(j.DeltaR(t))

                    if t.Index in lepto and t.FromRes == 0:
                        Backup["DeltaR_TopJet_Leptonic_Spect"].append(j.DeltaR(t))

                    Backup["DeltaR_TopJet_All"].append(j.DeltaR(t))

                for j in t.Decay:
                    for k in t.Decay:
                        if j == k:
                            continue
                        Backup["DeltaR_JetsOfSameTop_All"].append(j.DeltaR(k))

                        if t.FromRes == 1:
                            Backup["DeltaR_JetsOfSameTop_Signal"].append(j.DeltaR(k))

                        if t.FromRes == 0:
                            Backup["DeltaR_JetsOfSameTop_Spectator"].append(j.DeltaR(k))


    OutputDir = "TopJetTrajectory"
    T = HistogramCombineTemplate()    
    T.Title = "DeltaR and PDGID of Jet"
    HT_3 = Histograms_Template("Parton PDGID-1", "DeltaR", "Sampled Jets" , 100, 0, 3, Backup["DeltaR_TopJet_Top_1"]) 
    HT_4 = Histograms_Template("Parton PDGID-2", "DeltaR", "Sampled Jets" , 100, 0, 3, Backup["DeltaR_TopJet_Top_2"])
    HT_5 = Histograms_Template("Parton PDGID-3", "DeltaR", "Sampled Jets" , 100, 0, 3, Backup["DeltaR_TopJet_Top_3"])
    HT_6 = Histograms_Template("Parton PDGID-4", "DeltaR", "Sampled Jets" , 100, 0, 3, Backup["DeltaR_TopJet_Top_4"])
    HT_7 = Histograms_Template("Parton PDGID-5", "DeltaR", "Sampled Jets" , 100, 0, 3, Backup["DeltaR_TopJet_Top_5"])
    HT_8 = Histograms_Template("Parton PDGID-21", "DeltaR", "Sampled Jets" , 100, 0, 3, Backup["DeltaR_TopJet_Top_21"])

    T.Histograms = [HT_3, HT_4, HT_5, HT_6, HT_7, HT_8]
    T.Filename = "dR_debug"
    T.Save("Plots/" + OutputDir)

    OutputDir = "TopJetTrajectory"
    T = HistogramCombineTemplate()    
    T.Title = "DeltaR and PDGID of Truth Jet"
    HT_3 = Histograms_Template("Parton PDGID-1", "DeltaR", "Sampled Jets" , 100, 0, 3,  Backup["DeltaR_TopTruthJet_Top_1"]) 
    HT_4 = Histograms_Template("Parton PDGID-2", "DeltaR", "Sampled Jets" , 100, 0, 3,  Backup["DeltaR_TopTruthJet_Top_2"])
    HT_5 = Histograms_Template("Parton PDGID-3", "DeltaR", "Sampled Jets" , 100, 0, 3,  Backup["DeltaR_TopTruthJet_Top_3"])
    HT_6 = Histograms_Template("Parton PDGID-4", "DeltaR", "Sampled Jets" , 100, 0, 3,  Backup["DeltaR_TopTruthJet_Top_4"])
    HT_7 = Histograms_Template("Parton PDGID-5", "DeltaR", "Sampled Jets" , 100, 0, 3,  Backup["DeltaR_TopTruthJet_Top_5"])
    HT_8 = Histograms_Template("Parton PDGID-21", "DeltaR", "Sampled Jets" , 100, 0, 3, Backup["DeltaR_TopTruthJet_Top_21"])

    T.Histograms = [HT_3, HT_4, HT_5, HT_6, HT_7, HT_8]
    T.Filename = "dR_debug_truthjet"
    T.Save("Plots/" + OutputDir)


    OutputDir = "TopJetTrajectory"
    T = HistogramCombineTemplate()    
    T.Title = "DeltaR Between Top and Jets"
    HT_0 = Histograms_Template("All", "DeltaR", "Sampled Jets", 100, 0, 3, Backup["DeltaR_TopJet_All"], "dR_Jet_Top_All", OutputDir)  
    HT_2 = Histograms_Template("Signal", "DeltaR", "Sampled Jets", 100, 0, 3, Backup["DeltaR_TopJet_Signal"], "dR_Jet_Top_Signal", OutputDir) 
    HT_3 = Histograms_Template("Spectator", "DeltaR", "Sampled Jets", 100, 0, 3, Backup["DeltaR_TopJet_Spectator"], "dR_Jet_Top_Spectator", OutputDir) 
    HT_4 = Histograms_Template("Leptonic-Signal", "DeltaR", "Sampled Jets", 100, 0, 3, Backup["DeltaR_TopJet_Leptonic_Signal"], "dR_Jet_Top_Leptonic_Signal", OutputDir)
    HT_5 = Histograms_Template("Leptonic-Spect", "DeltaR", "Sampled Jets", 100, 0, 3, Backup["DeltaR_TopJet_Leptonic_Spect"], "dR_Jet_Top_Leptonic_Spect", OutputDir)
    T.Histograms = [HT_2, HT_3, HT_4, HT_5, HT_0]
    T.Filename = "dR_JetTop"
    T.Save("Plots/" + OutputDir)

    OutputDir = "TopJetTrajectory"
    T = HistogramCombineTemplate()    
    T.Title = "DeltaR Between Jets with Common Top"
    HT_0 = Histograms_Template("All", "DeltaR", "Sampled Jets", 100, 0, 3, Backup["DeltaR_JetsOfSameTop_All"], "dR_Jet_JetAll", OutputDir)  
    HT_2 = Histograms_Template("Signal", "DeltaR", "Sampled Jets", 100, 0, 3, Backup["DeltaR_JetsOfSameTop_Signal"], "dR_Jet_JetSignal", OutputDir) 
    HT_3 = Histograms_Template("Spectator", "DeltaR", "Sampled Jets", 100, 0, 3, Backup["DeltaR_JetsOfSameTop_Spectator"], "dR_Jet_JetSpectator", OutputDir) 
    T.Histograms = [HT_2, HT_3, HT_0]
    T.Filename = "dR_JetJet"
    T.Save("Plots/" + OutputDir)



    return True

