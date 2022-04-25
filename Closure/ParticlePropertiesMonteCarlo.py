from Functions.IO.Files import Directories
from Functions.IO.IO import PickleObject, UnpickleObject
from Functions.Plotting.Histograms import TH2F, TH1F, CombineHistograms


def Histograms_Template(Title, xTitle, yTitle, bins, Min, Max, Data, FileName, Dir, Color = None, Weight = None):
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
       

    Backup["TopsInJets_Tops_Energy"] = []
    Backup["TopsInJets_Tops_Eta"] = []
    Backup["TopsInJets_Tops_Phi"] = []
    Backup["TopsInJets_Tops_Pt"] = []
    
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
                else: 
                    for t in j.JetMapTops:

                        top = tops[t]
                        if len(j.JetMapTops) == 1:
                            Backup["TopsInJets_J_TN1_Energy"].append(top.e/1000)
                            Backup["TopsInJets_J_TN1_Eta"].append(top.eta)
                            Backup["TopsInJets_J_TN1_Phi"].append(top.phi)
                            Backup["TopsInJets_J_TN1_Pt"].append(top.pt/1000)

                            Backup["TopsInJets_J_N1_Energy"].append(j.e/1000)
                            Backup["TopsInJets_J_N1_Eta"].append(j.eta)               
                            Backup["TopsInJets_J_N1_Phi"].append(j.phi)
                            Backup["TopsInJets_J_N1_Pt"].append(j.pt/1000)
                   
                        if len(j.JetMapTops) == 2:
                            Backup["TopsInJets_J_TN2_Energy"].append(top.e/1000)
                            Backup["TopsInJets_J_TN2_Eta"].append(top.eta)
                            Backup["TopsInJets_J_TN2_Phi"].append(top.phi)
                            Backup["TopsInJets_J_TN2_Pt"].append(top.pt/1000)

                            Backup["TopsInJets_J_N2_Energy"].append(j.e/1000)
                            Backup["TopsInJets_J_N2_Eta"].append(j.eta)               
                            Backup["TopsInJets_J_N2_Phi"].append(j.phi)
                            Backup["TopsInJets_J_N2_Pt"].append(j.pt/1000)

                        if len(j.JetMapTops) == 3:
                            Backup["TopsInJets_J_TN3_Energy"].append(top.e/1000)
                            Backup["TopsInJets_J_TN3_Eta"].append(top.eta)
                            Backup["TopsInJets_J_TN3_Phi"].append(top.phi)
                            Backup["TopsInJets_J_TN3_Pt"].append(top.pt/1000)

                            Backup["TopsInJets_J_N3_Energy"].append(j.e/1000)
                            Backup["TopsInJets_J_N3_Eta"].append(j.eta)               
                            Backup["TopsInJets_J_N3_Phi"].append(j.phi)
                            Backup["TopsInJets_J_N3_Pt"].append(j.pt/1000)

   
    frac0 = len(Backup["NonTopJets_J_Energy"])/nj_sum
    frac1 = len(Backup["TopsInJets_J_N1_Energy"])/nj_sum
    frac2 = len(Backup["TopsInJets_J_N2_Energy"])/(2*nj_sum)
    frac3 = len(Backup["TopsInJets_J_N3_Energy"])/(3*nj_sum)

     
    if eta_cut != None or pt_cut != None and Plot == False:
        return [frac0,frac1,frac2,frac3]

    HT_1 = Histograms_Template("Fraction of Jets Containing n-Tops", "n-Tops", "Fraction of Jets", 4, 0, 4, [0,1,2,3], "n-Tops_Fractions", "ParticleProperties", Weight = [frac0,frac1,frac2,frac3]) 

    # Stack plot of Jet Energy as a function of number of merged tops
    OutputDir = "ParticleProperties/"
    if eta_cut != None:
        OutputDir += "_ETA_CUT_" + str(eta_cut)
    if pt_cut != None:
        OutputDir += "_PT_CUT_" + str(pt_cut)

    T = CombineHistograms()
    T.DefaultDPI = 500
    T.DefaultScaling = 7
    T.LabelSize = 15
    T.FontSize = 10
    T.LegendSize = 10
    T.Title = "Jet Energy as a function of number of Tops within Given Jet"
    HT_0 = Histograms_Template("0-Tops", "Energy (GeV)", "Sampled Jets", 100, 0, 500, Backup["NonTopJets_J_Energy"], "NonTopJetEnergy", OutputDir)  
    HT_1 = Histograms_Template("1-Tops", "Energy (GeV)", "Sampled Jets", 100, 0, 500, Backup["TopsInJets_J_N1_Energy"], "1TopsJetEnergy", OutputDir)  
    HT_2 = Histograms_Template("2-Tops", "Energy (GeV)", "Sampled Jets", 100, 0, 500, Backup["TopsInJets_J_N2_Energy"], "2TopsJetEnergy", OutputDir) 
    HT_3 = Histograms_Template("3-Tops", "Energy (GeV)", "Sampled Jets", 100, 0, 500, Backup["TopsInJets_J_N3_Energy"], "3TopsJetEnergy", OutputDir) 
    T.Histograms = [HT_0, HT_1, HT_2, HT_3]
    T.Filename = "JetEnergy_VS_N_tops"
    T.Save("Plots/" + OutputDir)

    T = CombineHistograms()
    T.DefaultDPI = 500
    T.DefaultScaling = 7
    T.LabelSize = 15
    T.FontSize = 10
    T.LegendSize = 10
    T.Title = "Jet PT as a function of number of Tops within Given Jet"
    HT_0 = Histograms_Template("0-Tops", "Transverse Momentum (GeV)", "Sampled Jets", 100, 0, 500, Backup["NonTopJets_J_Pt"], "NonTopJetPT", OutputDir)  
    HT_1 = Histograms_Template("1-Tops", "Transverse Momentum (GeV)", "Sampled Jets", 100, 0, 500, Backup["TopsInJets_J_N1_Pt"], "1TopsJetPT", OutputDir)  
    HT_2 = Histograms_Template("2-Tops", "Transverse Momentum (GeV)", "Sampled Jets", 100, 0, 500, Backup["TopsInJets_J_N2_Pt"], "2TopsJetPT", OutputDir) 
    HT_3 = Histograms_Template("3-Tops", "Transverse Momentum (GeV)", "Sampled Jets", 100, 0, 500, Backup["TopsInJets_J_N3_Pt"], "3TopsJetPT", OutputDir) 
    T.Histograms = [HT_0, HT_1, HT_2, HT_3]
    T.Filename = "JetPT_VS_N_tops"
    T.Save("Plots/" + OutputDir)

    # Stack plot of Tops in a given jet as a function of the Truth Top's Energy/PT
    T = CombineHistograms()
    T.DefaultDPI = 500
    T.DefaultScaling = 7
    T.LabelSize = 15
    T.FontSize = 10
    T.LegendSize = 10
    T.Title = "Stack Plot of n-Tops within a given Jet as a Function of the Truth Top's Energy"
    HT_1 = Histograms_Template("1-Tops", "Energy (GeV)", "Sampled Jets", 100, 150, 1500, Backup["TopsInJets_J_TN1_Energy"], "1Tops_TopEnergy", OutputDir)  
    HT_2 = Histograms_Template("2-Tops", "Energy (GeV)", "Sampled Jets", 100, 150, 1500, Backup["TopsInJets_J_TN2_Energy"], "2Tops_TopEnergy", OutputDir) 
    HT_3 = Histograms_Template("3-Tops", "Energy (GeV)", "Sampled Jets", 100, 150, 1500, Backup["TopsInJets_J_TN3_Energy"], "3Tops_TopEnergy", OutputDir) 
    T.Histograms = [HT_1, HT_2, HT_3]
    T.Filename = "TruthTopEnergy_VS_N_Tops_InJet"
    T.Save("Plots/" + OutputDir)

    T = CombineHistograms()
    T.DefaultDPI = 500
    T.DefaultScaling = 7
    T.LabelSize = 15
    T.FontSize = 10
    T.LegendSize = 10
    T.Title = "Stack Plot of n-Tops within a given Jet as a Function of the Truth Top's PT"
    HT_1 = Histograms_Template("1-Tops", "Transverse Momentum (GeV)", "Sampled Jets", 100, 150, 1500, Backup["TopsInJets_J_TN1_Pt"], "1Tops_TopPT", OutputDir)  
    HT_2 = Histograms_Template("2-Tops", "Transverse Momentum (GeV)", "Sampled Jets", 100, 150, 1500, Backup["TopsInJets_J_TN2_Pt"], "2Tops_TopPT", OutputDir) 
    HT_3 = Histograms_Template("3-Tops", "Transverse Momentum (GeV)", "Sampled Jets", 100, 150, 1500, Backup["TopsInJets_J_TN3_Pt"], "3Tops_TopPT", OutputDir) 
    T.Histograms = [HT_1, HT_2, HT_3]
    T.Filename = "TruthTopPT_VS_N_Tops_InJet"
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

def JetMergingFrequencyFractionPlot(Dir, Process):
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

