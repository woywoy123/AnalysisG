# Produce the plotting of the events in the analysis (basically a sanity check) 
from Functions.Event.Event import EventGenerator
from Functions.Particles.Particles import Particle
from Functions.IO.IO import UnpickleObject, PickleObject
from Functions.Plotting.Histograms import TH2F, TH1F, SubfigureCanvas, CombineHistograms

def TestTops():
   
    x = UnpickleObject("EventGenerator")
    x = x.Events 

    # Top mass containers 
    Top_Mass = []
    Top_Mass_From_Children = []
    Top_Mass_From_Children_init = []
    Top_Mass_From_Truth_Jets = []
    Top_Mass_From_Truth_Jets_init = []
    Top_Mass_From_Truth_Jets_NoAnomaly = []
    Top_Mass_From_Jets = []
    for i in x:
        ev = x[i]["nominal"]
        
        tops = ev.TruthTops
        for k in tops:
            k.CalculateMass()
            Top_Mass.append(k.Mass_GeV)
            
            k.CalculateMassFromChildren()
            Top_Mass_From_Children.append(k.Mass_GeV)
            Top_Mass_From_Children_init.append(k.Mass_init_GeV)
    
        for k in tops:

            # Calculation of top mass from truth jets + detector leptons 
            tmp = []
            for j in k.Decay:
                tmp += j.Decay
            k.Decay = tmp
            tmp = []
            for j in k.Decay_init:
                tmp += j.Decay_init
            k.Decay_init = tmp
            
            k.CalculateMassFromChildren()
            Top_Mass_From_Truth_Jets.append(k.Mass_GeV)
            Top_Mass_From_Truth_Jets_init.append(k.Mass_init_GeV)
        
            if ev.Anomaly_TruthMatch == False:
                Top_Mass_From_Truth_Jets_NoAnomaly.append(k.Mass_GeV)

            # Now we calculate from detector jets
            tmp = []
            for j in k.Decay:
                if j.Type == "truthjet":
                    tmp += j.Decay
                else:
                    tmp.append(j)
            k.Decay = tmp
            k.CalculateMassFromChildren()
            Top_Mass_From_Jets.append(k.Mass_GeV)

    # Tops from Truth information figures 
    s = SubfigureCanvas()
    s.Filename = "TopMasses"

    t = TH1F() 
    t.Title = "The Mass of Truth Top From 'truth_top_*'"
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.xBins = 200
    t.xData = Top_Mass
    t.CompileHistogram()
    s.AddObject(t)

    tc = TH1F()
    tc.Title = "The Predicted Truth Top Mass Derived From 'truth_top_child_*'"
    tc.xTitle = "Mass (GeV)"
    tc.yTitle = "Entries"
    tc.xBins = 200
    tc.xMin = 160
    tc.xData = Top_Mass_From_Children
    tc.CompileHistogram()
    s.AddObject(tc)   

    tc_init = TH1F()
    tc_init.Title = "The Predicted Truth Top Mass Derived From 'truth_top_child_init_*'"
    tc_init.xTitle = "Mass (GeV)"
    tc_init.yTitle = "Entries"
    tc_init.xMin = 160
    tc_init.xBins = 200
    tc_init.xData = Top_Mass_From_Children_init
    tc_init.CompileHistogram()
    s.AddObject(tc_init)
    
    s.SaveFigure()

    # Comparison Plot of the Truth Top Mass from children and truth jets
    s = SubfigureCanvas()
    s.Filename = "TopMassesDetector"
 
    tc = TH1F()
    tc.Title =  "The Predicted Truth Top Mass Derived From 'truth_top_child_*'"
    tc.xTitle = "Mass (GeV)"
    tc.yTitle = "Entries"
    tc.xMin = 160
    tc.xBins = 200
    tc.xData = Top_Mass_From_Children
    tc.CompileHistogram()
    s.AddObject(tc)   

    t = TH1F()
    t.Title = "The Predicted Truth Top Mass Derived From 'truth_jets_*' \n (with detector leptons) matched to 'truth_top_child_*'"
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.xMin = 160
    t.xBins = 200
    t.xData = Top_Mass_From_Truth_Jets
    t.CompileHistogram()
    s.AddObject(t)
    
    tc_init = TH1F()
    tc_init.Title = "The Predicted Truth Top Mass Derived From 'truth_jets_*' \n (with detector leptons) matched to 'truth_top_child_init*'"
    tc_init.xTitle = "Mass (GeV)"
    tc_init.yTitle = "Entries"
    tc_init.xMin = 160
    tc_init.xBins = 200
    tc_init.xData = Top_Mass_From_Truth_Jets_init
    tc_init.CompileHistogram()
    s.AddObject(tc_init)
    s.SaveFigure()


    # Comparison of Top mass from truthjets vs top_child information + No Anomalous Event matching 
    s = SubfigureCanvas()
    s.Filename = "TopMassesDetectorNoAnomalous"
 
    tc = TH1F()
    tc.Title =  "The Predicted Truth Top Mass Derived From 'truth_top_child_*'"
    tc.xTitle = "Mass (GeV)"
    tc.yTitle = "Entries"
    tc.xMin = 160
    tc.xBins = 200
    tc.xData = Top_Mass_From_Children
    tc.CompileHistogram()
    s.AddObject(tc)   

    t = TH1F()
    t.Title = "The Predicted Truth Top Mass Derived From 'truth_jets_*' \n (with detector leptons) matched to 'truth_top_child_*'"
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.xMin = 160
    t.xBins = 200
    t.xData = Top_Mass_From_Truth_Jets
    t.CompileHistogram()
    s.AddObject(t)
    
    tc_init = TH1F()
    tc_init.Title = "The Predicted Truth Top Mass Derived From 'truth_jets_*' \n (with detector leptons) matched to 'truth_top_child_*' (No Truth Jet Missmatch)"
    tc_init.xTitle = "Mass (GeV)"
    tc_init.yTitle = "Entries"
    tc_init.xMin = 160
    tc_init.xBins = 200
    tc_init.xData = Top_Mass_From_Truth_Jets_NoAnomaly
    tc_init.CompileHistogram()
    s.AddObject(tc_init)
    s.SaveFigure()

    # Comparison of Top mass from jets vs top_child information + No Anomalous Event matching 
    s = SubfigureCanvas()
    s.Filename = "TopMassesDetectorJet"
 
    tc = TH1F()
    tc.Title =  "The Predicted Truth Top Mass Derived From 'truth_top_child_*'"
    tc.xTitle = "Mass (GeV)"
    tc.yTitle = "Entries"
    tc.xMin = 160
    tc.xBins = 200
    tc.xData = Top_Mass_From_Children
    tc.CompileHistogram()
    s.AddObject(tc)   

    t = TH1F()
    t.Title = "The Predicted Truth Top Mass Derived From 'truth_jets_*'\n (with detector leptons) matched to 'truth_top_child_*'"
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.xMin = 160
    t.xBins = 200
    t.xData = Top_Mass_From_Truth_Jets
    t.CompileHistogram()
    s.AddObject(t)
    
    tc_init = TH1F()
    tc_init.Title = "The Predicted Truth Top Mass Derived From 'jets_*'\n (with detector leptons) matched to 'truth_top_child_*'"
    tc_init.xTitle = "Mass (GeV)"
    tc_init.yTitle = "Entries"
    tc_init.xMin = 160
    tc_init.xBins = 200
    tc_init.xData = Top_Mass_From_Jets
    tc_init.CompileHistogram()
    s.AddObject(tc_init)
    s.SaveFigure()

    return True

def TestResonance():
    x = UnpickleObject("EventGenerator")
    x = x.Events 

    # Top mass containers 
    Res_TruthTops = []
    Res_Child = []
    Res_Child_init = []
    Res_TruthJet = []
    Res_TruthJet_NoAnomaly = []
    Res_Jet = []
    for i in x:
        ev = x[i]["nominal"]
        
        tops = ev.TruthTops
        Z_ = Particle(True)
        Sigs = []
        for t in tops:
            # Signal Tops from Resonance 
            if t.FromRes == 0:
                continue
            Sigs.append(t)

        # Resonance Mass from Truth Tops 
        Z_.Decay = Sigs
        Z_.CalculateMassFromChildren()
        Res_TruthTops.append(Z_.Mass_GeV)
        
        Z_.Decay = []
        # Resonance Mass from Truth Children
        for t in Sigs:
            Z_.Decay += t.Decay
            Z_.Decay_init += t.Decay_init

        Z_.CalculateMassFromChildren()
        Res_Child.append(Z_.Mass_GeV)
        Res_Child_init.append(Z_.Mass_init_GeV)

        Z_.Decay = []
        Z_.Decay_init = []
        #Resonance Mass from Truth Jets
        for t in Sigs:
            for tc in t.Decay:
                Z_.Decay.append(tc)

        Z_.CalculateMassFromChildren()
        Res_TruthJet.append(Z_.Mass_GeV)
        
        Z_.Decay = []
        #Resonance Mass from Truth Jets Good Matching 
        if ev.Anomaly_TruthMatch == False: 
            for t in Sigs:
                for tc in t.Decay:
                    Z_.Decay.append(tc)
        
            Z_.CalculateMassFromChildren()
            Res_TruthJet_NoAnomaly.append(Z_.Mass_GeV)
 
        Z_.Decay = []  
        #Resonance Mass from Truth Jets
        for t in Sigs:
            for tc in t.Decay:
                if tc.Type == "truthjet":
                    Z_.Decay += tc.Decay
                else:
                    Z_.Decay.append(tc)

        Z_.CalculateMassFromChildren()
        Res_Jet.append(Z_.Mass_GeV)
 
    # Tops from Truth information figures 
    s = SubfigureCanvas()
    s.Filename = "ResonanceMassTruthParticles"

    t = TH1F() 
    t.Title = "The Mass of Resonance Using 'truth_top_*'"
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.xBins = 200
    t.xData = Res_TruthTops
    t.CompileHistogram()
    s.AddObject(t)

    tc = TH1F()
    tc.Title = "The Predicted Resonance Mass Derived From 'truth_top_child_*'"
    tc.xTitle = "Mass (GeV)"
    tc.yTitle = "Entries"
    tc.xBins = 200
    tc.xData = Res_Child
    tc.CompileHistogram()
    s.AddObject(tc)   

    tc_init = TH1F()
    tc_init.Title = "The Predicted Resonance Mass Derived From 'truth_top_child_init*'"
    tc_init.xTitle = "Mass (GeV)"
    tc_init.yTitle = "Entries"
    tc_init.xBins = 200
    tc_init.xData = Res_Child_init
    tc_init.CompileHistogram()
    s.AddObject(tc_init)
    
    s.SaveFigure()

    # Comparison Plot of the Truth Top Mass from children and truth jets
    s = SubfigureCanvas()
    s.Filename = "ResonanceMassTruthJets"
 
    t = TH1F() 
    t.Title = "The Mass of Resonance Using 'truth_top_*'"
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.xBins = 200
    t.xData = Res_TruthTops
    t.CompileHistogram()
    s.AddObject(t)

    tc = TH1F()
    tc.Title =  "The Predicted Resonance Mass Derived From 'truth_jets_*'\n (with Detector Leptons)"
    tc.xTitle = "Mass (GeV)"
    tc.yTitle = "Entries"
    tc.xBins = 200
    tc.xData = Res_TruthJet
    tc.CompileHistogram()
    s.AddObject(tc)   

    tc_init = TH1F()
    tc_init.Title = "The Predicted Resonance Mass Derived From 'truth_jets_*'\n (with Detector Leptons) (Good Matching)"
    tc_init.xTitle = "Mass (GeV)"
    tc_init.yTitle = "Entries"
    tc_init.xBins = 200
    tc_init.xData = Res_TruthJet_NoAnomaly
    tc_init.CompileHistogram()
    s.AddObject(tc_init)
    
    s.SaveFigure()


    # Comparison of Top mass from truthjets vs top_child information + No Anomalous Event matching 
    s = SubfigureCanvas()
    s.Filename = "ResonanceMassDetector"
 
    t = TH1F() 
    t.Title = "The Mass of Resonance Using 'truth_top_*'"
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.xBins = 200
    t.xData = Res_TruthTops
    t.CompileHistogram()
    s.AddObject(t)

    tc = TH1F()
    tc.Title =  "The Predicted Resonance Mass Derived From 'truth_jets_*'\n (with Detector Leptons)"
    tc.xTitle = "Mass (GeV)"
    tc.yTitle = "Entries"
    tc.xBins = 200
    tc.xData = Res_TruthJet
    tc.CompileHistogram()
    s.AddObject(tc)   

    tc_init = TH1F()
    tc_init.Title = "The Predicted Resonance Mass Derived From 'jets_*'\n (with Detector Leptons)"
    tc_init.xTitle = "Mass (GeV)"
    tc_init.yTitle = "Entries"
    tc_init.xBins = 200
    tc_init.xData = Res_Jet
    tc_init.CompileHistogram()
    s.AddObject(tc_init)
    
    s.SaveFigure()
    
    return True

def TestBackGroundProcesses():
    def CreateEvents(direc, event):
        ev = EventGenerator(direc, Stop = event)
        ev.SpawnEvents()
        ev.CompileEvent(SingleThread = True)
        return ev

    def GetNJets(ev, tree):
        Jet_Number = []
        for i in ev.Events:
            Event = ev.Events[i][tree]
            
            n = 0
            for k in Event.DetectorParticles:
                if k.Type == "jet":
                    n += 1
            Jet_Number.append(n)
        return Jet_Number
    
    dir_ttbar = "/CERN/Grid/Samples/NAF/2021-05-05-2cRC-all/mc16a/postProcessed_ttbar_PhPy8_Total.root"
    dir_4t = "/CERN/Grid/Samples/NAF/2021-05-05-2cRC-all/mc16a/postProcessed_4tops_aMCPy8_AFII.root"
    dir_SingleT = "/CERN/Grid/Samples/NAF/2021-05-05-2cRC-all/mc16a/postProcessed_SingleTop.root"
    dir_SingleT_Rare = "/CERN/Grid/Samples/NAF/2021-05-05-2cRC-all/mc16a/postProcessed_SingleTop_rare.root"
    
    dir_ttH = "/CERN/Grid/Samples/NAF/2021-05-05-2cRC-all/mc16a/postProcessed_ttH.root"
    dir_ttW = "/CERN/Grid/Samples/NAF/2021-05-05-2cRC-all/mc16a/postProcessed_ttW.root"

    dir_ttZ = "/CERN/Grid/Samples/NAF/2021-05-05-2cRC-all/mc16a/postProcessed_ttZ.root"
    dir_VH = "/CERN/Grid/Samples/NAF/2021-05-05-2cRC-all/mc16a/postProcessed_VH.root"

    dir_VV = "/CERN/Grid/Samples/NAF/2021-05-05-2cRC-all/mc16a/postProcessed_VV.root"
    dir_Wjets = "/CERN/Grid/Samples/NAF/2021-05-05-2cRC-all/mc16a/postProcessed_Wjets_Sherpa221.root"

    dir_Zjets_ee = "/CERN/Grid/Samples/NAF/2021-05-05-2cRC-all/mc16a/postProcessed_Zjets_ee_Sherpa221.root"
    dir_Zjets_mumu = "/CERN/Grid/Samples/NAF/2021-05-05-2cRC-all/mc16a/postProcessed_Zjets_mumu_Sherpa221.root"
    dir_Zjets_tautau = "/CERN/Grid/Samples/NAF/2021-05-05-2cRC-all/mc16a/postProcessed_Zjets_tautau_Sherpa221.root"

    
    Processes = [dir_ttbar, dir_4t, dir_SingleT, dir_SingleT_Rare, dir_ttH, dir_VH, dir_VV, dir_Wjets, dir_Zjets_ee, dir_Zjets_tautau, dir_Zjets_mumu]
    Map = {}
    for i in Processes:
        p = i.split("/")
        proc = p[len(p) - 1].strip(".root")
        name = "-".join(proc.split("_")[1:])
        Map[name] = i
        
        ev = CreateEvents(i, 5000)
        PickleObject(ev, proc)
   
    Jets = []
    Names = []
    for i in Map:
        ev = UnpickleObject("postProcessed_" + i.replace("-", "_"))
        Jets.append(GetNJets(ev, "tree"))
        Names.append(i)
    
    J = []
    for i in Jets:
        l = 0
        for k in i:
            l += k
        J.append(k)
    
    Njet_P = TH2F()
    Njet_P.Title = "N-Jets vs Process Considered"
    Njet_P.Filename = "n_jets_process"
    Njet_P.yData = Jets
    Njet_P.xData = Names
    Njet_P.yMin = 0
    Njet_P.xTitle = "Process"
    Njet_P.yTitle = "N-Jets"
    Njet_P.SaveFigure("Plots/Njets/")
    return True
