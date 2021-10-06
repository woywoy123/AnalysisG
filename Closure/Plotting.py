# Produce the plotting of the events in the analysis (basically a sanity check) 
from Functions.Event.Event import EventGenerator
from Functions.Particles.Particles import Particle
from Functions.IO.IO import PickleObject, UnpickleObject
from Functions.Plotting.Histograms import TH1F, SubfigureCanvas, CombineHistograms

def TestTops():
    dir = "/home/tnom6927/Downloads/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root/user.pgadow.24765302._000001.output.root"
    
    Events = -1
    x = EventGenerator(dir, DebugThresh = Events)
    x.SpawnEvents()
    x.CompileEvent()
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
    t.Bins = 200
    t.Data = Top_Mass
    t.CompileHistogram()
    s.AddObject(t)

    tc = TH1F()
    tc.Title = "The Predicted Truth Top Mass Derived From 'truth_top_child_*'"
    tc.xTitle = "Mass (GeV)"
    tc.yTitle = "Entries"
    tc.Bins = 200
    tc.xMin = 160
    tc.Data = Top_Mass_From_Children
    tc.CompileHistogram()
    s.AddObject(tc)   

    tc_init = TH1F()
    tc_init.Title = "The Predicted Truth Top Mass Derived From 'truth_top_child_init_*'"
    tc_init.xTitle = "Mass (GeV)"
    tc_init.yTitle = "Entries"
    tc_init.xMin = 160
    tc_init.Bins = 200
    tc_init.Data = Top_Mass_From_Children_init
    tc_init.CompileHistogram()
    s.AddObject(tc_init)
    
    s.CompileFigure()
    s.SaveFigure()

    # Comparison Plot of the Truth Top Mass from children and truth jets
    s = SubfigureCanvas()
    s.Filename = "TopMassesDetector"
 
    tc = TH1F()
    tc.Title =  "The Predicted Truth Top Mass Derived From 'truth_top_child_*'"
    tc.xTitle = "Mass (GeV)"
    tc.yTitle = "Entries"
    tc.xMin = 160
    tc.Bins = 200
    tc.Data = Top_Mass_From_Children
    tc.CompileHistogram()
    s.AddObject(tc)   

    t = TH1F()
    t.Title = "The Predicted Truth Top Mass Derived From 'truth_jets_*' \n (with detector leptons) matched to 'truth_top_child_*'"
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.xMin = 160
    t.Bins = 200
    t.Data = Top_Mass_From_Truth_Jets
    t.CompileHistogram()
    s.AddObject(t)
    
    tc_init = TH1F()
    tc_init.Title = "The Predicted Truth Top Mass Derived From 'truth_jets_*' \n (with detector leptons) matched to 'truth_top_child_init*'"
    tc_init.xTitle = "Mass (GeV)"
    tc_init.yTitle = "Entries"
    tc_init.xMin = 160
    tc_init.Bins = 200
    tc_init.Data = Top_Mass_From_Truth_Jets_init
    tc_init.CompileHistogram()
    s.AddObject(tc_init)
    
    s.CompileFigure()
    s.SaveFigure()


    # Comparison of Top mass from truthjets vs top_child information + No Anomalous Event matching 
    s = SubfigureCanvas()
    s.Filename = "TopMassesDetectorNoAnomalous"
 
    tc = TH1F()
    tc.Title =  "The Predicted Truth Top Mass Derived From 'truth_top_child_*'"
    tc.xTitle = "Mass (GeV)"
    tc.yTitle = "Entries"
    tc.xMin = 160
    tc.Bins = 200
    tc.Data = Top_Mass_From_Children
    tc.CompileHistogram()
    s.AddObject(tc)   

    t = TH1F()
    t.Title = "The Predicted Truth Top Mass Derived From 'truth_jets_*' \n (with detector leptons) matched to 'truth_top_child_*'"
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.xMin = 160
    t.Bins = 200
    t.Data = Top_Mass_From_Truth_Jets
    t.CompileHistogram()
    s.AddObject(t)
    
    tc_init = TH1F()
    tc_init.Title = "The Predicted Truth Top Mass Derived From 'truth_jets_*' \n (with detector leptons) matched to 'truth_top_child_*' (No Truth Jet Missmatch)"
    tc_init.xTitle = "Mass (GeV)"
    tc_init.yTitle = "Entries"
    tc_init.xMin = 160
    tc_init.Bins = 200
    tc_init.Data = Top_Mass_From_Truth_Jets_NoAnomaly
    tc_init.CompileHistogram()
    s.AddObject(tc_init)
    
    s.CompileFigure()
    s.SaveFigure()

    # Comparison of Top mass from jets vs top_child information + No Anomalous Event matching 
    s = SubfigureCanvas()
    s.Filename = "TopMassesDetectorJet"
 
    tc = TH1F()
    tc.Title =  "The Predicted Truth Top Mass Derived From 'truth_top_child_*'"
    tc.xTitle = "Mass (GeV)"
    tc.yTitle = "Entries"
    tc.xMin = 160
    tc.Bins = 200
    tc.Data = Top_Mass_From_Children
    tc.CompileHistogram()
    s.AddObject(tc)   

    t = TH1F()
    t.Title = "The Predicted Truth Top Mass Derived From 'truth_jets_*'\n (with detector leptons) matched to 'truth_top_child_*'"
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.xMin = 160
    t.Bins = 200
    t.Data = Top_Mass_From_Truth_Jets
    t.CompileHistogram()
    s.AddObject(t)
    
    tc_init = TH1F()
    tc_init.Title = "The Predicted Truth Top Mass Derived From 'jets_*'\n (with detector leptons) matched to 'truth_top_child_*'"
    tc_init.xTitle = "Mass (GeV)"
    tc_init.yTitle = "Entries"
    tc_init.xMin = 160
    tc_init.Bins = 200
    tc_init.Data = Top_Mass_From_Jets
    tc_init.CompileHistogram()
    s.AddObject(tc_init)
    
    s.CompileFigure()
    s.SaveFigure()



def TestResonance():
    dir = "/home/tnom6927/Downloads/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root/user.pgadow.24765302._000001.output.root"

    Events = -1
    x = EventGenerator(dir, DebugThresh = Events)
    x.SpawnEvents()
    x.CompileEvent()
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
    t.Bins = 200
    t.Data = Res_TruthTops
    t.CompileHistogram()
    s.AddObject(t)

    tc = TH1F()
    tc.Title = "The Predicted Resonance Mass Derived From 'truth_top_child_*'"
    tc.xTitle = "Mass (GeV)"
    tc.yTitle = "Entries"
    tc.Bins = 200
    tc.Data = Res_Child
    tc.CompileHistogram()
    s.AddObject(tc)   

    tc_init = TH1F()
    tc_init.Title = "The Predicted Resonance Mass Derived From 'truth_top_child_init*'"
    tc_init.xTitle = "Mass (GeV)"
    tc_init.yTitle = "Entries"
    tc_init.Bins = 200
    tc_init.Data = Res_Child_init
    tc_init.CompileHistogram()
    s.AddObject(tc_init)
    
    s.CompileFigure()
    s.SaveFigure()

    # Comparison Plot of the Truth Top Mass from children and truth jets
    s = SubfigureCanvas()
    s.Filename = "ResonanceMassTruthJets"
 
    t = TH1F() 
    t.Title = "The Mass of Resonance Using 'truth_top_*'"
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.Bins = 200
    t.Data = Res_TruthTops
    t.CompileHistogram()
    s.AddObject(t)

    tc = TH1F()
    tc.Title =  "The Predicted Resonance Mass Derived From 'truth_jets_*'\n (with Detector Leptons)"
    tc.xTitle = "Mass (GeV)"
    tc.yTitle = "Entries"
    tc.Bins = 200
    tc.Data = Res_TruthJet
    tc.CompileHistogram()
    s.AddObject(tc)   

    tc_init = TH1F()
    tc_init.Title = "The Predicted Resonance Mass Derived From 'truth_jets_*'\n (with Detector Leptons) (Good Matching)"
    tc_init.xTitle = "Mass (GeV)"
    tc_init.yTitle = "Entries"
    tc_init.Bins = 200
    tc_init.Data = Res_TruthJet_NoAnomaly
    tc_init.CompileHistogram()
    s.AddObject(tc_init)
    
    s.CompileFigure()
    s.SaveFigure()


    # Comparison of Top mass from truthjets vs top_child information + No Anomalous Event matching 
    s = SubfigureCanvas()
    s.Filename = "ResonanceMassDetector"
 
    t = TH1F() 
    t.Title = "The Mass of Resonance Using 'truth_top_*'"
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.Bins = 200
    t.Data = Res_TruthTops
    t.CompileHistogram()
    s.AddObject(t)

    tc = TH1F()
    tc.Title =  "The Predicted Resonance Mass Derived From 'truth_jets_*'\n (with Detector Leptons)"
    tc.xTitle = "Mass (GeV)"
    tc.yTitle = "Entries"
    tc.Bins = 200
    tc.Data = Res_TruthJet
    tc.CompileHistogram()
    s.AddObject(tc)   

    tc_init = TH1F()
    tc_init.Title = "The Predicted Resonance Mass Derived From 'jets_*'\n (with Detector Leptons)"
    tc_init.xTitle = "Mass (GeV)"
    tc_init.yTitle = "Entries"
    tc_init.Bins = 200
    tc_init.Data = Res_Jet
    tc_init.CompileHistogram()
    s.AddObject(tc_init)
    
    s.CompileFigure()
    s.SaveFigure()


def TestResonanceMassForEnergies():
    dirs = ["user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root",
    "user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r9364_p3980.bsm4t-21.2.164-1-0-mc16a_output_root",
    "user.pgadow.310846.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root",
    "user.pgadow.310846.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r9364_p3980.bsm4t-21.2.164-1-0-mc16a_output_root",
    "user.pgadow.310847.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root",
    "user.pgadow.310847.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r9364_p3980.bsm4t-21.2.164-1-0-mc16a_output_root",
    "user.pgadow.313180.MGPy8EG.DAOD_TOPQ1.e8080_s3126_r10201_p3980.bsm4t-21.2.164-1-0-mc16d_output_root",
    "user.pgadow.313180.MGPy8EG.DAOD_TOPQ1.e8080_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root",
    "user.pgadow.313180.MGPy8EG.DAOD_TOPQ1.e8080_s3126_r9364_p3980.bsm4t-21.2.164-1-0-mc16a_output_root",
    "user.pgadow.313181.MGPy8EG.DAOD_TOPQ1.e8081_s3126_r10201_p3980.bsm4t-21.2.164-1-0-mc16d_output_root",
    "user.pgadow.313181.MGPy8EG.DAOD_TOPQ1.e8081_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root",
    "user.pgadow.313181.MGPy8EG.DAOD_TOPQ1.e8081_s3126_r9364_p3980.bsm4t-21.2.164-1-0-mc16a_output_root",
    "user.pgadow.313346.MGPy8EG.DAOD_TOPQ1.e8080_s3126_r10201_p3980.bsm4t-21.2.164-1-0-mc16d_output_root",
    "user.pgadow.313346.MGPy8EG.DAOD_TOPQ1.e8080_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root",
    "user.pgadow.313346.MGPy8EG.DAOD_TOPQ1.e8080_s3126_r9364_p3980.bsm4t-21.2.164-1-0-mc16a_output_root"]
    
    DSIDKeys = {"310844" : "_1TeV", "310845" : "1TeV", "313346" : "1.25TeV", "310846" : "1.5TeV", "310847" : "2TeV", "313180" : "2.5TeV", "313181" : "3TeV"}

    for i in dirs:
        x = EventGenerator("/CERN/Grid/SignalSamples/"+i)
        x.SpawnEvents()
        x.CompileEvent( particle = "TruthTops" )
        energy = DSIDKeys[i.split(".")[2]]

        ResonanceFromSignalTops = []
        ResonanceFromSpectatorTops = []
        SpectatorSignalTops = []
        for f in x.Events:
            F = x.Events[f]
            for e in F:
                Zp = Particle(True) # Resonance
                Sp = Particle(True) # Spectators
                
                try:
                    tops = F[e]["nominal"].TruthTops
                except TypeError:
                    tops = F["nominal"].TruthTops

                if len(tops) != 4:
                    continue
                
                for t in tops:
                    if t.Signal == 1:
                        Zp.Decay.append(t)
                    if t.Signal == 0:
                        Sp.Decay.append(t)

                    SSp = Particle(True) # Spectator and Signal 
                    for ti in tops:
                        if t.Signal != ti.Signal:
                            SSp.Decay.append(t)
                            SSp.Decay.append(ti)
                    
                    SSp.CalculateMassFromChildren()
                    SpectatorSignalTops.append(SSp.Mass_GeV)
                
                Zp.CalculateMassFromChildren()
                Sp.CalculateMassFromChildren()

                ResonanceFromSignalTops.append(Zp.Mass_GeV)
                ResonanceFromSpectatorTops.append(Sp.Mass_GeV)
                
        TT = TH1F()
        TT.Title = "SignalPair"
        TT.xTitle = "Mass (GeV)"
        TT.yTitle = "Entries"
        TT.Bins = 1000
        TT.xMin = 0
        TT.xMax = 5000
        TT.Data = ResonanceFromSignalTops

        TS = TH1F()
        TS.Title = "SignalSpectator"
        TS.xTitle = "Mass (GeV)"
        TS.yTitle = "Entries"
        TS.Bins = 1000
        TS.xMin = 0
        TS.xMax = 5000
        TS.Data = SpectatorSignalTops

        SS = TH1F()
        SS.Title = "Spectator"
        SS.xTitle = "Mass (GeV)"
        SS.yTitle = "Entries"
        SS.Bins = 1000
        SS.xMin = 0
        SS.xMax = 5000
        SS.Data = ResonanceFromSpectatorTops

        Com = CombineHistograms()
        Com.Histograms = [TT, TS, SS]
        Com.Alpha = 0.7
        Com.Title = "Mass Spectrum of Different Top Pair Combinations with: " + energy + " " + i.split(".")[5]
        Com.CompileStack()
        Com.Save("Plots/Spectrum/")






