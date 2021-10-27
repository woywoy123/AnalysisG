# Produce the plotting of the events in the analysis (basically a sanity check) 
from Functions.Event.Event import EventGenerator
from Functions.Particles.Particles import Particle
from Functions.IO.IO import PickleObject, UnpickleObject
from Functions.Plotting.Histograms import TH2F, TH1F, SubfigureCanvas, CombineHistograms

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
    
    s.CompileFigure()
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
    
    s.CompileFigure()
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
        TT.xBins = 1000
        TT.xMin = 0
        TT.xMax = 5000
        TT.xData = ResonanceFromSignalTops

        TS = TH1F()
        TS.Title = "SignalSpectator"
        TS.xTitle = "Mass (GeV)"
        TS.yTitle = "Entries"
        TS.xBins = 1000
        TS.xMin = 0
        TS.xMax = 5000
        TS.xData = SpectatorSignalTops

        SS = TH1F()
        SS.Title = "Spectator"
        SS.xTitle = "Mass (GeV)"
        SS.yTitle = "Entries"
        SS.xBins = 1000
        SS.xMin = 0
        SS.xMax = 5000
        SS.xData = ResonanceFromSpectatorTops

        Com = CombineHistograms()
        Com.Histograms = [TT, TS, SS]
        Com.Alpha = 0.7
        Com.Title = "Mass Spectrum of Different Top Pair Combinations with: " + energy + " " + i.split(".")[5]
        Com.CompileStack()
        Com.Save("Plots/Spectrum/")


def TestRCJetAssignments():
    
    signal_dir = "/home/tnom6927/Downloads/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root/user.pgadow.24765302._000001.output.root"
    background_dir = "/home/tnom6927/Downloads/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root/postProcessed_ttW.root"

    Event = -1
    back = EventGenerator(background_dir, DebugThresh = Event)
    back.SpawnEvents()
    back.CompileEvent()

    sig = EventGenerator(signal_dir, DebugThresh = Event)
    sig.SpawnEvents()
    sig.CompileEvent()


    #PickleObject(sig, "Signal")
    #PickleObject(back, "ttW")   
    
    #sig = UnpickleObject("Signal")
    #back = UnpickleObject("ttW")
    
    res_truth_tops = []
    res_rc_onlysignal = []
    res_rc_atleastone = []
    res_jets = []

    signa_rc = []
    back_rc = []

    signa_rc_Flav = []
    signa_rc_Mass = []
    
    for i in sig.Events:
        sig_events = sig.Events[i]["nominal"]
        
        # Fill truth resonance from Tops 
        Z_prime = Particle(True)
        for t in sig_events.TruthTops:
            if t.Signal == 1:
                Z_prime.Decay.append(t)
        Z_prime.CalculateMassFromChildren()
        res_truth_tops.append(Z_prime.Mass_GeV)
        
        # Get Resonance where all RCJet consistuents are matched to signal
        Z_prime_all = Particle(True)
        for rc in sig_events.DetectorParticles:
            if rc.Type == "rcjet" or rc.Type == "el" or rc.Type == "mu":
                pass
            else:
                continue
            if rc.Signal == 1:
                Z_prime_all.Decay.append(rc)
        
        Z_prime_all.CalculateMassFromChildren()
        if Z_prime_all.Mass_GeV > 10:
            for k in Z_prime_all.Decay:
                if k.Type == "rcjet":
                    for f in k.Constituents:
                        if f.Type == "rcjetsub":
                            signa_rc_Flav.append(f.Flav)
                            signa_rc_Mass.append(Z_prime_all.Mass_GeV)
                elif k.Type == "mu":
                    signa_rc_Flav.append(13)
                    signa_rc_Mass.append(Z_prime_all.Mass_GeV)
                elif k.Type == "el":
                    signa_rc_Flav.append(11)
                    signa_rc_Mass.append(Z_prime_all.Mass_GeV)
            res_rc_onlysignal.append(Z_prime_all.Mass_GeV)
            
        # Get Resonance where at least one RCjet consistuent is signal
        Z_prime_one = Particle(True)
        for rc in sig_events.DetectorParticles:
            if rc.Type == "rcjet" or rc.Type == "el" or rc.Type == "mu":
                pass
            else:
                continue
            if rc.Signal == 2:
                Z_prime_one.Decay.append(rc)
        
        Z_prime_one.CalculateMassFromChildren()
        if Z_prime_one.Mass_GeV > 10:
            res_rc_atleastone.append(Z_prime_one.Mass_GeV)
        
        # Reconstruct Resonance from jet objects
        Z_prime_jets = Particle(True)
        for rc in sig_events.DetectorParticles:
            if rc.Type == "jet" or rc.Type == "el" or rc.Type == "mu":
                pass
            else:
                continue
            
            if rc.Signal == 1:
                Z_prime_jets.Decay.append(rc)
        Z_prime_jets.CalculateMassFromChildren()
        if Z_prime_jets.Mass_GeV > 10:
            res_jets.append(Z_prime_jets.Mass_GeV)
        
        #Calculate mass of jets
        for rc in sig_events.RCJets:
            rc.CalculateMass()
            signa_rc.append(rc.Mass_GeV)
        
    
        #Calculate mass of jets
        try:
            back_events = back.Events[i]["tree"]
            for rc in back_events.RCJets:
                rc.CalculateMass()
                back_rc.append(rc.Mass_GeV)
        except:
            pass

    Res_TT = TH1F()
    Res_TT.Title = "Res->TruthTops"
    Res_TT.xTitle = "Mass (GeV)"
    Res_TT.yTitle = "Entries"
    Res_TT.xBins = 1500
    Res_TT.xMin = 0
    Res_TT.xMax = 1500
    Res_TT.xData = res_truth_tops
    Res_TT.SaveFigure("Plots/RCJetSpectrum/")

    Res_Jet = TH1F()
    Res_Jet.Title = "Res->Jets+Leptons"
    Res_Jet.xTitle = "Mass (GeV)"
    Res_Jet.yTitle = "Entries"
    Res_Jet.xBins = 1500
    Res_Jet.xMin = 0
    Res_Jet.xMax = 1500
    Res_Jet.xData = res_jets
    Res_Jet.SaveFigure("Plots/RCJetSpectrum/")

    ResRCAll_TT = TH1F()
    ResRCAll_TT.Title = "Res->RCJet_OnlySignal+Leptons"
    ResRCAll_TT.xTitle = "Mass (GeV)"
    ResRCAll_TT.yTitle = "Entries"
    ResRCAll_TT.xBins = 1500
    ResRCAll_TT.xMin = 0
    ResRCAll_TT.xMax = 1500
    ResRCAll_TT.xData = res_rc_onlysignal
    ResRCAll_TT.SaveFigure("Plots/RCJetSpectrum/")

    ResRCOne_TT = TH1F()
    ResRCOne_TT.Title = "Res->RCJet_AtLeastOneSignal+Leptons"
    ResRCOne_TT.xTitle = "Mass (GeV)"
    ResRCOne_TT.yTitle = "Entries"
    ResRCOne_TT.xBins = 1500
    ResRCOne_TT.xMin = 0
    ResRCOne_TT.xMax = 1500
    ResRCOne_TT.xData = res_rc_atleastone
    ResRCOne_TT.SaveFigure("Plots/RCJetSpectrum/")

    RC_Sig = TH1F()
    RC_Sig.Title = "Signal->RCJet_Mass"
    RC_Sig.xTitle = "Mass (GeV)"
    RC_Sig.yTitle = "Entries"
    RC_Sig.xBins = 1500
    RC_Sig.xMin = 0
    RC_Sig.xMax = 1500
    RC_Sig.xData = signa_rc 
    RC_Sig.SaveFigure("Plots/RCJetSpectrum/")
   
    RC_TTW = TH1F()
    RC_TTW.Title = "Background->RCJet_ttW"
    RC_TTW.xTitle = "Mass (GeV)"
    RC_TTW.yTitle = "Entries"
    RC_TTW.xBins = 1500
    RC_TTW.xMin = 0
    RC_TTW.xMax = 1500
    RC_TTW.xData = res_rc_atleastone   
    RC_TTW.SaveFigure("Plots/RCJetSpectrum/")

    Res = CombineHistograms()
    Res.Histograms = [Res_TT, Res_Jet, ResRCAll_TT, ResRCOne_TT]
    Res.Alpha = 0.7
    Res.Title = "Mass Spectrum for Resonance From Truth Tops and RC and Jets"
    Res.Save("Plots/RCJetSpectrum/")

    Com = CombineHistograms()
    Com.Histograms = [RC_Sig, RC_TTW]
    Com.Alpha = 0.7
    Com.Title = "Mass of RC Jets from Signal and Background Samples"
    Com.Save("Plots/RCJetSpectrum/")


    MassPID = TH2F()
    MassPID.Title = "RC Jet Mass Signal Sample vs PID"
    MassPID.xBins = 60 #max(signa_rc_Flav)
    MassPID.xMin = 0
    MassPID.xTitle = "Flavour"
    MassPID.xData = signa_rc_Flav
    MassPID.yData = signa_rc_Mass
    MassPID.yBins = 500
    MassPID.yMin = 0
    MassPID.yMax = 1500
    MassPID.yTitle = "GeV"
    MassPID.SaveFigure("Plots/RCJetSpectrum/")
