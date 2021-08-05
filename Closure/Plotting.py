# Produce the plotting of the events in the analysis (basically a sanity check) 
from Functions.Event.Event import EventGenerator
from Functions.Particles.Particles import Particle
from Functions.IO.IO import PickleObject, UnpickleObject
from Functions.Plotting.Histograms import TH1F, SubfigureCanvas

def TestTops():
    dir = "/home/tnom6927/Downloads/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root/user.pgadow.24765302._000001.output.root"
    
    Events = -1
    x = EventGenerator(dir, DebugThresh = Events)
    x.SpawnEvents()
    x.CompileEvent()

    #PickleObject(x, "AllEvents")
    #x = UnpickleObject("AllEvents")
    x = x.Events

    # Top mass containers 
    Top_Mass = []
    Top_Mass_From_Children = []
    Top_Mass_From_Children_init = []
    Top_Mass_From_Detector = []
    Top_Mass_From_Detector_init = []
    Top_Mass_From_Detector_NoAnomaly = []
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
            tmp = []
            for j in k.Decay:
                tmp += j.Decay
            k.Decay = tmp
            tmp = []
            for j in k.Decay_init:
                tmp += j.Decay_init
            k.Decay_init = tmp
            
            k.CalculateMassFromChildren()
            Top_Mass_From_Detector.append(k.Mass_GeV)
            Top_Mass_From_Detector_init.append(k.Mass_init_GeV)
        
            if ev.Anomaly == True:
                continue
            Top_Mass_From_Detector_NoAnomaly.append(k.Mass_GeV)



    # Tops from Truth information figures 
    s = SubfigureCanvas()
    s.Filename = "TopMasses"

    t = TH1F() 
    t.Title = "Mass of the Truth Tops"
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.Bins = 200
    t.Data = Top_Mass
    t.CompileHistogram()
    s.AddObject(t)

    tc = TH1F()
    tc.Title = "Mass of the Tops From Children"
    tc.xTitle = "Mass (GeV)"
    tc.yTitle = "Entries"
    tc.Bins = 200
    tc.xMin = 160
    tc.Data = Top_Mass_From_Children
    tc.CompileHistogram()
    s.AddObject(tc)   

    tc_init = TH1F()
    tc_init.Title = "Mass of the Tops From Children INIT"
    tc_init.xTitle = "Mass (GeV)"
    tc_init.yTitle = "Entries"
    tc_init.xMin = 160
    tc_init.Bins = 200
    tc_init.Data = Top_Mass_From_Children_init
    tc_init.CompileHistogram()
    s.AddObject(tc_init)
    
    s.CompileFigure()
    s.SaveFigure()

    # Tops from Truth + Detector information 
    s = SubfigureCanvas()
    s.Filename = "TopMassesDetector"
 
    tc = TH1F()
    tc.Title = "Mass of the Tops From Children (Monte Carlo Truth)"
    tc.xTitle = "Mass (GeV)"
    tc.yTitle = "Entries"
    tc.xMin = 160
    tc.Bins = 200
    tc.Data = Top_Mass_From_Children
    tc.CompileHistogram()
    s.AddObject(tc)   

    t = TH1F()
    t.Title = "Mass of the Truth Tops Detector"
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.xMin = 160
    t.Bins = 200
    t.Data = Top_Mass_From_Detector
    t.CompileHistogram()
    s.AddObject(t)
    
    tc_init = TH1F()
    tc_init.Title = "Mass of the Tops Detector INIT"
    tc_init.xTitle = "Mass (GeV)"
    tc_init.yTitle = "Entries"
    tc_init.xMin = 160
    tc_init.Bins = 200
    tc_init.Data = Top_Mass_From_Detector_init
    tc_init.CompileHistogram()
    s.AddObject(tc_init)
    
    s.CompileFigure()
    s.SaveFigure()


    # Tops from Truth + Detector information + No Anomalous Events
    s = SubfigureCanvas()
    s.Filename = "TopMassesDetectorNoAnomaly"

    tc = TH1F()
    tc.Title = "Mass of the Tops From Init Children (Monte Carlo Truth)"
    tc.xTitle = "Mass (GeV)"
    tc.yTitle = "Entries"
    tc.xMin = 160
    tc.Bins = 200
    tc.Data = Top_Mass_From_Children_init
    tc.CompileHistogram()
    s.AddObject(tc)   
    
    t = TH1F()
    t.Title = "Mass of Tops From Detector"
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.xMin = 160
    t.Bins = 200
    t.Data = Top_Mass_From_Detector
    t.CompileHistogram()
    s.AddObject(t)
    
    tc_init = TH1F()
    tc_init.Title = "Mass of the Tops Detector Without Anomalous Events"
    tc_init.xTitle = "Mass (GeV)"
    tc_init.yTitle = "Entries"
    tc_init.xMin = 160
    tc_init.Bins = 200
    tc_init.Data = Top_Mass_From_Detector_NoAnomaly
    tc_init.CompileHistogram()
    s.AddObject(tc_init)
    
    s.CompileFigure()
    s.SaveFigure()



















        
def TestResonance():
    dir = "/home/tnom6927/Downloads/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root/user.pgadow.24765302._000001.output.root"
    
    #Events = -1
    #x = EventGenerator(dir, DebugThresh = Events)
    #x.SpawnEvents()
    #x.CompileEvent()

    #PickleObject(x, "AllEvents")
    x = UnpickleObject("AllEvents")
    x = x.Events

    # Top mass containers 
    Top_Mass = []
    Top_Mass_From_Children = []
    Top_Mass_From_Children_init = []
    Resonance_From_Tops = []
    Resonance_From_Children = []
    Resonance_From_Children_Init = []
    
    Resonance_From_Detector_Objects = []
    Resonance_From_Detector_Objects_Init = []
    for i in x:
        ev = x[i]["nominal"]
        
        Z_ = Particle(True)
        tops = ev.TruthTops
        for k in tops:
            k.CalculateMass()
            Top_Mass.append(k.Mass_GeV)
            if k.FromRes == 1:
                Z_.Decay.append(k)

        Z_.CalculateMassFromChildren()
        Resonance_From_Tops.append(Z_.Mass_GeV)
        Z_.Decay = []
        Z_.Decay_init = []
        for k in tops:
            k.CalculateMassFromChildren()
            Top_Mass_From_Children.append(k.Mass_GeV)
            Top_Mass_From_Children_init.append(k.Mass_init_GeV)
            
            if k.FromRes == 1:
                Z_.Decay += k.Decay
                Z_.Decay_init += k.Decay_init

        Z_.CalculateMassFromChildren()
        Resonance_From_Children.append(Z_.Mass_GeV)
        Resonance_From_Children_Init.append(Z_.Mass_init_GeV)
        
        Z_.Decay = []
        Z_.Decay_init = []
        for k in tops:
            if k.FromRes != 1:
                continue
            for j in k.Decay:
                Z_.Decay += j.Decay

            for j in k.Decay_init: 
                Z_.Decay_init += j.Decay_init

        Z_.CalculateMassFromChildren()
        Resonance_From_Detector_Objects.append(Z_.Mass_GeV)
        Resonance_From_Detector_Objects_Init.append(Z_.Mass_init_GeV)

    # Tops figures 
    t = TH1F()
    t.Title = "Mass of the Truth Tops"
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.Bins = 1000
    t.Data = Top_Mass
    t.CompileHistogram()

    tc = TH1F()
    tc.Title = "Mass of the Tops From Children"
    tc.xTitle = "Mass (GeV)"
    tc.yTitle = "Entries"
    tc.Bins = 1000
    tc.Data = Top_Mass_From_Children
    tc.CompileHistogram()

    tc_init = TH1F()
    tc_init.Title = "Mass of the Tops From Children INIT"
    tc_init.xTitle = "Mass (GeV)"
    tc_init.yTitle = "Entries"
    tc_init.Bins = 1000
    tc_init.Data = Top_Mass_From_Children_init
    tc_init.CompileHistogram()

    s = SubfigureCanvas()
    s.Filename = "TopMasses"
    s.AddObject(t)
    s.AddObject(tc)   
    s.AddObject(tc_init)
    s.CompileFigure()
    s.SaveFigure()
