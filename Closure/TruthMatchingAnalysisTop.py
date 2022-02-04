from Functions.IO.IO import File, PickleObject, UnpickleObject
from Functions.Event.EventGenerator import EventGenerator
from Functions.Plotting.Histograms import TH2F, TH1F
from Functions.Particles.Particles import Particle

def TestSimpleTruthMatching():
    
    Dir = "/home/tnom6927/Downloads/SimpleTTBAR/Out_0/output.root"

    E = EventGenerator(Dir, Stop = -1)
    E.SpawnEvents(True)
    E.CompileEvent(SingleThread = True)
    del E

    Dir = "/home/tnom6927/Downloads/SimpleTTBAR/Out_2/output.root"

    E = EventGenerator(Dir, Stop = -1)
    E.SpawnEvents(True)
    E.CompileEvent(SingleThread = True)
    del E

    Dir = "/home/tnom6927/Downloads/SimpleTTBAR/Out_3/output.root"

    E = EventGenerator(Dir, Stop = -1)
    E.SpawnEvents(True)
    E.CompileEvent(SingleThread = True)
    del E


    Dir = "/home/tnom6927/Downloads/SimpleTTBAR/Out_4/output.root"

    E = EventGenerator(Dir, Stop = -1)
    E.SpawnEvents(True)
    E.CompileEvent(SingleThread = True)
    del E

    Dir = "/home/tnom6927/Downloads/SimpleTTBAR/Out_5/output.root"

    E = EventGenerator(Dir, Stop = -1)
    E.SpawnEvents(True)
    E.CompileEvent(SingleThread = True)
    del E

    Dir = "/home/tnom6927/Downloads/SimpleTTBAR/Out_6/output.root"

    E = EventGenerator(Dir, Stop = -1)
    E.SpawnEvents(True)
    E.CompileEvent(SingleThread = True)
    del E

    Dir = "/home/tnom6927/Downloads/SimpleTTBAR/Out_7/output.root"

    E = EventGenerator(Dir, Stop = -1)
    E.SpawnEvents(True)
    E.CompileEvent(SingleThread = True)
    del E


    return True

def TestTopShapes():
    Dir = "/home/tnom6927/Downloads/SimpleTTBAR/Out_0/output.root"

    E = EventGenerator(Dir, Stop = -1)
    E.SpawnEvents(True)
    E.CompileEvent(SingleThread = True)
    #
    #PickleObject(E, "Debug.pkl")
    E = UnpickleObject("Debug.pkl")
   

    Top_Mass = []
    Top_MassPreFSR = []
    Top_MassPostFSR = []

    Top_FromChildren_Mass = []
    Top_FromChildren_MassPostFSR = []

    for i in E.Events:
        event = E.Events[i]["nominal"]
        tt = event.TruthTops
        tprf = event.TopPreFSR
        tpof = event.TopPostFSR

        for k in tt:
            k.CalculateMass()
            Top_Mass.append(k.Mass_GeV)

            k.CalculateMassFromChildren()
            Top_FromChildren_Mass.append(k.Mass_init_GeV)
        
        for k in tprf:
            k.CalculateMass()
            Top_MassPreFSR.append(k.Mass_GeV)
 
        for k in tpof:
            k.CalculateMass()
            Top_MassPostFSR.append(k.Mass_GeV)

            k.CalculateMassFromChildren()
            Top_FromChildren_MassPostFSR.append(k.Mass_init_GeV)
 

    # Tops from Truth information figures 
    t = TH1F() 
    t.Title = "Mass of Truth Top using m_truth branch"
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.xBins = 500
    t.xMin = 172
    t.xMax = 173
    t.xData = Top_Mass
    t.Filename = "TruthTops.png"
    t.SaveFigure("Plots/TestCustomAnalysisTop")

    t = TH1F() 
    t.Title = "Mass of Top Based on Ghost Pre-FSR"
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.xBins = 500
    t.xMin = 170
    t.xMax = 175
    t.xData = Top_MassPreFSR
    t.Filename = "TruthTopsPreFSR.png"
    t.SaveFigure("Plots/TestCustomAnalysisTop")

    t = TH1F() 
    t.Title = "Mass of Top Based on Ghost Post-FSR"    
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.xBins = 500
    t.xMin = 170
    t.xMax = 175
    t.xData = Top_MassPostFSR
    t.Filename = "TruthTopsPostFSR.png"
    t.SaveFigure("Plots/TestCustomAnalysisTop")


    t = TH1F() 
    t.Title = "Mass of Truth Top using m_truth branch (Children)"
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.xBins = 500
    t.xMin = 160
    t.xMax = 180
    t.xData = Top_FromChildren_Mass
    t.Filename = "TruthTops_Children.png"
    t.SaveFigure("Plots/TestCustomAnalysisTop")

    t = TH1F() 
    t.Title = "Mass of Top Based on Ghost Post-FSR (Children)"    
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.xBins = 500
    t.xMin = 160
    t.xMax = 180
    t.xData = Top_FromChildren_MassPostFSR
    t.Filename = "TruthTopsPostFSR_Children.png"
    t.SaveFigure("Plots/TestCustomAnalysisTop")














    return True
