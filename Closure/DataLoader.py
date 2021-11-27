from Functions.Event.Event import EventGenerator
from Functions.GNN.Graphs import GenerateDataLoader
from Functions.IO.IO import PickleObject, UnpickleObject
from Functions.Plotting.Histograms import TH1F
from Functions.Particles.Particles import Particle

events = 100
Tree = "nominal"
root_dir = "/CERN/Grid/SignalSamples"

def TestAnomalousStatistics(ev):

    A_D = 0
    A_TM = 0
    A_TM_init = 0
    A_All = 0
    for i in ev.Events:
        Event = ev.Events[i]["nominal"]

        if Event.Anomaly_Detector == True and Event.Anomaly_TruthMatch == True and Event.Anomaly_TruthMatch_init == True: 
            A_All += 1

        if Event.Anomaly_Detector == True: 
            A_D += 1

        if Event.Anomaly_TruthMatch == True: 
            A_TM += 1

        if Event.Anomaly_TruthMatch_init == True: 
            A_TM_init += 1

    print("Number of events considered: " + str(i))
    print("Number of events with Detector, Truth and Truth_init not well matched particles: " + str(A_All) + " " + str(round(100*float(A_All)/float(i), 4)) + "%")
    print("Number of events with Truth not well matched particles: " + str(A_TM) + " " + str(round(100*float(A_TM)/float(i), 4)) + "%")
    print("Number of events with Truth_init not well matched particles: " + str(A_TM_init) + " " + str(round(100*float(A_TM_init)/float(i), 4)) + "%")
    print("Number of events with Detector particles not well matched with Truth particles: " + str(A_D) + " " + str(round(100*float(A_D)/float(i), 4)) + "%")


def TestSignalDirectory():
    dir = root_dir
    
    ev = EventGenerator(dir)
    ev.SpawnEvents()
    ev.CompileEvent(SingleThread = True, particle = "TruthTops")
    
    for i in ev.Events:
        print(i, ev.Events[i], ev.EventIndexFileLookup(i)) 

def TestSingleTopFile():
    dir = "/CERN/Grid/Samples/NAF/2021-05-05-2cRC-all/PlayGround"
    
    ev = EventGenerator(dir)
    ev.SpawnEvents()
    ev.CompileEvent(SingleThread = True)
    PickleObject(ev, "SingleTop")
    
    ev = UnpickleObject("SingleTop")
    TopMass = []
    Masses = []
    for i in ev.Events:
        Event = ev.Events[i]
        Particles = Event["tree"].DetectorParticles
        
        print("Event -> " + str(i))

        Energetic = 0
        Part = ""
        P = Particle(True)
        for k in Particles:
            if "rc" in k.Type:
                continue
 
            if k.Type == "jet":
                if Energetic < k.pt and k.truthPartonLabel == 5:
                    Energetic = k.pt
                    Part = k

                print(k.truthflav, k.truthPartonLabel, k.isTrueHS, k.Type, k.Index)
            
            if k.Type == "mu" or k.Type == "el":
                print(k.charge, k.Type, k.Index)
                P.Decay.append(k)

            k.CalculateMass()
            Masses.append(k.Mass_GeV)

        if isinstance(Part, Particle):
            P.Decay.append(Part)
            P.CalculateMassFromChildren()
            TopMass.append(P.Mass_GeV)
    
    T = TH1F()
    T.Title = "Mass of Top"
    T.xTitle = "Mass (GeV)"
    T.yTitle = "Entries"
    T.xBins = 1000
    T.xData = TopMass
    T.xMin = 0
    T.xMax = 1000
    T.SaveFigure("Plots/SingleTop/")

    T = TH1F()
    T.Title = "Masses of all Particles"
    T.xTitle = "Mass (GeV)"
    T.yTitle = "Entries"
    T.xBins = 150
    T.xData = Masses
    T.xMin = 0
    T.xMax = 150
    T.SaveFigure("Plots/SingleTop/")
