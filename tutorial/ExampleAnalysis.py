from AnalysisTopGNN import Analysis
from Event import Event 
from EventGraph import SimpleDataGraph
from Features import *
from GNN import ExampleGNN










from AnalysisTopGNN import Analysis
from Event import Event
from AnalysisTopGNN.Plotting import TH1F


# // ======================== Running the Event Compiler ============================= //
# Sample 1: Resonance Top Sample
Ana = Analysis()
Ana.InputSample("Test", "/home/tnom6927/Downloads/samples/tttt/DAOD_TOPQ1.21955717._000003.root")
Ana.Event = Event
Ana.EventCache = True
Ana.Threads = 12
Ana.VerboseLevel = 1
Ana.Launch()

# Sample 2: Adding some Background 
Ana2 = Analysis()
Ana2.InputSample("Test", "/home/tnom6927/Downloads/samples/t/QU_14.root")
Ana2.Event = Event
Ana2.EventCache = True
Ana2.Threads = 12
Ana2.VerboseLevel = 1
Ana2.Launch()

All = Ana + Ana2

ResonanceMass = []
for i in All:
    
    # Access the event properties 
    ev = i.Trees["nominal"]
   
    # Collect resonance tops 
    resonance = []
    for top in ev.Tops:
        if top.FromResonance == 0:
            continue
        resonance.append(top)

    # Sum tops quarks from resonance.
    Res = sum(resonance)

    # if the list is empty, Res will be 0.
    if Res == 0:
        continue
    ResonanceMass.append(Res.Mass)

Res = TH1F()
Res.xData = ResonanceMass
Res.Title = "Invariant Mass of Resonance Derived from Resonance Tops"
Res.xTitle = "Mass of Resonance (GeV)"
Res.yTitle = "Entries (Arb.)"
Res.xMin = 0 
Res.xStep = 100
Res.Filename = "MassOfResonance"
Res.OutputDirectory = "Example"
Res.SaveFigure()




    # ===== Some nice additional features ===== #
    # Get the event hash
    # hash_ = i.Filename
    
    # Get the ROOT name from which this event originate from
    # rootname = All.HashToROOT(hash_)
    
    # Return a specific event object from the hash 
    # SpecificEvent = All[hash_]
    
    # Check if a given event hash is present in sample:
    # print(hash_ in Ana2)
     
    # Retrieve event object attributes  
    # print(ev.phi)
    # print(ev.met)
    # print(ev.SomeJets)
    # print(ev.Tops)
 







exit()

# Do some fancy condor submission scripting 
from AnalysisTopGNN.Submission import Condor

#Sample 1: Resonance Top Sample
ttt = Analysis()
ttt.InputSample("tttt", "/home/tnom6927/Downloads/samples/tttt/DAOD_TOPQ1.21955717._000003.root")
ttt.Event = Event
ttt.EventCache = True
ttt.DumpPickle = True # < Dumps events as pickle files 
ttt.VerboseLevel = 1
ttt.Threads = 12

# Sample 2: Adding some Background 
ttbar = Analysis()
ttbar.InputSample("ttbar", "/home/tnom6927/Downloads/samples/ttbar/DAOD_TOPQ1.27296255._000017.root")
ttbar.Event = Event
ttbar.EventCache = True
ttbar.DumpPickle = True 
ttbar.VerboseLevel = 1
ttbar.Threads = 12


# Create Graph Data
AnaGraph = Analysis()
AnaGraph.InputSample("tttt")
AnaGraph.InputSample("ttbar")
AnaGraph.DataCache = True
AnaGraph.DumpHDF5 = True
AnaGraph.VerboseLevel = 1
AnaGraph.EventGraph = SimpleDataGraph

# Add Truth to the Graph 
AnaGraph.AddGraphTruth(SignalEvent, "SignalEvent")
AnaGraph.AddNodeTruth(TruthNodeFromRes, "FromRes")
AnaGraph.AddEdgeTruth(TruthResonancePair, "ResPairs")

# Add Observable Attributes to Graph, i.e. things we train the GNN on.
AnaGraph.AddGraphFeature(nJets, "nJets")
AnaGraph.AddNodeFeature(Energy, "e")
AnaGraph.AddNodeFeature(Phi, "phi")
AnaGraph.AddNodeFeature(Eta, "eta")
AnaGraph.AddNodeFeature(PT, "pt")
AnaGraph.AddEdgeFeature(EdgeMass, "mass")

# Whether to construct a fully connected Graph with Self-Loops for nodes.
AnaGraph.FullyConnect = True 
AnaGraph.SelfLoop = True 

# Training a very simple Graph Neural Network
Op = Analysis()
Op.InputSample("tttt")
Op.InputSample("ttbar")
Op.Device = "cuda"
Op.Tree = "nominal"
Op.VerboseLevel = 1
Op.Epochs = 10
Op.kFolds = 10
Op.BatchSize = 10
Op.Optimizer = {"ADAM" : {"lr" : 0.001, "weight_decay" : 0.001}}
Op.Scheduler = {"ExponentialLR" : {"gamma" : 0.5}}
Op.Model = ExampleGNN() # <==== Some defined Graph Neural Network Model

# Offload the compiling to Condor...
Con = Condor()
Con.ProjectName = "HelloWorld"
Con.CondaEnv = "GNN"
Con.AddJob("tttt", Ana, "12GB", "1h")
Con.AddJob("ttbar", Ana2, "12GB", "1h")
Con.AddJob("GraphSample", AnaGraph, "12GB", "12h", waitfor = ["tttt", "ttbar"])
Con.AddJob("GNN", Op, "12GB", "1h", waitfor = ["GraphSample"])

#Con.LocalDryRun() # Test the script before submission 
Con.DumpCondorJobs()

