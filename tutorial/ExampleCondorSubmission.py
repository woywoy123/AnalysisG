from AnalysisTopGNN import Analysis 
from AnalysisTopGNN.Submission import Condor 
from ObjectDefinitions.Event import Event 
from ObjectDefinitions.EventGraph import SimpleDataGraph
from ObjectDefinitions.GNN import ExampleGNN
from ObjectDefinitions.Features import *

def MakeEvents(SampleName, Directory):
    Ana = Analysis()
    Ana.InputSample(SampleName, Directory)
    Ana.Event = Event
    Ana.EventCache = True
    Ana.DumpPickle = True # < Dumps events as pickle files 
    Ana.VerboseLevel = 1
    Ana.Threads = 12
    return Ana

def MakeGraphs(SampleName):
    Graph = Analysis()
    Graph.InputSample(SampleName)
    Graph.DataCache = True 
    Graph.DumpHDF5 = True 
    Graph.VerboseLevel = 1
    Graph.Event = Event
    Graph.EventGraph = SimpleDataGraph
    
    # Add Truth to the Graph 
    Graph.AddGraphTruth(SignalEvent, "SignalEvent")
    Graph.AddNodeTruth(TruthNodeFromRes, "FromRes")
    Graph.AddEdgeTruth(TruthResonancePair, "ResPairs")
    
    # Add Observable Attributes to Graph, i.e. things we train the GNN on.
    Graph.AddGraphFeature(nJets, "nJets")
    Graph.AddNodeFeature(Energy, "e")
    Graph.AddNodeFeature(Phi, "phi")
    Graph.AddNodeFeature(Eta, "eta")
    Graph.AddNodeFeature(PT, "pt")
    Graph.AddEdgeFeature(EdgeMass, "mass")
    
    # Whether to construct a fully connected Graph with Self-Loops for nodes.
    Graph.FullyConnect = True 
    Graph.SelfLoop = True 
    return Graph 

SampleDir = "/home/tnom6927/Downloads/samples"

#Sample 1: Resonance Top Sample
tttt = MakeEvents("tttt", SampleDir + "/tttt/DAOD_TOPQ1.21955717._000003.root")
tttt_Graph = MakeGraphs("tttt")

# Sample 2: Adding some Background 
ttbar = MakeEvents("ttbar", SampleDir + "/ttbar/DAOD_TOPQ1.27296255._000017.root")
ttbar_Graph = MakeGraphs("ttbar")

# =============== Merge the samples together and create training/test samples ========== #
CombinedSamples = Analysis()
CombinedSamples.InputSample("ttbar")
CombinedSamples.InputSample("tttt")
CombinedSamples.VerboseLevel = 1
CombinedSamples.DataCache = True # Here we select the Graph objects
CombinedSamples.TrainingSampleName = "ExampleSample" # Shuffle the sample and give it a name 
CombinedSamples.TrainingPercentage = 80 # How much of this we want to use for training 

# ============ Training a very simple Graph Neural Network ============ #
# With scheduler
Op1 = Analysis()
Op1.RunName = "WithScheduler"
Op1.InputSample("tttt")
Op1.InputSample("ttbar")
Op1.Device = "cuda"
Op1.Tree = "nominal"
Op1.TrainingSampleName = "ExampleSample"
Op1.VerboseLevel = 1
Op1.Epochs = 10
Op1.kFolds = 10
Op1.BatchSize = 10
Op1.Optimizer = {"ADAM" : {"lr" : 0.001, "weight_decay" : 0.001}}
Op1.Scheduler = {"ExponentialLR" : {"gamma" : 0.5}}
Op1.Model = ExampleGNN() # <==== Some defined Graph Neural Network Model

# Without scheduler
Op2 = Analysis()
Op2.RunName = "WithoutScheduler"
Op2.InputSample("tttt")
Op2.InputSample("ttbar")
Op2.Device = "cuda"
Op2.Tree = "nominal"
Op2.TrainingSampleName = "ExampleSample"
Op2.VerboseLevel = 1
Op2.Epochs = 10
Op2.kFolds = 10
Op2.BatchSize = 10
Op2.Optimizer = {"ADAM" : {"lr" : 0.001, "weight_decay" : 0.001}}
Op2.Model = ExampleGNN() # <==== Some defined Graph Neural Network Model

# ========= Define the Condor Submission ========== #
Con = Condor()
Con.ProjectName = "ExampleSubmission"
Con.CondaEnv = "GNN"
Con.AddJob("tttt", tttt, "2GB", "1h")
Con.AddJob("ttbar", ttbar, "2GB", "1h")
Con.AddJob("tttt_Graph", tttt_Graph, "2GB", "1h", waitfor = ["tttt"])
Con.AddJob("ttbar_Graph", ttbar_Graph, "2GB", "1h", waitfor = ["ttbar"])
Con.AddJob("MixedSamples", CombinedSamples, "2GB", "1h", waitfor = ["tttt_Graph", "ttbar_Graph"])
Con.AddJob("GNN_1", Op1, "2GB", "1h", waitfor = ["MixedSamples"])
Con.AddJob("GNN_2", Op2, "2GB", "1h", waitfor = ["MixedSamples"])
#Con.LocalDryRun()
Con.DumpCondorJobs()
