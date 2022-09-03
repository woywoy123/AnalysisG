from AnalysisTopGNN.Generators import Analysis 
from AnalysisTopGNN.Submission import Condor
from DelphesEvent import Event
from DelphesEventGraph import EventGraphTruthTopChildren
from NodeFeature import pdgid, PT, Eta, Mass, Energy, Index, Phi
from EdgeFeature import delta_energy, delta_eta, delta_phi, delta_pT, delta_Index
from GraphFeature import SignalSample, nTops
from AnalysisTopGNN.Models.TrivialModels import GraphNN
def pt(a):
        return 0

Dir = "/home/tnom6927/Dokumente/Project/Analysis/bsm4tops-gnn-analysis/DebugGNN/AnalysisTopGNN/tag_1_delphes_events.root"

#Sub = Condor()

Ana = Analysis()
Ana.EventImplementation = Event
Ana.EventGraph = EventGraphTruthTopChildren
Ana.CompileSingleThread = True
Ana.DataCache = True
#Ana.CPUThreads =  5
Ana.EventCache = False
#Ana.__CheckSettings = False 
Ana.Tree = "Delphes"
Ana.FullyConnect = True
Ana.AddNodeFeature("Index", Index)
Ana.AddEdgeFeature("Delta_Index", delta_Index)
#Ana.AddNodeFeature("Eta", Eta)
#Ana.AddNodeFeature("Mass", Mass)
#Ana.AddNodeFeature("E", Energy)
#Ana.AddNodeFeature("PDG", pdgid)
#Ana.AddNodeFeature("Phi", Phi)
#Ana.AddNodeTruth("Index", Index)
#Ana.AddEdgeFeature("Delta_E", delta_energy)
#Ana.AddEdgeFeature( "Delta_PT", delta_pT)
#Ana.AddEdgeFeature("Delta_Phi", delta_phi)
#Ana.AddEdgeFeature("Delta_Eta", delta_eta)
#Ana.AddEdgeTruth("Delta_Index", delta_Index)
#Ana.AddGraphTruth("Sample", SignalSample)
Ana.AddGraphFeature("NTops", nTops) 
Ana.VerboseLevel = 3
Ana.NEvent_Stop = 100
Ana.InputSample("Delphes", Dir)
Ana.Model = GraphNN
Ana.Launch()



