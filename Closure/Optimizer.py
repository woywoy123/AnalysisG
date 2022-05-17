from Functions.IO.IO import PickleObject, UnpickleObject
from Functions.Event.DataLoader import GenerateDataLoader
import Functions.FeatureTemplates.EdgeFeatures as ef
import Functions.FeatureTemplates.NodeFeatures as nf
import Functions.FeatureTemplates.GraphFeatures as gf
from Functions.GNN.Optimizer import Optimizer
from Functions.GNN.TrivialModels import GraphNN, NodeConv, EdgeConv, CombinedConv

def TestOptimizerGraph(Files, Level, Name, CreateCache):
    
    if CreateCache:
        DL = GenerateDataLoader()
        DL.AddGraphFeature("Signal", gf.Resonance)
        DL.AddGraphTruth("Signal", gf.Resonance)
        DL.SetDevice("cuda")
        for i in Files:
            ev = UnpickleObject(i + "/" + i)
            DL.AddSample(ev, "nominal", Level)
        DL.MakeTrainingSample(0)
        PickleObject(DL, "TestOptimizerGraph")
    DL = UnpickleObject("TestOptimizerGraph")


    op = Optimizer(DL)
    op.RunName = Name
    op.Epochs = 10
    op.kFold = 2
    op.Model = GraphNN()
    op.DefineOptimizer()
    op.KFoldTraining()

    return True

def TestOptimizerNode(Files, Level, Name, CreateCache):

    if CreateCache: 
        DL = GenerateDataLoader()
        DL.AddNodeFeature("x", nf.Signal)
        DL.AddNodeFeature("Sig", nf.Signal)
        DL.AddNodeTruth("x", nf.Signal)
        DL.SetDevice("cuda")
        for i in Files:
            ev = UnpickleObject(i + "/" + i)
            DL.AddSample(ev, "nominal", Level, True, False)
        DL.MakeTrainingSample(0)
        PickleObject(DL, Name)
    DL = UnpickleObject(Name)

    op = Optimizer(DL)
    op.BatchSize = 1
    op.kFold = 4
    op.RunName = Name
    op.Model = NodeConv(2, 2)
    op.KFoldTraining()

    return True

def TestOptimizerEdge(Files, Level, Name, CreateCache):

    if CreateCache: 
        DL = GenerateDataLoader()
        DL.AddEdgeFeature("x", ef.Signal)
        DL.AddEdgeTruth("x", ef.Signal)
        DL.SetDevice("cuda")
        for i in Files:
            ev = UnpickleObject(i + "/" + i)
            DL.AddSample(ev, "nominal", Level, False, True)
            break
        DL.MakeTrainingSample(0)
        PickleObject(DL, Name)
    DL = UnpickleObject(Name)

    op = Optimizer(DL)
    op.BatchSize = 10
    op.kFold = 100
    op.Epochs= 100
    op.RunName = Name
    op.Model = EdgeConv(1, 1)
    op.KFoldTraining()

    return True

def TestOptimizerCombined(Files, Level, Name, CreateCache):

    if CreateCache: 
        DL = GenerateDataLoader()

        # Edge Features 
        DL.AddEdgeFeature("dr", ef.d_r)
        DL.AddEdgeFeature("mass", ef.mass)       
        DL.AddEdgeFeature("signal", ef.Signal)
 
        # Node Features 
        DL.AddNodeFeature("eta", nf.eta)
        DL.AddNodeFeature("pt", nf.pt)       
        DL.AddNodeFeature("phi", nf.phi)      
        DL.AddNodeFeature("energy", nf.energy)
        DL.AddNodeFeature("signal", nf.Signal)
        
        # Graph Features 
        DL.AddGraphFeature("mu", gf.Mu)
        DL.AddGraphFeature("m_phi", gf.MissingPhi)       
        DL.AddGraphFeature("m_et", gf.MissingET)      
        DL.AddGraphFeature("signal", gf.Signal)       


        # Truth Stuff 
        DL.AddEdgeTruth("Topology", ef.Signal)
        DL.AddNodeTruth("NodeSignal", nf.Signal)
        DL.AddGraphTruth("GraphMuActual", gf.MuActual)
        DL.AddGraphTruth("GraphEt", gf.MissingET)
        DL.AddGraphTruth("GraphPhi", gf.MissingPhi)

        DL.SetDevice("cuda")
        for i in Files:
            ev = UnpickleObject(i + "/" + i)
            DL.AddSample(ev, "nominal", Level, True, True)
            break
        DL.MakeTrainingSample(0)
        PickleObject(DL, Name)
    DL = UnpickleObject(Name)

    op = Optimizer(DL)
    op.BatchSize = 1
    op.kFold = 100
    op.Epochs= 100
    op.RunName = Name
    op.Model = CombinedConv()
    op.KFoldTraining()

    return True
