from Functions.Event.Event import EventGenerator
from Functions.IO.IO import PickleObject, UnpickleObject
from Functions.GNN.GNN import Optimizer
from Functions.GNN.Graphs import CreateEventGraph, GenerateDataLoader
from Functions.Plotting.Graphs import GraphPainter

cache = False
events = -1
dir = "/home/tnom6927/Downloads/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root/user.pgadow.24765302._000001.output.root"

def Generate_Cache():
    def Generator(dir, events, compiler):
        ev = EventGenerator(dir, DebugThresh = events)
        ev.SpawnEvents()
        ev.CompileEvent(SingleThread = False, particle = compiler)
        if compiler == False:
            compiler = "Complete"
        PickleObject(ev, compiler)

    compiler = "TruthTops"
    Generator(dir, events, compiler)
    compiler = "TruthChildren"
    Generator(dir, events, compiler)
    compiler = "TruthJets"
    Generator(dir, events, compiler)
    compiler = "Detector"
    Generator(dir, events, compiler)
    compiler = False
    Generator(dir, events, compiler)

def TestGraphObjects():

    if cache == True:
        Generate_Cache()

    ev = UnpickleObject("TruthTops").Events

    # Draw the first event topology
    first_event = ev[0]["nominal"]
    tops = first_event.TruthTops

    Nodes = ["pt", "eta", "phi"]
    EG = CreateEventGraph(tops)
    EG.NodeAttributes = Nodes
    EG.CreateParticleNodes()
    EG.CreateParticlesEdgesAll()
    EG.CreateDefaultEdgeWeights()
    EG.CalculateNodeDifference()
    D = EG.ConvertToData()
    
    assert len(D["d_eta"]) == 12
    assert len(D["d_phi"]) == 12
    assert len(D["d_pt"]) == 12

    print("Passed, Edge Delta Attributes -> ", D["d_eta"])
    print("Passed, Edge Delta Attributes -> ", D["d_phi"])
    print("Passed, Edge Delta Attributes -> ", D["d_pt"]) 

    GD = GraphPainter(EG)
    GD.Title = "FirstEventGraph"
    GD.DrawAndSave("Plots/GraphTest/")
 
    # Now create the truth graph of the above 
    ETG = CreateEventGraph(tops)
    ETG.NodeAttributes = ["Signal"]
    ETG.CreateParticleNodes()
    ETG.CreateParticlesEdgesAll() 
    ETG.CalculateNodeMultiplication()
    D = EG.ConvertToData()

    GT = GraphPainter(ETG)
    GT.Title = "FirstEventTruthGraph"
    GT.DrawAttribute = "d_Signal"
    GT.DrawAndSave("Plots/GraphTest/")

def TestDataImport(Compiler = "TruthTops"):
    if cache == True:
       Generate_Cache()   
    ev = UnpickleObject(Compiler)
    EV = GenerateDataLoader(ev)
    EV.NodeAttributes = {"Index" : "Diff", "Signal" : "Multi", "dR" : "dR"} 
    EV.TorchDataLoader()
    DataFirst = EV.Loader[0]
    DataFirstEdgeAttr = DataFirst.EdgeAttributes
    
    GD = GraphPainter(DataFirst)
    for i in DataFirstEdgeAttr:
        GD.Title = Compiler + "->FirstEventGraph: " + i
        GD.DrawAttribute = i
        GD.DrawAndSave("Plots/GraphDataImportTest/")

# Closure test for the GNN being implemented in this codebase
def TestSimple4TopGNN():
    
    if cache == True:
        Generate_Cache()
    ev = UnpickleObject("TruthTops")
    L = GenerateDataLoader(ev)
    L.NodeAttributes = {"Signal" : "Multi"}
    L.TruthAttribute = {"Signal" : ""}
    L.TorchDataLoader()
    
    Op = Optimizer()
    Op.DataLoader = L.DataLoader
    Op.DefineEdgeConv(1, 2)
    Op.EpochLoop()

def Test4TopGNNInvMass():
    
    if cache == True:
        Generate_Cache()
    ev = UnpickleObject("TruthTops")
    L = GenerateDataLoader(ev)
    L.NodeAttributes = {"pt" : "", "eta" : "", "phi" : "", "e" : "", "M" : "invMass"}
    L.TruthAttribute = {"Signal" : ""}
    L.TorchDataLoader()
    
    Op = Optimizer()
    Op.DataLoader = L.DataLoader
    Op.DefineEdgeConv(4, 2)
    Op.EpochLoop()

def TestComplex4TopGNN():
    
    if cache == True:
        Generate_Cache()
    ev = UnpickleObject("TruthTops")
    L = GenerateDataLoader(ev)
    L.DefaultBatchSize = 20000
    L.NodeAttributes = {"pt" : "", "eta" : "", "phi" : ""}
    L.TruthAttribute = {"Signal" : ""}
    L.TorchDataLoader()
    
    Op = Optimizer()
    Op.Epochs = 100
    Op.DataLoader = L.DataLoader
    Op.DefineEdgeConv(3, 24)
    Op.EpochLoop()
    
