from Functions.Event.Event import EventGenerator
from Functions.IO.IO import PickleObject, UnpickleObject
from Functions.GNN.GNN import Optimizer
from Functions.GNN.Graphs import CreateEventGraph, GenerateDataLoader
from Functions.Plotting.Graphs import GraphPainter

cache = False
events = -1
dir = "/home/tnom6927/Downloads/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root/user.pgadow.24765302._000001.output.root"

def EvaluateTruthTopClassification(Events):
    n_e = 0 
    n_t = 0 
    n_t_c = 0; 
    n_e_c = 0; 
    for i in Events.Events:
        Event = Events.Events[i]["nominal"]
        c=0
        for k in Event.TruthTops:
            n_t+=1
            
            if k.ModelPredicts == k.Signal:
                n_t_c += 1
                c+=1

        if c == 4:
            n_e_c += 1
        n_e+=1
    
    print("Correctly Classified Events (%): ", float(n_e_c/n_e)*100)
    print("Correctly Classified Tops (%): ", float(n_t_c/n_t)*100)

    print("Number of Events: ", n_e)
    print("Number of Tops: ", n_t)

    print("Number of Correct Events: ", n_e_c)
    print("Number of Correct Tops: ", n_t_c)

    return float(n_e_c/n_e)*100, float(n_t_c/n_t)*100


def EvaluationOfGNN(Container, tree):
    n_e = 0
    n_c = 0
    n_c_rc = 0
    n_rc = 0
    for i in Container.Events:
        Event = Container.Events[i][tree]
        
        All = []
        All += Event.RCJets
        All += Event.Electrons
        All += Event.Muons
        All += Event.TruthTops
        All += Event.TruthChildren
        All += Event.Jets
        
        Objects = []
        for k in All:
            try:
                k.ModelPredicts
            except AttributeError:
                continue
            Objects.append(k)
        obj = 0
        for rc in Objects:
            if rc.Signal == rc.ModelPredicts:
                n_c_rc += 1
                obj += 1
            n_rc += 1
        n_e += 1

        if obj == len(Objects):
            n_c += 1

    print("Completely correctly classified Events (%): ", float(n_c/n_e)*100)
    print("Correctly Classified Objects in Event (%): ", float(n_c_rc/n_rc)*100)

    print("Number of Events: ", n_e)
    print("Number of Objects: ", n_rc)




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
    EV.TruthAttribute = {"Signal" : ""}
    EV.TorchDataLoader()
    DataFirst = EV.DataLoader[0]
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
    L.DefaultBatchSize = 100
    L.NodeAttributes = {"Signal" : "Multi"}
    L.TruthAttribute = {"Signal" : ""}
    L.TorchDataLoader()
    
    Op = Optimizer(L)
    Op.DefineEdgeConv(1, 2)
    Op.EpochLoop()
    Op.AssignPredictionToEvents(ev, "nominal")
   
    EvaluateTruthTopClassification(ev)

def Test4TopGNNInvMass():
    
    if cache == True:
        Generate_Cache()
    ev = UnpickleObject("TruthTops")
    
    L = GenerateDataLoader(ev)
    L.DefaultBatchSize = 1000
    L.NodeAttributes = {"pt" : "", "eta" : "", "phi" : "", "e" : "", "M" : "invMass"}
    L.TruthAttribute = {"Signal" : ""}
    L.TorchDataLoader()
    
    Op = Optimizer(L)
    Op.Epochs = 50
    Op.DefineEdgeConv(4, 2)
    Op.EpochLoop()
    Op.AssignPredictionToEvents(ev, "nominal")

    EvaluateTruthTopClassification(ev)


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

def TestRCJetAssignmentGNN():
    signal_dir = "/home/tnom6927/Downloads/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root/user.pgadow.24765302._000001.output.root"
    background_dir = "/home/tnom6927/Downloads/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root/postProcessed_ttW.root"

    Event = -1
    back = EventGenerator(background_dir, DebugThresh = Event)
    back.SpawnEvents()
    back.CompileEvent()

    sig = EventGenerator(signal_dir, DebugThresh = Event)
    sig.SpawnEvents()
    sig.CompileEvent()
   
    PickleObject(sig, "Signal_GNN")
    PickleObject(back, "ttW_GNN") 

    sig = UnpickleObject("Signal_GNN")
    back = UnpickleObject("ttW_GNN")

    sig_L = GenerateDataLoader(sig)
    sig_L.DefaultBatchSize = 1
    sig_L.ParticleLevel = "Detector"
    sig_L.NodeAttributes = {"pt" : "", "eta" : "", "phi" : "", "e" : "", "M" : "invMass"}
    sig_L.TruthAttribute = {"Signal" : ""}
    sig_L.TorchDataLoader("nominal", "signal")

    back_L = GenerateDataLoader(back)
    back_L.DefaultBatchSize = 1
    back_L.ParticleLevel = "Detector"
    back_L.NodeAttributes = {"pt" : "", "eta" : "", "phi" : "", "e" : "", "M" : "invMass"}
    back_L.TruthAttribute = {"Signal" : ""}
    back_L.TorchDataLoader("tree", "background")

    PickleObject(sig_L, "Signal_GNN_L")
    PickleObject(back_L, "ttW_GNN_L") 
    sig_L = UnpickleObject("Signal_GNN_L")
    back_L = UnpickleObject("ttW_GNN_L")


    Samples = [sig_L, back_L] 
    Op = Optimizer()
    Op.SampleHandler(Samples)
    Op.DefineEdgeConv(4, 3)
    Op.KFoldTraining()
    
