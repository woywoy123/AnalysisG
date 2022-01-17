from Functions.GNN.Graphs import GenerateDataLoader
from Functions.GNN.Optimizer import Optimizer
from Functions.IO.IO import UnpickleObject, PickleObject
from Functions.GNN.Metrics import EvaluationMetrics
from Functions.GNN.Models import InvMassGNN

def SimpleFourTops():
    def Signal(a):
        return int(a.Signal)

    def Charge(a):
        return float(a.Signal)


    ev = UnpickleObject("SignalSample.pkl")
    Loader = GenerateDataLoader()
    Loader.AddNodeFeature("x", Charge)
    Loader.AddNodeTruth("y", Signal)
    Loader.AddSample(ev, "nominal", "TruthTops")
    Loader.ToDataLoader()

    Sig = GenerateDataLoader()
    Sig.AddNodeFeature("x", Charge)
    Sig.AddSample(ev, "nominal", "TruthTops")

    op = Optimizer(Loader)
    op.DefaultBatchSize = 1
    op.Epochs = 10
    op.NotifyTime = 1
    op.kFold = 3
    op.DefineEdgeConv(1, 2)
    op.kFoldTraining()
    op.ApplyToDataSample(Sig, "Sig")    
    
    for i in Sig.DataLoader:
        for n_p in Sig.EventData[i]:
            p_v = n_p.NodeParticleMap
            for t in p_v:
                p = p_v[t] 
                try:
                    assert p.Sig == p.Signal
                except AssertionError:
                    return False

    return True


def GenerateTemplate(SignalSample = "SignalSample.pkl", Tree = "TruthChildren_init", Additional_Samples = "", OutputName = "LoaderSignalSample.pkl"):
    from skhep.math.vectors import LorentzVector

    def eta(a):
        return float(a.eta)
    def energy(a):
        return float(a.e)
    def pt(a):
        return float(a.pt)
    def phi(a):
        return float(a.phi)
    def Signal(a):
        return int(a.Index)
    def d_r(a, b):
        return float(a.DeltaR(b))
    def d_phi(a, b):
        return abs(a.phi - b.phi)
    def m(a, b):
        t_i = LorentzVector()
        t_i.setptetaphie(a.pt, a.eta, a.phi, a.e)

        t_j = LorentzVector()
        t_j.setptetaphie(b.pt, b.eta, b.phi, b.e)

        T = t_i + t_j
        return float(T.mass)

    Loader = GenerateDataLoader()
    Loader.AddNodeFeature("e", energy)
    Loader.AddNodeFeature("eta", eta)
    Loader.AddNodeFeature("pt", pt)
    Loader.AddNodeFeature("phi", phi)
    Loader.AddEdgeFeature("dr", d_r)
    Loader.AddEdgeFeature("dphi", d_phi)
    Loader.AddEdgeFeature("m", m)
    Loader.AddNodeTruth("y", Signal)
    
    if SignalSample != "":
        ev = UnpickleObject(SignalSample)
        Loader.AddSample(ev, "nominal", Tree)
    
    if Additional_Samples != "" and type(Additional_Samples) != list:
        ev = UnpickleObject(Additional_Sample)
        Loader.AddSample(ev, "tree", "JetLepton")
    else:
        for i in Additional_Samples:
            ev = UnpickleObject(i)
            Loader.AddSample(ev, "tree", "JetLepton")

    Loader.ToDataLoader()
    PickleObject(Loader, OutputName)


def TrainEvaluate(Model, Outdir):
    Loader = UnpickleObject("LoaderSignalSample.pkl")
    
    op = Optimizer(Loader)
    op.DefaultBatchSize = 25
    op.Epochs = 20
    op.kFold = 10
    op.LearningRate = 1e-5
    op.WeightDecay = 1e-3
    op.MinimumEvents = 250

    if Model == "InvMassNode":
        op.DefineInvMass(4, Target = "Nodes")
    if Model == "InvMassEdge":
        op.DefineInvMass(64, Target = "Edges")

    if Model == "PathNetNode":
        op.DefinePathNet(Target = "Nodes")

    if Model == "PathNetEdge":
        op.DefinePathNet(Target = "Edges")

    op.kFoldTraining()

    eva = EvaluationMetrics()
    eva.Sample = op
    eva.LossTrainingPlot("Plots/" + Outdir, False)

def TestInvMassGNN_Tops_Edge():
    GenerateTemplate(Tree = "TruthTops")
    TrainEvaluate("InvMassEdge", "GNN_Performance_Plots/InvMassGNN_Tops_Edge")
    return True

def TestInvMassGNN_Tops_Node():
    #GenerateTemplate(Tree = "TruthTops")
    TrainEvaluate("InvMassNode", "GNN_Performance_Plots/InvMassGNN_Tops_Node")
    return True

def TestInvMassGNN_Children_Edge():
    GenerateTemplate(Tree = "TruthChildren_init")
    TrainEvaluate("InvMassEdge", "GNN_Performance_Plots/InvMassGNN_Children_Edge")
    return True

def TestInvMassGNN_Children_Node():
    #GenerateTemplate(Tree = "TruthChildren_init")
    TrainEvaluate("InvMassNode", "GNN_Performance_Plots/InvMassGNN_Children_Node")
    return True

def TestPathNetGNN_Children_Edge():
    #GenerateTemplate("TruthChildren_init")
    TrainEvaluate("PathNetEdge", "GNN_Performance_Plots/PathNetGNN_Children_Edge")
    return True

def TestPathNetGNN_Children_Node():
    #GenerateTemplate("TruthChildren_init")
    TrainEvaluate("PathNetNode", "GNN_Performance_Plots/PathNetGNN_Children_Node")
    return True

def TestPathNetGNN_Data():
    GenerateTemplate("JetLepton")
    TrainEvaluate("PathNetEdge", "GNN_Performance_Plots/PathNetGNN_Data_Edge")
    return True


