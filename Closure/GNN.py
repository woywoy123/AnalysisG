from Functions.GNN.Graphs import GenerateDataLoader
from Functions.GNN.Optimizer import Optimizer
from Functions.IO.IO import UnpickleObject, PickleObject
from Functions.GNN.Metrics import EvaluationMetrics
from Functions.GNN.Models import InvMassGNN, InvMassAggr

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


def GenerateTemplate(Tree = "TruthChildren_init", Additional_Sample = ""):
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
    def m(a, b):
        t_i = LorentzVector()
        t_i.setptetaphie(a.pt, a.eta, a.phi, a.e)

        t_j = LorentzVector()
        t_j.setptetaphie(b.pt, b.eta, b.phi, b.e)

        T = t_i + t_j
        return float(T.mass)

    ev = UnpickleObject("SignalSample.pkl")
    Loader = GenerateDataLoader()
    Loader.AddNodeFeature("e", energy)
    Loader.AddNodeFeature("eta", eta)
    Loader.AddNodeFeature("pt", pt)
    Loader.AddNodeFeature("phi", phi)
    Loader.AddEdgeFeature("dr", d_r)
    Loader.AddEdgeFeature("m", m)
    Loader.AddNodeTruth("y", Signal)

    Loader.AddSample(ev, "nominal", Tree)
    
    if Additional_Sample != "":
        ev = UnpickleObject(Additional_Sample)
        Loader.AddSample(ev, "tree", "JetLepton")

    Loader.ToDataLoader()
    PickleObject(Loader, "LoaderSignalSample.pkl")


def TrainEvaluate(Model, Outdir):
    Loader = UnpickleObject("LoaderSignalSample.pkl")
    
    op = Optimizer(Loader)
    op.DefaultBatchSize = 25
    op.Epochs = 10
    op.kFold = 10
    op.LearningRate = 1e-5
    op.WeightDecay = 1e-3
    op.MinimumEvents = 250
    if Model == "InvMass":
        op.DefineInvMass(4)
    if Model == "PathNet":
        op.DefinePathNet()
    op.kFoldTraining()

    eva = EvaluationMetrics()
    eva.Sample = op
    eva.AddTruthAttribute("Signal")
    eva.AddPredictionAttribute("y")
    eva.ProcessSample()
    eva.LossTrainingPlot("Plots/" + Outdir, False)

def TestInvMassGNN_Children():
    #GenerateTemplate()
    TrainEvaluate("InvMass", "GNN_Performance_Plots")
    return True

def TestInvMassAggrGNN_Children():
    #GenerateTemplate()
    M = InvMassAggr(4)
    TrainEvaluate(M, "GNN_Performance_Plots")

    return True

def TestPathNetGNN_Children():
    #GenerateTemplate()
    TrainEvaluate("PathNet", "PathNet_Performance_Plots_Children")
    return True

def TestPathNetGNN_Data():
    GenerateTemplate("JetLepton")
    TrainEvaluate("PathNet", "PathNet_Performance_Plots_Data")
    return True


