from Functions.GNN.Graphs import GenerateDataLoader
from Functions.GNN.Optimizer import Optimizer
from Functions.IO.IO import UnpickleObject, PickleObject
from Functions.GNN.Metrics import EvaluationMetrics
from Functions.GNN.Models import InvMassGNN
from Functions.Plotting.Histograms import TH1F, CombineHistograms
from Functions.IO.Files import WriteDirectory

def SimpleFourTops():
    def Signal(a):
        return int(a.FromRes)

    def Charge(a):
        return float(a.FromRes)

    ev = UnpickleObject("tttt.pkl")
    Loader = GenerateDataLoader()
    Loader.AddNodeFeature("x", Charge)
    Loader.AddNodeTruth("y", Signal)
    Loader.AddSample(ev, "nominal", "TruthTops")

    op = Optimizer(Loader)
    op.DefaultBatchSize = 5000
    op.Epochs = 10
    op.NotifyTime = 1
    op.kFold = 2
    op.DefineEdgeConv(1, 2)
    op.kFoldTraining()

    return True


def GenerateTemplate(SignalSample = "SignalSample.pkl", Level = "TruthChildren_init", Additional_Samples = "", OutputName = "LoaderSignalSample.pkl", tree = "nominal", device = "cuda"):
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
        if a.Type == "truth_top":
            return int(a.FromRes)
        return int(a.Index)

    def ConnectionTruthJet(a, b):
        
        if a.Type == "truthjet" and b.Type == "truthjet":
            if a.GhostTruthJetMap[0] == 0:
                return False
            if b.GhostTruthJetMap[0] == 0:
                return False
            if len(list(set(b.GhostTruthJetMap).intersection(a.GhostTruthJetMap))) > 0:
                return True
            else:
                return False

        elif a.Type == "truthjet" and b.Type != "truthjet":
            if b.Index == 0:
                return False
            elif b.Index in a.GhostTruthJetMap:
                return True 
            else:
                return False

        elif a.Type != "truthjet" and b.Type == "truthjet":
            if a.Index == 0:
                return False
            elif a.Index in b.GhostTruthJetMap:
                return True 
            else:
                return False

        return a.Index == b.Index

    def ConnectionJet(a, b):
        if a.Type == "jet" and b.Type == "jet":
            if a.JetMapGhost[0] == 0:
                return False
            if b.JetMapGhost[0] == 0:
                return False
            if len(list(set(b.JetMapGhost).intersection(a.JetMapGhost))) > 0:
                return True
            else:
                return False

        elif a.Type == "jet" and b.Type != "jet":
            if b.Index == 0:
                return False
            elif b.Index in a.JetMapGhost:
                return True 
            else:
                return False

        elif a.Type != "jet" and b.Type == "jet":
            if a.Index == 0:
                return False
            elif a.Index in b.JetMapGhost:
                return True 
            else:
                return False

        return a.Index == b.Index

    def Connection(a, b):
        if a.Type == "truth_top":
            return a.Signal == b.Signal == 1
        
        elif a.Type != "truthjet" and b.Type != "truthjet":
            if a.Index != b.Index:
                return False
        return a.Index == b.Index

    def MergedNode(a):
        if a.Type == "truthjet":
            return len(a.GhostTruthJetMap)
        elif a.Type == "jet":
            return len(a.JetMapGhost)
        else:
            return 0

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
    Loader.Device_s = device
    Loader.AddNodeFeature("e", energy)
    Loader.AddNodeFeature("eta", eta)
    Loader.AddNodeFeature("pt", pt)
    Loader.AddNodeFeature("phi", phi)
    Loader.AddEdgeFeature("dr", d_r)
    Loader.AddEdgeFeature("dphi", d_phi)
    Loader.AddEdgeFeature("m", m)
    
    Loader.AddNodeTruth("y", Signal)
    
    if Level == "TruthJetsLep":
        Loader.AddNodeTruth("merged", MergedNode)
        Loader.AddEdgeTruth("edge_y", ConnectionTruthJet)

    elif Level == "JetsLep":
        Loader.AddNodeTruth("merged", MergedNode)
        Loader.AddEdgeTruth("edge_y", ConnectionJet)
    else:
        Loader.AddEdgeTruth("edge_y", Connection)

    
    if SignalSample != "":
        ev = UnpickleObject(SignalSample)
        Loader.AddSample(ev, "nominal", Level)
    
    if Additional_Samples != "" and type(Additional_Samples) != list:
        ev = UnpickleObject(Additional_Sample)
        Loader.AddSample(ev, tree, Level)
    else:
        for i in Additional_Samples:
            ev = UnpickleObject(i)
            Loader.AddSample(ev, tree, Level)
    
    Loader.MakeTrainingSample()
    PickleObject(Loader, OutputName)


def TrainEvaluate(Model, Outdir, Input = "LoaderSignalSample.pkl", ClusterOutputDir = ""):
    Loader = UnpickleObject(Input)
    
    op = Optimizer(Loader)
    op.TrainingName = Outdir
    if ClusterOutputDir != "":
        op.ModelOutdir = ClusterOutputDir
    op.DefaultBatchSize = 10
    op.Epochs = 100
    op.kFold = 10
    op.LearningRate = 1e-6
    op.WeightDecay = 1e-6
    op.MinimumEvents = 100

    if Model == "InvMassNode":
        op.DefineInvMass(4, Target = "Nodes")

    if Model == "InvMassNodeEdge":
        op.DefineInvMass(128, Target = "NodeEdges")

    if Model == "InvMassEdge":
        op.DefineInvMass(128, Target = "Edges")

    if Model == "PathNetNode":
        op.DefinePathNet(Target = "Nodes")

    if Model == "PathNetNodeEdge":
        op.DefinePathNet(Target = "NodeEdges")

    if Model == "PathNetEdge":
        op.DefinePathNet(complex = 128, out = 128, Target = "Edges")

    op.kFoldTraining()
    #op.LoadModelState()
    if ClusterOutputDir != "":
        PickleObject(op, ClusterOutputDir + "/" + Outdir + "/Optimizer.pkl")
        return 

    eva = EvaluationMetrics()
    eva.Sample = op
    eva.LossTrainingPlot("Plots/" + Outdir, False)
    
    Map_Truth = op.RebuildTruthParticles()
    Map_Pred = op.RebuildPredictionParticles()
    #FileMap = op.RebuildEventFileMapping()
    #for i in Map_Truth:
    #    print(len(Map_Truth[i]), len(op.Loader.EventMap[i].Particles), FileMap[i], i) #, len(Map_Pred[i]))
    
    tr = TH1F() 
    tr.Title = "Reconstruction Delta"
    total = 0
    total_r = 0
    for i in Map_Pred:
        if len(Map_Pred[i]) == 0:
            continue
        reco = len(Map_Pred[i])
        truth = len(Map_Truth[i])
        delta = truth - reco
        tr.xData.append(delta)
        total += truth
        total_r += reco

    tr.xTitle = "Missing Number of Reconstructed Tops"
    tr.yTitle = "Events"
    tr.DefaultDPI = 500
    tr.xBins = 10
    tr.xMin = -5
    tr.xMax = 5
    tr.Filename = "RecoDelta.png"
    tr.SaveFigure("Plots/" + Outdir)
    
    if total != 0:
        Output = ""
        Output += "Top Reconstruction Efficiency: " + str(round(100*total_r / total, 4)) + "%\n"
        Output += "Total Truth Tops: " + str(total) + "\n"
        Output += "Total Reco Tops: " + str(total_r) + "\n"
        WriteDirectory().WriteTextFile(Output, "Plots/" + Outdir, "ReconstructionInformation.txt")

    t = TH1F() 
    for i in Map_Truth:
        Part = Map_Truth[i]
        for j in Part:
            t.xData.append(j.Mass_GeV)
    
    t.Title = "Truth Graph"
    t.xTitle = "Mass (GeV)"
    t.yTitle = "Entries"
    t.DefaultDPI = 500
    t.xBins = 250
    t.xMin = 0
    t.xMax = 500
    t.Filename = "TruthGraph_Tops.png"
    t.SaveFigure("Plots/" + Outdir)


    tc = TH1F() 
    for i in Map_Pred:
        Part = Map_Pred[i]
        for j in Part:
            tc.xData.append(j.Mass_GeV)
    
    tc.Title = "Reconstruction (GNN)"
    tc.xTitle = "Mass (GeV)"
    tc.yTitle = "Entries"
    tc.DefaultDPI = 500
    tc.xBins = 250
    tc.xMin = 0
    tc.xMax = 500
    tc.Filename = "GNN_Tops.png"
    tc.SaveFigure("Plots/" + Outdir)


    Mass_Edge = CombineHistograms()
    Mass_Edge.DefaultDPI = 500
    Mass_Edge.Histograms = [t, tc]
    Mass_Edge.Title = "Reconstructed Mass of Top (GNN) and Truth"
    Mass_Edge.Filename = "TopMass.png"
    Mass_Edge.Save("Plots/" + Outdir)


def TestInvMassGNN_Tops_Edge():
    TrainEvaluate("InvMassEdge", "GNN_Performance_InvMassGNN_Tops_Edge")
    return True

def TestInvMassGNN_Tops_Node():
    TrainEvaluate("InvMassNode", "GNN_Performance_InvMassGNN_Tops_Node")
    return True

def TestInvMassGNN_Children_Edge():
    TrainEvaluate("InvMassEdge", "GNN_Performance_InvMassGNN_Children_Edge")
    return True

def TestInvMassGNN_Children_Node():
    TrainEvaluate("InvMassNode", "GNN_Performance_InvMassGNN_Children_Node")
    return True

def TestInvMassGNN_TruthJets():
    TrainEvaluate("InvMassEdge", "GNN_Performance_InvMassGNN_TruthJet_Edge")
    return True

def TestPathNetGNN_Tops_Edge():
    TrainEvaluate("PathNetEdge", "GNN_Performance_PathNetGNN_Tops_Edge")
    return True

def TestPathNetGNN_Children_Edge():
    TrainEvaluate("PathNetNodeEdge", "GNN_Performance_PathNetGNN_Children_Edge")
    return True

def TestPathNetGNN_Children_Node():
    TrainEvaluate("PathNetNode", "GNN_Performance_PathNetGNN_Children_Node")
    return True

def TestPathNetGNN_TruthJets():
    TrainEvaluate("PathNetEdge", "GNN_Performance_PathNetGNN_TruthJet_Edge")
    return True

