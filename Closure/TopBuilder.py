from Closure.GenericFunctions import CreateEventGeneratorComplete, CreateDataLoaderComplete
from Functions.GNN.Optimizer import Optimizer
from Functions.GNN.TrivialModels import CombinedConv
from Functions.IO.IO import UnpickleObject, PickleObject
import Functions.FeatureTemplates.EdgeFeatures as ef
import Functions.FeatureTemplates.NodeFeatures as nf
import Functions.FeatureTemplates.GraphFeatures as gf

from Functions.Particles.Particles import Particle

from Functions.Particles.TopBuilder import ParticleReconstructor

def TestBuilder(Files, CreateCache): 
    
    CreateCache = True

    it = 10
    EV = CreateEventGeneratorComplete(it, Files, ["D_" + str(i) for i in range(len(Files))], CreateCache, "TestBuilder")
    DL = CreateDataLoaderComplete(["D_" + str(i) for i in range(len(Files))], "TruthTopChildren", "TestBuilderData", CreateCache, "TestBuilder")
    DL.MakeTrainingSample(0)
    
    if CreateCache:
        op = Optimizer(DL)
        op.VerboseLevel = 1
        op.Model = CombinedConv()
        op.RunName = "TestBuilder"
        op.RunDir = "_Pickle/TestBuilder"
        op.KFoldTraining()
        PickleObject(op, "Debug.pkl")
    
    op = UnpickleObject("Debug.pkl")
    
    def ParticleAggre(sample, attr):
        P = {}
        sample = op.TrainingSample[0]
        for i in EV[0].Events[int(sample.i)]["nominal"].TopPostFSRChildren:
            if i.__dict__[attr] not in P:
                P[i.__dict__[attr]] = Particle()
            P[i.__dict__[attr]].Decay_init.append(i)

        Res_T = []
        for i in P:
            P[i].CalculateMassFromChildren()
            Res_T.append(round(P[i].Mass_init_GeV,2))
        return Res_T
   

    sample = op.TrainingSample[0]
    
    Res_T = ParticleAggre(sample, "FromRes")
    top = ParticleReconstructor(op.Model, sample) 
    top.VerboseLevel = 0
    top.TruthMode = True
    top.Prediction()
    res = [round(i[0], 2) for i in top.MassFromNodeFeature("N_T_NodeSignal").tolist()]
    
    res.sort()
    Res_T.sort()
    
    print(res, Res_T)
    assert res == Res_T

    # Reconstruct the tops
    Res_T = ParticleAggre(sample, "Index")   
    top = ParticleReconstructor(op.Model, sample) 
    top.VerboseLevel = 0
    top.TruthMode = True
    top.Prediction()
    res = [round(i[0], 2) for i in top.MassFromNodeFeature("N_T_Index").tolist()]

    res.sort()
    Res_T.sort()
    print(res, Res_T)   
    assert res == Res_T


    top = ParticleReconstructor(op.Model, sample) 
    top.TruthMode = True
    top.Prediction()
    res = [round(i[0], 2) for i in top.MassFromFeatureEdges("E_T_Topology").tolist()]
    
    res.sort()
    Res_T.sort()

    print(res, Res_T)   
    assert res == Res_T

    top = ParticleReconstructor(op.Model, sample) 
    top.TruthMode = True
    top.Prediction()
    res = [round(i[0], 2) for i in top.MassFromFeatureEdges("E_signal").tolist()]
    
    res.sort()
    Res_T.sort()

    print(res, Res_T)   
    assert res == Res_T





    return True




