from GenericFunctions import CreateEventGeneratorComplete, CreateDataLoaderComplete
from AnalysisTopGNN.Generators import Optimizer
from TrivialModels import CombinedConv
from AnalysisTopGNN.IO import UnpickleObject, PickleObject
from AnalysisTopGNN.Particles.Particles import Particle 
from AnalysisTopGNN.Reconstruction import Reconstructor

def TestBuilder(Files, CreateCache): 
    
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
        for i in EV[0].Events[int(sample.i)]["nominal"].TopPostFSRChildren:
            if i.__dict__[attr] not in P:
                P[i.__dict__[attr]] = Particle()
            P[i.__dict__[attr]].Children.append(i)

        Res_T = []
        for i in P:
            P[i].CalculateMass(P[i].Children)
            Res_T.append(round(P[i].Mass_GeV,2))
        return Res_T
   

    sample = op.TrainingSample[0]
    
    Res_T = ParticleAggre(sample, "FromRes")
    top = Reconstructor(op.Model, sample) 
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
    top = Reconstructor(op.Model, sample) 
    top.VerboseLevel = 0
    top.TruthMode = True
    top.Prediction()
    res = [round(i[0], 2) for i in top.MassFromNodeFeature("N_T_Index").tolist()]

    res.sort()
    Res_T.sort()
    print(res, Res_T)   
    assert res == Res_T


    top = Reconstructor(op.Model, sample) 
    top.TruthMode = True
    top.Prediction()
    res = [round(i[0], 2) for i in top.MassFromFeatureEdges("E_T_Topology").tolist()]
    
    res.sort()
    Res_T.sort()

    print(res, Res_T)   
    assert res == Res_T

    top = Reconstructor(op.Model, sample) 
    top.TruthMode = True
    top.Prediction()
    res = [round(i[0], 2) for i in top.MassFromFeatureEdges("E_signal").tolist()]
    
    res.sort()
    Res_T.sort()

    print(res, Res_T)   
    assert res == Res_T





    return True




