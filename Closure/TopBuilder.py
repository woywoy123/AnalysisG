from Closure.GenericFunctions import CreateEventGeneratorComplete, CreateDataLoaderComplete
from Functions.GNN.Optimizer import Optimizer
from Functions.GNN.TrivialModels import CombinedConv
from Functions.IO.IO import UnpickleObject, PickleObject
import Functions.FeatureTemplates.EdgeFeatures as ef
import Functions.FeatureTemplates.NodeFeatures as nf
import Functions.FeatureTemplates.GraphFeatures as gf

from Functions.Particles.TopBuilder import ParticleReconstructor

def TestBuilder(Files, CreateCache): 
    
    CreateCache = False

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

    top = ParticleReconstructor(op.Model, op.TrainingSample[0]) 
    top.Prediction()
    top.MassFromFeature("NodeSignal")

    return True




