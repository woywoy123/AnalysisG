from AnalysisTopGNN.Generators import Analysis
from AnalysisTopGNN.Events import Event, EventGraphTruthTopChildren, EventGraphTruthJetLepton
from Optimization import ModelTrainer
from Templates.EventFeatureTemplate import ApplyFeatures
from ExampleModel.BasicBaseLine import BasicBaseLineRecursion
from ExampleModel.CheatModel import CheatModel

def TestOptimizer(Files):
    
    Ana = Analysis()
    Ana.ProjectName = "Optimizer"
    Ana.InputSample("SingleTop", Files[1])
    Ana.InputSample("4Tops", Files[0])
    Ana.EventCache =  True
    Ana.DumpPickle = True
    Ana.chnk = 100
    Ana.EventStop = 500
    Ana.VerboseLevel = 1
    Ana.Event = Event
    Ana.Launch()

    Ana = Analysis()
    Ana.ProjectName = "Optimizer"
    Ana.InputSample("SingleTop")
    Ana.InputSample("4Tops")
    Ana.DumpHDF5 = True
    Ana.DataCache = False
    Ana.EventStop = None
    Ana.VerboseLevel = 1
    Ana.Threads = 2
    Ana.EventGraph = EventGraphTruthTopChildren
    ApplyFeatures(Ana, "TruthChildren")
    Ana.Launch()
    Ana.GenerateTrainingSample(90)
   
    op = ModelTrainer()
    op.ProjectName = "Optimizer"
    op.RunName = "CheatModel"
    op.Model = CheatModel()
    op.Tree = "nominal"
    op.Device = "cuda"
    op.BatchSize = 10
    op.Optimizer = {"ADAM" : { "lr" : 0.001, "weight_decay" : 0.00001 }}
    op.Scheduler = None #{"ExponentialLR" : {"gamma" : 0.9}}
    op.SplitSampleByNode = False
    op.AddAnalysis(Ana)
    op.Launch()

    op = ModelTrainer()
    op.ProjectName = "Optimizer"
    op.RunName = "BaseLine_01"
    op.Model = BasicBaseLineRecursion()
    op.Tree = "nominal"
    op.Device = "cuda"
    op.BatchSize = 50
    op.Optimizer = {"ADAM" : { "lr" : 0.0001, "weight_decay" : 0.00001 }}
    op.Scheduler = None #{"ExponentialLR" : {"gamma" : 0.9}}
    op.SplitSampleByNode = False
    op.AddAnalysis(Ana)
    op.Launch()
 
    from ModelComparison import ModelComparison

    Mod = ModelComparison()
    Mod.ProjectName = "Optimizer"
    Mod.Tree = "nominal"
    Mod.Device = "cuda"
    Mod.AddAnalysis(Ana)
    Mod.AddModel("CheatModel", "./Optimizer/TrainedModels/CheatModel", CheatModel(), 50) 
    Mod.AddModel("BaseLine", "./Optimizer/TrainedModels/BaseLine_01", BasicBaseLineRecursion(), 50) 
    Mod.Compile()


    return True
