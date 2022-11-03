from AnalysisTopGNN.Generators import Analysis, ModelEvaluator, Optimization
from AnalysisTopGNN.Events import Event, EventGraphTruthTopChildren, EventGraphTruthJetLepton

from Templates.EventFeatureTemplate import ApplyFeatures
from ExampleModel.BasicBaseLine import BasicBaseLineRecursion
from ExampleModel.CheatModel import CheatModel

def TestOptimizer(Files):
    
    Ana = Analysis()
    Ana.ProjectName = "Optimizer"
    Ana.InputSample("SingleTop", Files[1])
    Ana.InputSample("4Tops", Files[0])
    Ana.DataCache = True
    Ana.DumpHDF5 = True
    Ana.chnk = 100
    Ana.EventStop = 1000
    Ana.VerboseLevel = 3
    Ana.Event = Event
    Ana.EventGraph = EventGraphTruthTopChildren
    ApplyFeatures(Ana, "TruthChildren")
    Ana.Launch()
    
    Ana = Analysis()
    Ana.ProjectName = "Optimizer"
    Ana.InputSample("SingleTop")
    Ana.InputSample("4Tops")
    Ana.DataCache = True
    Ana.TrainingSampleName = "Test"
    Ana.TrainingPercentage = 80
    Ana.Launch()
  
    Ana = Analysis()
    Ana.ProjectName = "Optimizer"
    Ana.RunName = "CheatModel"
    Ana.TrainingSampleName = "Test"
    Ana.Model = CheatModel()
    Ana.Tree = "nominal"
    Ana.ContinueTraining = True
    Ana.BatchSize = 10
    Ana.Optimizer = {"ADAM" : { "lr" : 0.001, "weight_decay" : 0.00001 }}
    Ana.Launch()

    Ana = Analysis()
    Ana.ProjectName = "Optimizer"
    Ana.RunName = "BaseLine"
    Ana.TrainingSampleName = "Test"
    Ana.Model = BasicBaseLineRecursion()
    Ana.Tree = "nominal"
    Ana.ContinueTraining = True
    Ana.BatchSize = 50
    Ana.Optimizer = {"ADAM" : { "lr" : 0.0001, "weight_decay" : 0.00001 }}
    Ana.Launch()

    Ana = Analysis()
    Ana.ProjectName = "Optimizer"
    Ana.Tree = "nominal"
    Ana.TrainingSampleName = "Test"
    Ana.PlotNodeStatistics = True
    Ana.PlotTrainingStatistics = True
    Ana.PlotTrainSample = True
    Ana.PlotTestSample = True
    Ana.PlotEntireSample = True
    Ana.EvaluateModel("CheatModel", "./Optimizer/TrainedModels/CheatModel", CheatModel(), 10) 
    Ana.EvaluateModel("BasicBaseLine", "./Optimizer/TrainedModels/BaseLine", BasicBaseLineRecursion(), 50)
    Ana.Launch()

    return True
