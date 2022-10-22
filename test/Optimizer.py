from AnalysisTopGNN.Generators import Analysis
from AnalysisTopGNN.Events import Event, EventGraphTruthTopChildren
from Optimization import ModelTrainer
from Templates.EventFeatureTemplate import ApplyFeatures
from ExampleModel.BasicBaseLine import BasicBaseLineRecursion









def TestOptimizer(Files):
    
    #Ana = Analysis()
    #Ana.ProjectName = "Optimizer"
    #Ana.InputSample("SingleTop", Files[1])
    #Ana.InputSample("4Tops", Files[0])
    #Ana.EventCache =  True
    #Ana.DumpPickle = True 
    #Ana.EventStop = None
    #Ana.VerboseLevel = 1
    #Ana.Event = Event
    #Ana.Launch()


    Ana = Analysis()
    Ana.ProjectName = "Optimizer"
    Ana.InputSample("SingleTop")
    Ana.InputSample("4Tops")
    Ana.DumpHDF5 = True
    Ana.DataCache = False
    Ana.EventStop = 1000
    Ana.VerboseLevel = 1
    Ana.EventStop = None
    Ana.Threads = 4
    Ana.EventGraph = EventGraphTruthTopChildren
    ApplyFeatures(Ana, "TruthChildren")
    Ana.Launch()
    Ana.GenerateTrainingSample(90)
   
    print(len(Ana))


    op = ModelTrainer()
    op.Model = BasicBaseLineRecursion()
    op.Tree = "nominal"
    op.Device = "cuda"
    op.Optimizer = {"ADAM" : { "lr" : 0.01, "weight_decay" : 0.0001 }}
    op.SplitSampleByNode = True
    op.AddAnalysis(Ana)
    op.Launch()




    return True
