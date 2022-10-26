from AnalysisTopGNN.Generators import Analysis
from AnalysisTopGNN.Events import Event, EventGraphTruthTopChildren, EventGraphTruthJetLepton
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
    #Ana.chnk = 100
    #Ana.VerboseLevel = 1
    #Ana.Event = Event
    #Ana.Launch()

    Ana = Analysis()
    Ana.ProjectName = "Optimizer"
    Ana.InputSample("SingleTop")
    Ana.InputSample("4Tops")
    Ana.DumpHDF5 = True
    Ana.DataCache = False
    Ana.EventStop = None
    Ana.VerboseLevel = 1
    Ana.Threads = 2
    Ana.EventGraph = EventGraphTruthJetLepton
    ApplyFeatures(Ana, "TruthJets")
    Ana.Launch()
    Ana.GenerateTrainingSample(90)
   
    op = ModelTrainer()
    op.ProjectName = "Optimizer"
    op.RunName = "BaseLine"
    op.Model = BasicBaseLineRecursion()
    op.Tree = "nominal"
    op.Device = "cuda"
    op.BatchSize = 50
    op.Optimizer = {"ADAM" : { "lr" : 0.001, "weight_decay" : 0.0001 }}
    op.Scheduler = {"ExponentialLR" : {"gamma" : 0.9}}
    op.SplitSampleByNode = False
    op.AddAnalysis(Ana)
    op.Launch()
    
    return True
