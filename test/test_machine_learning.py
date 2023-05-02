from AnalysisG.Generators import RandomSamplers
from AnalysisG import Analysis
from AnalysisG.Events import Event, GraphChildren
from AnalysisG.Templates import ApplyFeatures
from AnalysisG.Templates import FeatureAnalysis
from AnalysisG.Generators import Optimizer

root1 = "./samples/sample1/smpl1.root"

def test_random_sampling():
    Ana = Analysis()
    Ana.InputSample(None, root1)
    Ana.Event = Event
    Ana.EventGraph = GraphChildren
    ApplyFeatures(Ana, "TruthChildren")
    Ana.Launch
   
    smpls = Ana.todict
    
    r = RandomSamplers()
    x = r.RandomizeEvents(smpls, 74)
    assert len(x) == len(smpls)
    
    x = r.MakeTrainingSample(smpls) 

    for i in x["train_hashes"]: assert "train" == smpls[i].TrainMode
    for i in x["test_hashes"]: assert "test" == smpls[i].TrainMode
    
    x = r.MakekFolds(smpls, 10)  
    train = {"k-" + str(i+1) : [] for i in range(10)}
    for f in x:
        for smpl in x[f]["train"]: train[f] += [smpl.hash]
        for smpl in x[f]["leave-out"]: assert smpl.hash not in train[f] 
    
    x = r.MakeDataLoader(smpls, SortByNodes = True)
    assert 12 in x
    assert 13 in x
    assert 14 in x

    x = r.MakeDataLoader(smpls)
    assert "all" in x
    assert len(x["all"]) == len(Ana)

def test_feature_analysis():
    Ana = Analysis()
    Ana.InputSample(None, root1)
    Ana.Event = Event
    Ana.EventGraph = GraphChildren 
    ApplyFeatures(Ana, "TruthChildren")
    Ana.nEvents = 10
    Ana.TestFeatures = True
    assert Ana.Launch

    def fx(a): return a.NotAFeature

    Ana = Analysis()
    Ana.InputSample(None, root1)
    Ana.Event = Event
    Ana.EventGraph = GraphChildren 
    ApplyFeatures(Ana, "TruthChildren")
    Ana.nEvents = 10
    Ana.TestFeatures = True
    Ana.AddGraphTruth(fx, "NotAFeature")
    assert Ana.Launch == False

def test_optimizer():
    from models.CheatModel import CheatModel
    Ana = Analysis()
    Ana.InputSample(None, root1)
    Ana.Event = Event
    Ana.ProjectName = "Project"
    Ana.EventGraph = GraphChildren 
    ApplyFeatures(Ana, "TruthChildren")
    Ana.EventStop = 50
    Ana.DataCache = True
    Ana.PurgeCache = True
    Ana.kFolds = 10
    Ana.Launch

    op = Optimizer(Ana)
    op.Model = CheatModel
    op.Device = "cuda"
    op.Optimizer = "ADAM"
    op.ContinueTraining = False
    op.DebugMode = False
    op.EnableReconstruction = True
    op.BatchSize = 2
    op.Launch
   
    op.rm("Project") 

def test_optimizer_analysis():
    from models.CheatModel import CheatModel
    Ana = Analysis()
    Ana.InputSample(None, root1)
    Ana.ProjectName = "Project"
    Ana.Event = Event
    Ana.EventGraph = GraphChildren
    ApplyFeatures(Ana, "TruthChildren")
    Ana.DataCache = True 
    Ana.kFolds = 2
    Ana.kFold = 1
    Ana.Epochs = 20
    Ana.Optimizer = "ADAM"
    Ana.RunName = "RUN"
    Ana.DebugMode = True
    Ana.OptimizerParams = {"lr" : 0.001}
    Ana.Scheduler = "ExponentialLR"
    Ana.ContinueTraining = False
    Ana.SchedulerParams = {"gamma" : 1}
    Ana.Device = "cuda"
    Ana.Model = CheatModel
    Ana.EnableReconstruction = True 
    Ana.BatchSize = 1
    Ana.Launch

if __name__ == "__main__":
    #test_random_sampling()
    #test_feature_analysis()
    #test_optimizer()
    #test_optimizer_analysis()
    pass
