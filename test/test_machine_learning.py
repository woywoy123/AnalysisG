from AnalysisG.Generators import RandomSamplers
from AnalysisG import Analysis
from AnalysisG.Events import Event, GraphChildren
from AnalysisG.Templates import ApplyFeatures
from AnalysisG.Templates import FeatureAnalysis

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
    train = {i+1 : [] for i in range(10)}
    for f in x:
        for smpl in x[f]["train"]: train[f] += smpl.i.tolist()
        for smpl in x[f]["leave-out"]: assert smpl.i.tolist()[0] not in train[f] 
    
    x = r.MakeDataLoader(smpls, SortByNodes = True)
    assert 12 in x
    assert 13 in x
    assert 14 in x

    x = r.MakeDataLoader(smpls)
    assert "All" in x
    assert len(x["All"]) == len(Ana)

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



if __name__ == "__main__":
    #test_random_sampling()
    #test_feature_analysis()
    pass
