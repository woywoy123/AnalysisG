from AnalysisTopGNN.Generators import Analysis
from AnalysisTopGNN.Events import Event
from AnalysisTopGNN.Events import EventGraphTruthJetLepton
from Templates.EventFeatureTemplate import ApplyFeatures

def TestRandomSampling(Files):

    Ana = Analysis()
    Ana.ProjectName = "RandomSampler"
    Ana.InputSample("SingleTop", Files[0])
    Ana.InputSample("4Tops", Files[1])
    Ana.Event = Event
    Ana.EventStop = 10
    Ana.EventCache = False
    Ana.DataCache = True
    Ana.DumpHDF5 = True
    Ana.EventGraph = EventGraphTruthJetLepton
    ApplyFeatures(Ana, "TruthJets")
    Ana.Launch()

    Ana = Analysis()
    Ana.ProjectName = "RandomSampler"
    Ana.InputSample("SingleTop")
    Ana.InputSample("4Tops")
    Ana.GenerateTrainingSample(90)

    Ana = Analysis()
    Ana.ProjectName = "RandomSampler"
    Ana.InputSample("SingleTop")
    Ana.InputSample("4Tops")
    Ana.GenerateSampleNodeDistributions()
    
    return True