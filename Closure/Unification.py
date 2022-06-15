from Functions.Unification.Unification import Unification 
from Functions.Event.Implementations.Event import Event 

def TestUnification(FileDir, Files):
    #U = Unification()
    #U.EventCache = True
    #for key, Dir in Files.items():
    #    U.InputSample(key, Dir)
    #U.Launch()
    TestUnificationDataLoader()
    return True

def TestUnificationDataLoader():
    import Functions.FeatureTemplates.ParticleGeneric.EdgeFeature as ef
    import Functions.FeatureTemplates.ParticleGeneric.NodeFeature as nf
    import Functions.FeatureTemplates.ParticleGeneric.GraphFeature as gf
    from Functions.Event.Implementations.EventGraphs import EventGraphTruthTops
    
    def Test(a):
        return float(a.Test)

    #U = Unification()
    #U.DataCache = True
    #U.EventGraph = EventGraphTruthTops
    #U.AddGraphFeature("mu", gf.mu)
    #U.AddGraphTruth("mu_actual", gf.mu_actual)
    #U.AddNodeFeature("Test", Test)
    #U.Launch()
    TestUnificationOptimizer() 
    return True

def TestUnificationOptimizer():
    U = Unification()
    U.Launch()
