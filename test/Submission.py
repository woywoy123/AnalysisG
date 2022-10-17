from Analysis import Analysis
from AnalysisTopGNN.Events import EventGraphTruthTopChildren, Event


def TestAnalysis(GeneralDir):

    def Test(ev):
        return int(len(ev.TruthTops) == 4)

    def EventGen(Dir, Name):
        Ana = Analysis()
        Ana.ProjectName = "TMPProject"
        Ana.InputSample(Name, Dir)
        Ana.EventCache = True
        Ana.Event = Event
        Ana.Threads = 1
        Ana.EventStop = 10
        Ana.DumpHDF5 = False
        Ana.DumpPickle = True
        Ana.Launch()
        return Ana

    def DataGen(Name):
        Ana = Analysis()
        Ana.ProjectName = "TMPProject"
        Ana.InputSample(Name)
        Ana.DataCache = True
        Ana.Event = Event
        Ana.EventGraph = EventGraphTruthTopChildren
        Ana.AddGraphFeature(Test)
        Ana.Threads = 10
        Ana.EventStop = 100
        Ana.DumpHDF5 = True
        Ana.DumpPickle = True
        Ana.Launch()
        return Ana




    
    ev = EventGen(GeneralDir + "/t", "SingleTop")
    ev += EventGen(GeneralDir + "/ttbar", "ttbar")
    #ev += EventGen(GeneralDir + "/tttt", "Signal")
    #ev += EventGen(GeneralDir + "/Zmumu", "Zmumu")
    #ev += EventGen([GeneralDir + "/t", GeneralDir + "/ttbar"], "Combined")
    
    print(ev._HashCache)

    for i in ev:
        print(i)

























    #DataGen("SingleTop")
    #DataGen("ttbar")
    #DataGen("Signal")
    #DataGen("Zmumu")

    #Ana = Analysis()
    #Ana.InputSample("SingleTop")
    #Ana.InputSample("ttbar")
    #Ana.InputSample("Signal")
    #Ana.InputSample("Zmumu")
    #Ana.EventCache = False
    #Ana.DataCache = False
    #Ana.DumpHDF5 = False
    #Ana.MergeSamples = True
    #Ana.GenerateTrainingSample = True
    #Ana.Threads = 10
    #Ana.ProjectName = "TMPProject"
    #Ana.Model = GraphNN()
    #Ana.Launch()





