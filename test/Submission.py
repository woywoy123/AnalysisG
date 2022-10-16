from Analysis import Analysis
from AnalysisTopGNN.Events import EventGraphTruthTopChildren, Event


def TestAnalysis(GeneralDir):

    def Test(ev):
        return int(len(ev.TruthTops) == 4)

    def EventGen(Dir, Name):
        Ana = Analysis()
        Ana.InputSample(Name, Dir)
        Ana.EventCache = True
        Ana.Event = Event
        Ana.Threads = 10
        Ana.EventStop = 100
        Ana.DumpHDF5 = False
        Ana.DumpPickle = True
        Ana.ProjectName = "TMPProject"
        Ana.Launch()

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





    #EventGen(GeneralDir + "/t", "SingleTop")
    #EventGen(GeneralDir + "/ttbar", "ttbar")
    #EventGen(GeneralDir + "/tttt", "Signal")
    #EventGen(GeneralDir + "/Zmumu", "Zmumu")
    #EventGen([GeneralDir + "/t", GeneralDir + "/ttbar"], "Combined")
 

    DataGen("SingleTop")
    DataGen("ttbar")
    DataGen("Signal")
    DataGen("Zmumu")

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





