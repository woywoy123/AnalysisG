import torch
from AnalysisTopGNN.Tools import Threading
from AnalysisTopGNN.Notification import GraphGenerator
from AnalysisTopGNN.Samples import SampleTracer, Graphs
from AnalysisTopGNN.Tools import RandomSamplers
from AnalysisTopGNN.Generators import Settings

class GraphGenerator(GraphGenerator, SampleTracer, RandomSamplers):
    
    def __init__(self):
        
        self.Caller = "GRAPHGENERATOR"
        Settings.__init__(self)
        SampleTracer.__init__(self, self)


    def SetAttribute(self, c_name, fx, container):
        if c_name == "P_" or c_name == "T_":
            c_name += fx.__name__ 
        elif c_name == "":
            c_name += fx.__name__ 

        if c_name not in container:
            container[c_name] = fx
        else:
            self.Warning("Found Duplicate " + c_name + " Attribute")
   
    # Define the observable features
    def AddGraphFeature(self, fx, name = ""):
        self.SetAttribute(name, fx, self.GraphAttribute)

    def AddNodeFeature(self, fx, name = ""):
        self.SetAttribute(name, fx, self.NodeAttribute)

    def AddEdgeFeature(self, fx, name = ""):
        self.SetAttribute(name, fx, self.EdgeAttribute)

    
    # Define the truth features used for supervised learning 
    def AddGraphTruth(self, fx, name = ""):
        self.SetAttribute("T_" + name, fx, self.GraphAttribute)

    def AddNodeTruth(self, fx, name = ""):
        self.SetAttribute("T_" + name, fx, self.NodeAttribute)

    def AddEdgeTruth(self, fx, name = ""):
        self.SetAttribute("T_" + name, fx, self.EdgeAttribute)

    
    # Define any last minute changes to attributes before adding to graph
    def AddGraphPreprocessing(self, name, fx):
        self.SetAttribute("P_" + name, fx, self.GraphAttribute)

    def AddNodePreprocessing(self, name, fx):
        self.SetAttribute("P_" + name, fx, self.NodeAttribute)

    def AddEdgePreprocessing(self, name, fx):
        self.SetAttribute("P_" + name, fx, self.EdgeAttribute)

    def SetDevice(self):
        if self.Device:
            self.Device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def TestFeatures(self, SamplingSize = 100):
        self.SetDevice()
        self.CheckSettings()
        Events = self.RandomEvents(self.Tracer.Events, SamplingSize)
        self.TestEvent(Events, self.EventGraph)

    def AddSamples(self, Events, Tree):
        for ev in Events:
            evobj = Events[ev]
            if self.EventStart > ev:
                self.Tracer.Events[ev] = None
                continue
            if self.EventStop != None and self.EventStop < ev:
                self.Tracer.Events[ev] = None
                continue
            if evobj.Compiled:
                continue

            trees = {}
            if Tree == False:
                trees |= {tr : self.MakeGraph(evobj.Trees[tr], ev) for tr in evobj.Trees}
            else:
                trees |= {Tree : self.MakeGraph(evobj.Trees[Tree], ev)}
            self.Tracer.Events[ev].Trees = trees

    def CompileEventGraph(self):

        self.AddInfo("Name", [self.EventGraph.__qualname__])
        self.AddInfo("Path", [self.EventGraph.__module__])
        self.AddInfo("EventCode", [self.GetSourceFile(self.EventGraph)])   
        
        Features = {c_name : self.GetSourceCode(self.GraphAttribute[c_name]) for c_name in self.GraphAttribute}
        self.AddInfo("GraphFeatures", [Features])
        Features = {c_name : self.GetSourceCode(self.NodeAttribute[c_name]) for c_name in self.NodeAttribute}   
        self.AddInfo("NodeFeatures", [Features])
        Features = {c_name : self.GetSourceCode(self.EdgeAttribute[c_name]) for c_name in self.EdgeAttribute}
        self.AddInfo("EdgeFeatures", [Features])
         
        self.AddInfo("Device", [self.Device])
        self.AddInfo("SelfLoop", [self.SelfLoop])
        self.AddInfo("FullyConnect", [self.FullyConnect])

        if self._PullCode:
            return self

        self.SetDevice()
        self.CheckSettings()
        self.AddSamples(self.Tracer.Events, self.Tree)
        
        def function(inpt):
            return [i.MakeGraph() if i != None else True for i in inpt]

        events = list(self.Tracer.Events.values())
        self.Tracer.Events = {}
        TH = Threading(events, function, self.Threads, self.chnk)
        TH.VerboseLevel = self.VerboseLevel
        TH.Start()
        for i in TH._lists:
            if i == True:
                continue
            self.Tracer.Events[i.Filename] = i
        self.EdgeAttribute = {}
        self.GraphAttribute = {}
        self.NodeAttribute = {}
        self.MakeCache()
