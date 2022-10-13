import torch
from AnalysisTopGNN.Tools import Threading
from AnalysisTopGNN.Notification import GraphGenerator
from AnalysisTopGNN.Samples import SampleTracer, Graphs
from AnalysisTopGNN.Tools import RandomSamplers

class GraphGenerator(GraphGenerator, SampleTracer, Graphs, RandomSamplers):
    
    def __init__(self):
        self.VerboseLevel = 3
        self.Caller = "GraphGenerator"
        self.Tracer = None
        self.EventStart = 0
        self.EventStop = None
        self.Threads = 12
        self.chnk = 1000
        Graphs.__init__(self)
   
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

    def ImportTracer(self, Inpt):
        if self.EventStart == 0:
            self.EventStart = Inpt.EventStart 
        if self.EventStop == None:
            self.EventStop = Inpt.EventStop

        self.VerboseLevel = Inpt.VerboseLevel 
        self.BeginTrace(Inpt)
        self.MakeCache()

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
            if self.EventStart > ev or ev > self.EventStop:
                self.Tracer.Events[ev] = None
                continue
            trees = {}
            if Tree == False:
                trees |= {tr : self.MakeGraph(evobj.Trees[tr], ev) for tr in evobj.Trees}
            else:
                trees |= {Tree : self.MakeGraph(evobj.Trees[Tree], ev)}
            self.Tracer.Events[ev].Trees = trees

    def CompileEventGraph(self):
        self.SetDevice()
        self.CheckSettings()
        self.AddInfo("Name", self.EventGraph.__class__)
        self.AddInfo("EventCode", self.GetSourceFile(self.EventGraph))   
        
        Features = {c_name : self.GetSourceCode(self.GraphAttribute[c_name]) for c_name in self.GraphAttribute}
        self.AddInfo("GraphFeatures", Features)
        Features = {c_name : self.GetSourceCode(self.NodeAttribute[c_name]) for c_name in self.NodeAttribute}   
        self.AddInfo("NodeFeatures", Features)
        Features = {c_name : self.GetSourceCode(self.EdgeAttribute[c_name]) for c_name in self.EdgeAttribute}
        self.AddInfo("EdgeFeatures", Features)
         
        self.AddInfo("Device", self.Device)
        self.AddInfo("SelfLoop", self.SelfLoop)
        self.AddInfo("FullyConnect", self.FullyConnect)
        self.AddSamples(self.Tracer.Events, self.Tree)
        
        def function(inpt):
            return [i.MakeGraph() if i != None else True for i in inpt]

        events = list(self.Tracer.Events.values())
        TH = Threading(events, function, 2, self.chnk)
        TH.VerboseLevel = self.VerboseLevel
        TH.Start()
        for i in TH._lists:
            if i == True:
                continue
            self.Tracer.Events[i.EventIndex] = i


