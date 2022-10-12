#from AnalysisTopGNN.Tools import Notification 
#from AnalysisTopGNN.IO import ExportToDataScience
#from AnalysisTopGNN.Tools import Threading
#from AnalysisTopGNN.Parameters import Parameters

#from sklearn.model_selection import ShuffleSplit
#import numpy as np
#import importlib
import torch
#import time

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

        print(Events)

    def CompileEventGraph(self):
        self.SetDevice()
        self.CheckSettings()
        self.AddInfo("Name", self.EventGraph.__class__)
        self.AddInfo("EventCode", self.GetSourceFile(self.EventGraph))   
        
        Features = {c_name : self.GetSourceCode(self.GraphAttribute[c_nam]) for c_name in self.GraphAttribute}
        self.AddInfo("GraphFeatures", Features)
        Features = {c_name : self.GetSourceCode(self.NodeAttribute[c_nam]) for c_name in self.NodeAttribute}   
        self.AddInfo("NodeFeatures", Features)
        Features = {c_name : self.GetSourceCode(self.EdgeAttribute[c_nam]) for c_name in self.EdgeAttribute}
        self.AddInfo("EdgeFeatures", Features)
         
        self.AddInfo("Device", self.Device) 





        #fx_m = self.EventGraph.__module__ 
        #fx_n = self.EventGraph.__name__
        #for it in sorted(EventGeneratorInstance.Events):
        #    try:
        #        ev = EventGeneratorInstance.Events[it][Tree]
        #    except KeyError:
        #        self.Fail("SUPPLIED TREE " +  Tree + " NOT FOUND. TREES IN EVENTGENERATOR; " + ",".join( EventGeneratorInstance.Events[it].keys()))
        #    if self._iter == self.NEvents:
        #        break
        #    
        #    fx = getattr(importlib.import_module(fx_m), fx_n)
        #    event = fx(ev)
        #    event.iter = self._iter
        #    event.SelfLoop = SelfLoop
        #    event.FullyConnect = FullyConnect
        #    event.EdgeAttr = self.EdgeAttribute
        #    event.NodeAttr = self.NodeAttribute
        #    event.GraphAttr = self.GraphAttribute
        #    self.DataContainer[self._iter] = event

        #    if self.CleanUp:
        #        EventGeneratorInstance.Events[it][Tree] = []

        #    if self._iter > 0 and EventGeneratorInstance.EventIndexFileLookup(it) != self.FileTraces["Samples"][-1] and self._iter -1 not in self.FileTraces["End"]:
        #        self.FileTraces["End"].append(self._iter-1)

        #    if len(self.FileTraces["Samples"]) == 0 or EventGeneratorInstance.EventIndexFileLookup(it) != self.FileTraces["Samples"][-1]:
        #        self.FileTraces["Samples"].append(EventGeneratorInstance.EventIndexFileLookup(it))
        #        self.FileTraces["Tree"].append(Tree)
        #        self.FileTraces["Start"].append(self._iter)
        #        self.FileTraces["Level"].append(fx_m + "." + fx_n)
        #        self.FileTraces["SelfLoop"].append(SelfLoop)
        #    self._iter += 1
        #self.FileTraces["End"].append(self._iter-1)
        #self.Notify("FINISHED CONVERSION")
    
    def ProcessSamples(self):
        # ==== Use Threading to Speed up conversion ==== # 
        def function(inpt):
            out = []
            for z in inpt:
                out.append(z.ConvertToData())
                del z
            return out
      
        if self.EventGraph == None:
            return 

        tmp = list(self.DataContainer.keys())
        val = list(self.DataContainer.values())
        tmp = [k for k,j in zip(tmp, val) if isinstance(j, self.EventGraph)]
        val = [k for k in val if isinstance(k, self.EventGraph)]
        if len(val) == 0:
            return 

        TH = Threading(val, function, self.Threads, self.chnk)
        TH.Start()
        for j, i in zip(tmp, TH._lists):
            self.DataContainer[j] = i
            i.to(device = self.Device, non_blocking = True)
        del TH

    def MakeTrainingSample(self, ValidationSize = None):
        def MakeSample(Shuff, InputList):
            if isinstance(Shuff, int):
                Shuff = self.DataContainer
            for i in Shuff:
                n_p = int(self.DataContainer[i].num_nodes)
                if n_p == 0:
                    continue
                if n_p not in InputList:
                    InputList[n_p] = []
                InputList[n_p].append(self.DataContainer[i])
        
        if ValidationSize != None:
            self.ValidationSize = ValidationSize

        self.ProcessSamples() 
        self.Notify("!WILL SPLIT DATA INTO TRAINING/VALIDATION (" + 
                str(self.ValidationSize) + "%) - TEST (" + str(100 - self.ValidationSize) + "%) SAMPLES")

        All = np.array(list(self.DataContainer))
        
        if self.ValidationSize > 0 and self.ValidationSize < 100:
            rs = ShuffleSplit(n_splits = 1, test_size = float((100-self.ValidationSize)/100), random_state = 42)
            for train_idx, test_idx in rs.split(All):
                pass
            MakeSample(train_idx, self.TrainingSample)
            MakeSample(test_idx, self.ValidationSample)
        else:
            MakeSample(-1, self.TrainingSample)

    def RecallFromCache(self, SampleList, Directory):

        def function(inpt):
            exp = ExportToDataScience()
            exp.VerboseLevel = 0
            out = []
            for i in inpt:
                out += list(exp.ImportEventGraph(i, self.DataCacheDir).values())
            return out

        if Directory == None:
            return SampleList
        
        if isinstance(SampleList, str):
            Exp = ExportToDataScience()
            Exp.VerboseLevel = 0
            dic = Exp.ImportEventGraph(SampleList, Directory)
            return dic[list(dic)[0]]
        elif isinstance(SampleList, list) == False: 
            self.Fail("WRONG SAMPLE INPUT! Expected list, got: " + type(SampleList))
        
        TH = Threading(SampleList, function, self.Threads, self.chnk)
        TH.VerboseLevel = self.VerboseLevel
        TH.Caller = self.Caller
        TH.Start()
        self.SetDevice(self.Device, TH._lists)
        return TH._lists
            


