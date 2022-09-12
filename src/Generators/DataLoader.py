from AnalysisTopGNN.Tools import Notification 
from AnalysisTopGNN.IO import ExportToDataScience
from AnalysisTopGNN.Tools import Threading
from AnalysisTopGNN.Parameters import Parameters

from sklearn.model_selection import ShuffleSplit
import numpy as np
import importlib
import torch
import time

class GenerateDataLoader(Notification, Parameters):
    
    def __init__(self):
        self.Caller = "GenerateDataLoader"
       
        self.Notification()
        self.Computations()
        self.GenerateDataLoader()
        self._iter = 0

        self.SetDevice(self.Device)

    def __SetAttribute(self, c_name, fx, container):
        if c_name not in container:
            container[c_name] = fx
        else:
            self.Warning("Found Duplicate " + c_name + " Attribute")
    
    # Define the observable features
    def AddGraphFeature(self, name, fx):
        self.__SetAttribute(name, fx, self.GraphAttribute)

    def AddNodeFeature(self, name, fx):
        self.__SetAttribute(name, fx, self.NodeAttribute)

    def AddEdgeFeature(self, name, fx):
        self.__SetAttribute(name, fx, self.EdgeAttribute)

    
    # Define the truth features used for supervised learning 
    def AddGraphTruth(self, name, fx):
        self.__SetAttribute("T_" + name, fx, self.GraphAttribute)

    def AddNodeTruth(self, name, fx):
        self.__SetAttribute("T_" + name, fx, self.NodeAttribute)

    def AddEdgeTruth(self, name, fx):
        self.__SetAttribute("T_" + name, fx, self.EdgeAttribute)

    
    # Define any last minute changes to attributes before adding to graph
    def AddGraphPreprocessing(self, name, fx):
        self.__SetAttribute("P_" + name, fx, self.GraphAttribute)

    def AddNodePreprocessing(self, name, fx):
        self.__SetAttribute("P_" + name, fx, self.NodeAttribute)

    def AddEdgePreprocessing(self, name, fx):
        self.__SetAttribute("P_" + name, fx, self.EdgeAttribute)


    def SetDevice(self, device, SampleList = None):
        if torch.cuda.is_available() and "cuda" in str(device):
            self.Device = device
        else:
            self.Device = "cpu"
        self.Device = torch.device(self.Device)
        
        if SampleList == None:
            SampleList = [self.DataContainer[i] for i in self.DataContainer]
        for i in SampleList:
            i.to(self.Device)

    def AddSample(self, EventGeneratorInstance, Tree, SelfLoop = False, FullyConnect = True):
        
        try:
            for i in EventGeneratorInstance.FileEventIndex:
                self.Notify("ADDING SAMPLE -> (" + Tree + ") " + i)
        except:
            self.Warning("FAILED TO ADD SAMPLE " + str(type(EventGeneratorInstance)))
       
        attrs = 0
        if len(list(self.EdgeAttribute)) == 0:
            self.Warning("NO EDGE FEATURES PROVIDED")
            attrs+=1
        if len(list(self.NodeAttribute)) == 0:
            self.Warning("NO NODE FEATURES PROVIDED")
            attrs+=1
        if len(list(self.GraphAttribute)) == 0:
            self.Warning("NO GRAPH FEATURES PROVIDED")
            attrs+=1
        if attrs == 3:
            self.Fail("NO ATTRIBUTES DEFINED!")
        
        self.Notify("!DATA BEING PROCESSED ON: " + str(self.Device))
        
        if self.EventGraph == "":
            self.Fail("EVENT GRAPH NOT DEFINED. Import an EventGraph implementation (See Functions/Event/Implementations)")
            return
        
        fx_m = self.EventGraph.__module__ 
        fx_n = self.EventGraph.__name__
        for it in sorted(EventGeneratorInstance.Events):
            try:
                ev = EventGeneratorInstance.Events[it][Tree]
            except KeyError:
                self.Fail("SUPPLIED TREE " +  Tree + " NOT FOUND. TREES IN EVENTGENERATOR; " + ",".join( EventGeneratorInstance.Events[it].keys()))
            if self._iter == self.NEvents:
                break
            
            fx = getattr(importlib.import_module(fx_m), fx_n)
            event = fx(ev)
            event.iter = self._iter
            event.SelfLoop = SelfLoop
            event.FullyConnect = FullyConnect
            event.EdgeAttr = self.EdgeAttribute
            event.NodeAttr = self.NodeAttribute
            event.GraphAttr = self.GraphAttribute
            self.DataContainer[self._iter] = event
            
            if self.CleanUp:
                EventGeneratorInstance.Events[it][Tree] = []

            if self._iter > 0 and EventGeneratorInstance.EventIndexFileLookup(it) != self.FileTraces["Samples"][-1] and self._iter -1 not in self.FileTraces["End"]:
                self.FileTraces["End"].append(self._iter-1)

            if len(self.FileTraces["Samples"]) == 0 or EventGeneratorInstance.EventIndexFileLookup(it) != self.FileTraces["Samples"][-1]:
                self.FileTraces["Samples"].append(EventGeneratorInstance.EventIndexFileLookup(it))
                self.FileTraces["Tree"].append(Tree)
                self.FileTraces["Start"].append(self._iter)
                self.FileTraces["Level"].append(fx_m + "." + fx_n)
                self.FileTraces["SelfLoop"].append(SelfLoop)
            self._iter += 1
        self.FileTraces["End"].append(self._iter-1)
        self.Notify("FINISHED CONVERSION")
    
    def ProcessSamples(self):
        # ==== Use Threading to Speed up conversion ==== # 
        def function(inpt):
            out = []
            for z in inpt:
                out.append(z.ConvertToData())
            return out
      
        if self.EventGraph == None:
            return 

        tmp = list(self.DataContainer.keys())
        val = list(self.DataContainer.values())
        tmp = [k for k,j in zip(tmp, val) if isinstance(j, self.EventGraph)]
        val = [k for k in val if isinstance(k, self.EventGraph)]
        
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
        if Directory == None:
            return SampleList
        
        if isinstance(SampleList, str):
            Exp = ExportToDataScience()
            Exp.VerboseLevel = 0
            dic = Exp.ImportEventGraph(SampleList, Directory)
            return dic[list(dic)[0]]
        elif isinstance(SampleList, list) == False: 
            self.Fail("WRONG SAMPLE INPUT! Expected list, got: " + type(SampleList))
        
        Exp = ExportToDataScience()
        Exp.VerboseLevel = 0
        Out = [] 
        for i in range(len(SampleList)):
            Out.append(Exp.ImportEventGraph(SampleList[i], Directory).popitem()[1])
        self.SetDevice(self.Device, Out)
        return Out
            


