from Functions.Tools.Alerting import Notification 
from Functions.Event.EventGraph import *
from sklearn.model_selection import ShuffleSplit
import numpy as np
import importlib
           
class GenerateDataLoader(Notification):
    
    def __init__(self):
        self.Verbose = True
        Notification.__init__(self, self.Verbose)
        self.Device_S = "cpu"
        self.__iter = 0
        self.NEvents = -1
        self.CleanUp = True
        self.Caller = "GenerateDataLoader"
        
        self.GraphAttribute = {}
        self.NodeAttribute = {}
        self.EdgeAttribute = {}

        self.DataContainer = {}
        self.TrainingSample = {}
        self.ValidationSample = {}
        
        self.FileTraces = {}
        self.FileTraces["Tree"] = []
        self.FileTraces["Start"] = []
        self.FileTraces["End"] = []
        self.FileTraces["Level"] = []
        self.FileTraces["SelfLoop"] = []
        self.FileTraces["Samples"] = []

        self.SetDevice(self.Device_S)

    def __SetAttribute(self, c_name, fx, container):
        if c_name not in container:
            container[c_name] = fx
        else:
            self.Warning("Found Duplicate " + c_name + " Attribute")

    def AddGraphFeature(self, name, fx):
        self.__SetAttribute(name, fx, self.GraphAttribute)

    def AddGraphTruth(self, name, fx):
        self.__SetAttribute("T_" + name, fx, self.GraphAttribute)

    def AddNodeFeature(self, name, fx):
        self.__SetAttribute(name, fx, self.NodeAttribute)

    def AddNodeTruth(self, name, fx):
        self.__SetAttribute("T_" + name, fx, self.NodeAttribute)

    def AddEdgeFeature(self, name, fx):
        self.__SetAttribute(name, fx, self.EdgeAttribute)

    def AddEdgeTruth(self, name, fx):
        self.__SetAttribute("T_" + name, fx, self.EdgeAttribute)

    def SetDevice(self, device):
        self.Device_S = device
        self.Device = torch.device(device)

        for i in self.DataContainer:
            self.DataContainer[i].to(self.Device)

    def AddSample(self, EventGeneratorInstance, Tree, Level, SelfLoop = False, FullyConnect = True):
        
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

        self.Notify("!DATA BEING PROCESSED ON: " + self.Device_S)
        self.len = len(EventGeneratorInstance.Events)

        if Level == "TruthTops":
            fx_s = EventGraphTruthTops
        elif Level == "TruthTopChildren":
            fx_s = EventGraphTruthTopChildren
        elif Level == "TruthJetLepton":
            fx_s = EventGraphTruthJetLepton
        elif Level == "DetectorParticles":
            fx_s = EventGraphDetector
        else:
            self.Fail("EVENT GRAPH NOT DEFINED. See EventGraph.py")
            return
        fx_m = fx_s.__module__ 
        fx_n = fx_s.__name__
        for it in sorted(EventGeneratorInstance.Events):
            ev = EventGeneratorInstance.Events[it][Tree]
            self.ProgressInformation("CONVERSION")
            if self.__iter == self.NEvents:
                break

            fx = getattr(importlib.import_module(fx_m), fx_n)
            event = fx(ev)
            event.iter = self.__iter
            event.SelfLoop = SelfLoop
            event.FullyConnect = FullyConnect
            event.EdgeAttr = self.EdgeAttribute
            event.NodeAttr = self.NodeAttribute
            event.GraphAttr = self.GraphAttribute
            DataObject = event.ConvertToData()
            DataObject.to(device = self.Device, non_blocking = True)
            self.DataContainer[self.__iter] = DataObject
            
            if self.CleanUp:
                EventGeneratorInstance.Events[it][Tree] = []

            if self.__iter > 0 and EventGeneratorInstance.EventIndexFileLookup(it) != self.FileTraces["Samples"][-1] and self.__iter -1 not in self.FileTraces["End"]:
                self.FileTraces["End"].append(self.__iter-1)

            if len(self.FileTraces["Samples"]) == 0 or EventGeneratorInstance.EventIndexFileLookup(it) != self.FileTraces["Samples"][-1]:
                self.FileTraces["Samples"].append(EventGeneratorInstance.EventIndexFileLookup(it))
                self.FileTraces["Tree"].append(Tree)
                self.FileTraces["Start"].append(self.__iter)
                self.FileTraces["Level"].append(Level)
                self.FileTraces["SelfLoop"].append(SelfLoop)

            self.__iter += 1
        self.FileTraces["End"].append(self.__iter-1)
        self.Notify("FINISHED CONVERSION")
        self.ResetAll()

    def MakeTrainingSample(self, ValidationSize = 50):
        def MakeSample(Shuff, InputList):
            if isinstance(Shuff, int):
                Shuff = self.DataContainer
            for i in Shuff:
                n_p = int(self.DataContainer[i].num_nodes)
                if n_p not in InputList:
                    InputList[n_p] = []
                InputList[n_p].append(self.DataContainer[i])


        self.Notify("!WILL SPLIT DATA INTO TRAINING/VALIDATION (" + 
                str(ValidationSize) + "%) - TEST (" + str(100 - ValidationSize) + "%) SAMPLES")

        All = np.array(list(self.DataContainer))
        
        if ValidationSize > 0 and ValidationSize < 100:
            rs = ShuffleSplit(n_splits = 1, test_size = float((100-ValidationSize)/100), random_state = 42)
            for train_idx, test_idx in rs.split(All):
                pass
            MakeSample(train_idx, self.TrainingSample)
            MakeSample(test_idx, self.ValidationSample)
        else:
            MakeSample(-1, self.TrainingSample)

