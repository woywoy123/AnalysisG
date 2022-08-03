from AnalysisTopGNN.IO import WriteDirectory
from AnalysisTopGNN.Tools import Notification
from AnalysisTopGNN.Generators import Analysis
import inspect

class Condor(WriteDirectory, Notification):
    def __init__(self):
        WriteDirectory.__init__(self)
        Notification.__init__(self)
        self._Jobs = {}
        self._Time = {}
        self._Memory = {}
        self._wait = {}
        self._Complete = {}
        self._sequence = {}
        self.Hostname = None
        self.Password = None
        self.OutputDirectory = None 
        self.DisableEventCache = False
        self.DisableDataCache = False
        self.DisableRebuildTrainingSample = False
        self.Condor_Script = {}
        self._Source = {}
    
    def AddJob(self, name, instance, memory, time, waitfor = None):
        if name not in self._Jobs: 
            self._Jobs[name] = instance
        
        if name not in self._wait:
            self._wait[name] = []
        
        if waitfor != None:
            if isinstance(waitfor, str):
                self._wait[name].append(waitfor)
            elif isinstance(waitfor, list):
                self._wait[name] += waitfor 
    
        if name not in self._Memory:
            self._Memory[name] = memory
            self._Time[name] = time 
    
    def __Sequencer(self):
        def Recursion(inpt, key):
            dic = []
            dic.append(key)
            for i in inpt:
                dic += Recursion(self._wait[i], i)
            return dic
        
        for i in self._wait:
            out = Recursion(self._wait[i], i)
            new = []
            out.reverse()
            for k in out:
                if k in new:
                    continue
                new.append(k)
            self._sequence[i] = new
            self._Complete[i] = False

    def LocalDryRun(self):
        self.__Sequencer()
        for i in self._sequence:
            for j in self._sequence[i]:
                if self._Complete[j] == True:
                    continue
                if self.DisableEventCache == True:
                    self._Jobs[j].EventCache = False
                if self.DisableDataCache == True:
                    self._Jobs[j].DataCache = False
                if self.DisableRebuildTrainingSample == True:
                    self._Jobs[j].RebuildTrainingSample = False
                self._Jobs[j].Launch()
                self._Complete[j] = True

    def DumpCondorJobs(self):
        self.__Sequencer()
        self.MakeDir("CondorDump")
        for i in self._sequence:
            for j in self._sequence[i]:
                if self._Complete[j] == True:
                    continue
                Mem = self._Memory[j]
                clust_t = self._Time[j]
                Template = {}

                for k in self._Jobs[j].__dict__:
                    simple = False
                    obj = self._Jobs[j].__dict__[k]
                    obj_n = type(obj).__name__

                    if obj_n == "dict":
                        for p in obj:
                            if type(obj[p]).__name__ == "function":
                                Template[p + "_Functions"] = inspect.getsource(obj[p])
                                simple = None
                            else:
                                simple = True
                                break
                        if simple == False:
                           if len(obj) == 0:
                               simple = True

                    
                    if obj_n in ["str", "bool", "int", "NoneType", "list", "float"]:
                        simple = True 

                    if simple:
                        Template[k] = obj
                        continue
                    elif simple == None:
                        continue
                   
                    if type(obj).__name__ == "type":
                        Template[k] = inspect.getsource(obj)
                        continue
                    x = open(str(type(obj).__module__).replace(".", "/") + ".py", "r")
                    Template[k] = "".join(x.readlines())
                
                self.MakeDir("CondorDump/" + j)


                F = open("CondorDump/" + j + "/Functions.py", "w")
                F.write("")
                F.close()
                for k in Template:
                    if "_Functions" in k:
                        F = open("CondorDump/" + j + "/Functions.py", "a")
                        F.write(Template[k])
                        F.close()
                        continue
                    if "Model" == k and Template[k] != None:
                        F = open("CondorDump/" + j + "/Model.py", "w")
                        F.write(Template[k])
                        F.close()
                        continue
                    if "EventImplementation" == k and Template[k] != None:
                        F = open("CondorDump/" + j + "/EventImplementation.py", "w")
                        F.write(Template[k])
                        F.close()
                        continue
                    if "EventGraph" == k and Template[k] != None:
                        F = open("CondorDump/" + j + "/EventGraph.py", "w")
                        F.write(Template[k])
                        F.close()
                        continue 

                    if k.startswith("_"):
                        continue
                     
