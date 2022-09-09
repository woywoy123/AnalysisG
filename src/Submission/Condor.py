from AnalysisTopGNN.IO import WriteDirectory
from AnalysisTopGNN.Tools import Notification
from AnalysisTopGNN.Generators import Analysis
import inspect
import os, stat 

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
        self._Device = {}
        self.OutputDirectory = None 
        self.DisableEventCache = False
        self.DisableDataCache = False
        self.CondaEnv = "GNN"
        self.ProjectName = None
        self.Tree = None
    
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

        if self.ProjectName == None:
            self.Warning("INHERITING PROJECT NAME FROM INSTANCE!")
            self.ProjectName = instance.ProjectName 
        else:
            instance.ProjectName = self.ProjectName

        if self.OutputDirectory != None:
            instance.OutputDirectory = self.OutputDirectory

        if self.Tree != None:
            instance.Tree = self.Tree
    
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
                
                self.Notify("---------------------------")
                self.Notify("CURRENTLY RUNNING JOB: " + j)
                self._Jobs[j].Launch()
                self._Complete[j] = True

    def DumpCondorJobs(self):
        self.__Sequencer()
        self.MakeDir(self.ProjectName + "/CondorDump")
        DAG = []
        for i in self._sequence:
            self.MakeDir(self.ProjectName + "/CondorDump/" + i)
            for j in self._sequence[i]:
                configs = []
                configs += ["from AnalysisTopGNN.Generators import Analysis"]
                configs += ["Ana = Analysis()"]
                for k in self._Jobs[j].__dict__:
                    obj = self._Jobs[j].__dict__[k]
                    if k.startswith("_") and k != "_SampleDir":
                        continue
                    if k == "Event" and obj != None:
                        configs += ["from EventImplementation import *"]
                        configs += ["Ana.Event = " + obj.__name__]
                        F = open(self.ProjectName + "/CondorDump/" + i + "/EventImplementation.py", "w")
                        F.write("".join(open(inspect.getfile(obj), "r").readlines()))
                        F.close()

                    elif k == "Model" and obj != None:
                        configs += ["from ModelImplementation import *"]
                        configs += ["Ana.Model = " + type(obj).__name__ + "()"]
                        F = open(self.ProjectName + "/CondorDump/" + i + "/ModelImplementation.py", "w")
                        F.write("".join(open(type(obj).__module__.replace(".", "/") +".py", "r").readlines()))
                        F.close()

                    elif k == "EventGraph" and obj != None:
                        configs += ["from EventGraphImplementation import *"]
                        configs += ["Ana.EventGraph = " + obj.__name__]
                        F = open(self.ProjectName + "/CondorDump/" + i + "/EventGraphImplementation.py", "w")
                        F.write("".join(open(inspect.getfile(obj), "r").readlines()))
                        F.close()

                    elif k.endswith("Attribute"):
                        F = open(self.ProjectName + "/CondorDump/" + i + "/" + k +".py", "w")
                        configs += ["from " + k + " import *"]
                        for l in obj:
                            configs += ["Ana." + k + '["' + l + '"] = ' + obj[l].__name__]
                            F.write(inspect.getsource(obj[l]))
                        F.close()
                    else:
                        if k == "Device":
                            self._Device[j] = obj
                        if isinstance(obj, str):
                            obj = '"' + obj + '"'
                        configs += ["Ana." + k + " = " + str(obj)]
                
            configs += ["Ana.Launch()"]
            F = open(self.ProjectName + "/CondorDump/" + i + "/Spawn.py", "w")
            F.write("\n".join(configs))
            F.close()

            F = open(self.ProjectName + "/CondorDump/" + i + "/Spawn.sh", "w")
            sk = ["#!/bin/bash", "source ~/.bashrc", 'eval "$(conda shell.bash hook)"', "conda activate GNN", "python " + i + "/Spawn.py"]
            F.write("\n".join(sk))
            F.close()
            os.chmod(self.ProjectName + "/CondorDump/" + i + "/Spawn.sh", stat.S_IRWXU)
            
            sk = ["executable = " + i + "/Spawn.sh", "error = results.error.$(ClusterID)", 'Requirements = OpSysAndVer == "CentOS7"']
            if self._Device[i] == "cpu":
                sk += ["Request_CPUs = " + str(self._Jobs[i].__dict__["Threads"])]
            else:
                sk += ["Request_GPUs = " + str(1)]
 
            x = None
            clust_t = self._Time[i]
            if clust_t.endswith("h"):
                x = 60*60*float(clust_t.replace("h", ""))
            elif clust_t.endswith("m"):
                x = 60*float(clust_t.replace("m", ""))
            elif clust_t.endswith("s"):
                x = float(clust_t.replace("s", ""))
            if x != None:
                sk += ["+RequestRuntime = " + str(x)]
            
            x = None
            Mem = self._Memory[i]
            if Mem.endswith("GB"):
                x = 1024*float(Mem.replace("GB", ""))
            elif Mem.endswith("MB"):
                x = float(Mem.replace("MB", ""))
            if x != None:
                sk += ["+Request_Memory = " + str(x)]
            sk += ["queue"]

            F = open(self.ProjectName + "/CondorDump/" + i + "/" + i + ".submit", "w")
            F.write("\n".join(sk))
            F.close()
            s = "JOB " + i + " " + i + "/" + i + ".submit"
            if s not in DAG:
                DAG.append(s)
            
            for p in self._sequence[i]:
                if p == i:
                    continue
                s = "PARENT " + p + " CHILD " + i
                if s not in DAG:
                    DAG.append(s)
        F = open(self.ProjectName + "/CondorDump/DAGSUBMISSION.submit", "w")
        F.write("\n".join(DAG))
        F.close()
