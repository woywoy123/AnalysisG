from AnalysisTopGNN.Tools import Tools
from AnalysisTopGNN.Notification import Condor
from AnalysisTopGNN.Generators import Settings
import os, stat 

class Condor(Tools, Condor):
    def __init__(self):
        Settings.__init__(self)
        self._Jobs = {}
        self._Time = {}
        self._Memory = {}
        self._wait = {}
        self._Complete = {}
        self._sequence = {}
        self._Device = {}
        
        self.SkipEventCache = False
        self.SkipDataCache = False
        self.CondaEnv = "GNN"
        
        self.Caller = "CONDOR"
    
    def AddJob(self, name, instance, memory, time, waitfor = None):
        if name not in self._Jobs: 
            self._Jobs[name] = instance
        
        self.AddListToDict(self._wait, name)
        
        if waitfor == None:
            pass
        elif isinstance(waitfor, str):
            self._wait[name].append(waitfor)
        elif isinstance(waitfor, list):
            self._wait[name] += waitfor 
    
        if name not in self._Memory:
            self._Memory[name] = memory
            self._Time[name] = time

        self.ProjectInheritance(instance)

        if self.OutputDirectory:
            instance.OutputDirectory = self.OutputDirectory

        if self.Tree:
            instance.Tree = self.Tree
        
        if self.VerboseLevel != 3:
            instance.VerboseLevel = self.VerboseLevel
    
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
                if self.SkipEventCache == True:
                    self._Jobs[j].EventCache = False
                if self.SkipDataCache == True:
                    self._Jobs[j].DataCache = False
                
                self.RunningJob(j)
                
                self._Jobs[j].Launch()
                self._Complete[j] = True
                del self._Jobs[j]
                self._Jobs[j] = None
        self.FinishExit()

    def DumpCondorJobs(self):
        self.__Sequencer()
        self.mkdir(self.ProjectName + "/CondorDump")
        DAG = []
        for i in self._sequence:
            self.mkdir(self.ProjectName + "/CondorDump/" + i)
            for j in self._sequence[i]:

                if self.SkipDataCache:
                    self._Jobs[i].DataCache = False
                if self.SkipEventCache:
                    self._Jobs[i].EventCache = False

                configs = []
                configs += ["from AnalysisTopGNN.Generators import Analysis"]
                
                self._Jobs[j]._PullCode = True
                self._Jobs[j].Launch()

                Setting = self._Jobs[j].Settings.DumpSettings(self._Jobs[j])
                trace = Setting["Tracer"].EventInfo
                self._Jobs[j]._PullCode = False
                
                if "EVENTGENERATOR" in trace:
                    EventCode = ["from AnalysisTopGNN.Templates import EventTemplate \n"]
                    EventCode += ["from ParticleCode import * \n\n"]
                    EventCode += trace["EVENTGENERATOR"]["EventCode"]
                    
                    F = open(self.ProjectName + "/CondorDump/" + i + "/EventImplementation.py", "w")
                    F.write("".join(EventCode))
                    F.close()
                  
                    F = open(self.ProjectName + "/CondorDump/" + i + "/ParticleCode.py", "w")
                    F.write(trace["EVENTGENERATOR"]["ParticleCode"][0])
                    F.close()

                    configs += ["from EventImplementation import *\n\n"]
                    configs += ["Ana = Analysis()"]
                    configs += ["Ana.Event = " + trace["EVENTGENERATOR"]["Name"][-1]]
                
                if "GRAPHGENERATOR" in trace:

                    EventCode = ["from AnalysisTopGNN.Templates import EventGraphTemplate \n\n"]
                    EventCode += trace["GRAPHGENERATOR"]["EventCode"]
                    
                    F = open(self.ProjectName + "/CondorDump/" + i + "/EventGraphImplementation.py", "w")
                    F.write("".join(EventCode))
                    F.close()
                  
                    
                    Features = {"GraphFeatures" : {}, "NodeFeatures" : {}, "EdgeFeatures" : {}}
                    for f in Features:
                        if f not in trace["GRAPHGENERATOR"]:
                            continue

                        F = open(self.ProjectName + "/CondorDump/" + i + "/" + f + ".py", "w")
                        for t in trace["GRAPHGENERATOR"][f]:
                            F.write("\n".join(list(t.values())))

                            f_names = {str(k) : str(t.split(":")[0].split("(")[0].split(" ")[1]) for k, t in zip(t, t.values())}
                            Features[f] |= f_names
                        F.close()
                        configs += ["from " + f + " import *"]

                    configs += ["from EventGraphImplementation import *\n\n"]

                    configs += ["Ana = Analysis()"]
                    configs += ["Ana.EventGraph = " + trace["GRAPHGENERATOR"]["Name"][-1]]

                    for f in Features:
                        configs += ["Ana." + f.replace("Features", "Attribute") + " = {" + ",".join([" '" + p + "' : " + k for p, k in zip(Features[f], Features[f].values())]) + "}" ]
                
                for s in Setting:
                    if s in ["Tracer", "_PullCode"]:
                        continue
                    if s == "OutputDirectory":
                        Setting[s] = self.RemoveTrailing(Setting[s], "/")
                        Setting[s] = "'" + Setting[s] + "'"
                    if s == "ProjectName":
                        Setting[s] = "'" + Setting[s] + "'"

                    if s == "Tree":
                        Setting[s] = "'" + Setting[s] + "'"
                    
                    configs += ["Ana." + s + " = " + str(Setting[s])]
                
                for s in self._Jobs[j]._SampleMap:
                    if len(self._Jobs[j]._SampleMap[s]) == 0:
                        configs += ["Ana.InputSample('" + s + "')"]
                    else:
                        configs += ["Ana.InputSample('" + s + "', " + str(self._Jobs[j]._SampleMap[s]) + ")"]
                
                
            configs += ["Ana.Launch()"]
            F = open(self.ProjectName + "/CondorDump/" + i + "/main.py", "w")
            F.write("\n".join(configs))
            F.close()

            F = open(self.ProjectName + "/CondorDump/" + i + "/Spawn.sh", "w")
            sk = ["#!/bin/bash", "source ~/.bashrc", 'eval "$(conda shell.bash hook)"', "conda activate " + self.CondaEnv, "python " + i + "/main.py"]
            F.write("\n".join(sk))
            F.close()
            os.chmod(self.ProjectName + "/CondorDump/" + i + "/Spawn.sh", stat.S_IRWXU)
            
            sk = ["executable = " + i + "/Spawn.sh", "error = " + i + "/results.error.$(ClusterID)", 'Requirements = OpSysAndVer == "CentOS7"']
            if i not in self._Device:
                self._Device[i] = "cpu"
            if self._Device[i] == "cpu":
                sk += ["Request_Cpus = " + str(self._Jobs[i].__dict__["Threads"])]
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
                sk += ["Request_Memory = " + str(x)]
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
        
        F = open(self.ProjectName + "/CondorDump/RunAsBash.sh", "w")
        F.write("#!/bin/bash\n")
        for j in DAG:
            if j.startswith("JOB") == False:
                continue
            x = j.split(" ")[-1]
            x = x.split("/")[0]
            x += "/Spawn.sh\n"
            F.write("bash " + x)
        F.close()
