from AnalysisTopGNN.IO import WriteDirectory
from AnalysisTopGNN.Tools import Notification

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
            print("-> " + i) 
            for j in self._sequence[i]:
                if self._Complete[j] == True:
                    continue
                if self.DisableEventCache == True:
                    self._Jobs[j].EventCache = False
                if self.DisableDataCache == True:
                    self._Jobs[j].DataCache = False
                print("-> " + j)
                self._Jobs[j].Launch()
                self._Complete[j] = True
