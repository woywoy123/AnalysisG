from multiprocessing import Process, Pipe
import numpy as np

class DataTypeCheck:

    def __init__(self):
        pass
    
    def AddToList(self, obj):
        output = []
        if isinstance(obj, str):
            output.append(obj)
        if isinstance(obj, list):
            output += obj
        return output
    
    def DictToList(self, dic):
        Out = []
        for i in dic:
            if isinstance(dic[i], list):
                Out += self.AddToList(dic[i])
        return Out
    
class Threading:
    def __init__(self, lists, threads = 32):
        self.__threads = threads
        self.__lists = lists
        self.Result = []

    def StartWorkers(self):
        
        Processes = []
        self.Result = []
        
        sub_p = []
        res = []
        for i in range(len(self.__lists)):
            recv, send = Pipe(False)
            P = Process(target = self.__lists[i].Runner, args=(send,i))
            Processes.append(P)
            sub_p.append(recv) 

            P.start()

            if len(sub_p) == self.__threads:
                for p in sub_p:
                    re = p.recv()
                    res.append(re)
                sub_p = []
        
        for p in sub_p:
            re = p.recv()
            res.append(re)
        
        for i in range(len(self.__lists)):
            for j in res:
                if i in j:
                    self.__lists[i].SetResults(j[i])
        
        for p in Processes:
            p.join()
        
        self.Result = self.__lists

    def TestWorker(self):
        for i in range(len(self.__lists)):
            self.__lists[i].TestRun()
        self.Result = self.__lists



class TemplateThreading:
    def __init__(self, name, source_name, target_name, source_value, function):
        self.__name = name 
        self.__source_name = source_name
        self.__target_name = target_name
        self.__source_value = source_value
        self.__function = function

    def Runner(self, q, index):
        self.__result = self.__function(self.__source_value)
        out = {}
        out[index] = self.__result
        q.send(out)
    
    def TestRun(self):
        self.__result = self.__function(self.__source_value)
    
    def SetResults(self, res):
        self.__result = res

    def SetAttribute(self, obj):
        j = getattr(obj, self.__target_name)
        j[self.__name] = self.__result
        setattr(obj, self.__target_name, j)

