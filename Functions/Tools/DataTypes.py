from multiprocessing import Process, Pipe
from Functions.Tools.Alerting import Notification

class Threading(Notification):
    def __init__(self, lists, obj, threads = 12):
        self.__threads = threads
        self.__lists = lists
        Notification.__init__(self)
        self.Verbose = True
        self.Caller = "MULTITHREADING"
        self.Object = obj

    def StartWorkers(self):
        
        self.Notify("!!STARTING " + str(len(self.__lists)) + " WORKERS")
        
        sub_p = []
        it = 0
        for i in range(len(self.__lists)):
            recv, send = Pipe(False)
            P = Process(target = self.__lists[i].Runner, args=(send,i))
            sub_p.append(recv) 

            P.start()

            if len(sub_p) == self.__threads:
                for p in sub_p:
                    re = p.recv()
                    it += 1
                    for j in re:
                        self.__lists[j].SetAttribute(self.Object, re[j])
                        self.__lists[j] = 0
                    self.Notify("!!!PROGRESS " + str(round(100*float(it) / float(len(self.__lists)), 2)) + "% COMPLETE")    
                    del re
                    del p
                sub_p = []

        for p in sub_p:
            re = p.recv()
            it += 1
            for j in re:
                self.__lists[j].SetAttribute(self.Object, re[j])
            self.Notify("!!!PROGRESS " + str(round(100*float(it) / float(len(self.__lists)), 2)) + "% COMPLETE")      
            del p
            del re 

    def TestWorker(self):
        for i in range(len(self.__lists)):
            self.__lists[i].SetAttribute(self.Object, self.__lists[i].TestRun())


class TemplateThreading:
    def __init__(self, name, source_name, target_name, source_value, function):
        self.__name = name 
        self.__source_name = source_name
        self.__target_name = target_name
        self.__source_value = source_value
        self.__function = function

    def Runner(self, q, index):
        out = {}
        out[index] = self.__function(self.__source_value)        
        del self.__source_value
        del self.__source_name
        del self.__function
        q.send(out)
    
    def TestRun(self):
        return self.__function(self.__source_value)
    
    def SetAttribute(self, obj, result):
        j = getattr(obj, self.__target_name)
        j[self.__name] = result
        setattr(obj, self.__target_name, j)

