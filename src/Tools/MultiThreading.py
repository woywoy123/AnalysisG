import multiprocessing
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")
from torch.multiprocessing import Process, Pipe
from AnalysisTopGNN.Tools import Notification
import math
import gc

class TemplateThreading:

    def __init__(self, function, val, index):
        self._v = val
        self._f = function
        self._i = index
        self.i = None
    
    def Runner(self, q):
        _r = self._f(self._v) 
        for i in self._v:
            del i
        try:
            q.send(_r)
            for i in _r:
                del i 
        except:
            q.send(False)

    def MainThread(self):
        r = self._f(self._v) 
        for i in self._v:
            del i
        return r


class Threading(Notification):
    def __init__(self, lists, Function, threads = 12, chnk_size = None):
        self._threads = threads
        self._lists = lists
        self._function = Function
        Notification.__init__(self)
        self.Caller = "MULTITHREADING"
        
        if chnk_size != None:
            _quant = int(len(lists)/chnk_size)+1
        else:
            _quant = self._threads
        
        cpu_ev = math.ceil(len(self._lists) / _quant)
        self._chnk = [self._lists[i : i+cpu_ev] for i in range(0, len(self._lists), cpu_ev)]
        self._indx = [[i, i + cpu_ev] for i in range(0, len(self._lists), cpu_ev)]
        self._lists = [i for i in range(len(self._lists))]
      
    def Start(self):
        it = 1
        tmp = 0
        for i in range(len(self._chnk)):
            recv, send = Pipe(False)

            T = TemplateThreading(self._function, self._chnk[i], self._indx[i])
            T.i = i
            P = Process(target = T.Runner, args = (send, ))
            self._chnk[i] = [P, recv, T]

            P.start()
            self.Notify("!!STARTED JOB " + str(i+1) + "/" + str(len(self._chnk)))
            
            if i+1 == self._threads*it:
                self.CheckJobs(tmp, i+1)
                tmp = i+1
                it+=1
        
        if i+1 < self._threads*it:
            self.CheckJobs(tmp, i+1)
        gc.collect()

    def CheckJobs(self, start, end):
        w = 1
        for t in range(start, end):
            i = self._chnk[t]
            try:
                out = i[1].recv()
            except:
                i[0].terminate()
                out = False
            indx = i[2]._i

            if out == False:
                out = i[2].MainThread()
            for j, d in zip(range(indx[0], indx[1]), out):
                self._lists[j] = d

            del self._chnk[t][2]
            del self._chnk[t][1]
            del self._chnk[t][0]
            self._chnk[t] = None
            self.Notify("!!WORKER FINISHED " + str(w) + "/" + str(self._threads))
            w+=1
