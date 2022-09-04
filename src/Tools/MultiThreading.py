import multiprocessing
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")

from multiprocessing import Process, Pipe
from AnalysisTopGNN.Tools import Notification
import math


done = multiprocessing.Event()
class TemplateThreading:

    def __init__(self, function, val, index):
        self._v = val
        self._f = function
        self._i = index
        self._r = None
    
    def Runner(self, q):
        self._r = self._f(self._v) 
        q.send(self._r)
        done.wait()

class Threading(Notification):
    def __init__(self, lists, Function, threads = 12):
        self._threads = threads
        self._lists = lists
        self._function = Function
        Notification.__init__(self)
        self.Verbose = True
        self.Caller = "MULTITHREADING"

        cpu_ev = math.ceil(len(self._lists) / self._threads)
        self._chnk = [self._lists[i : i+cpu_ev] for i in range(0, len(self._lists), cpu_ev)]
        self._indx = [[i, i + cpu_ev] for i in range(0, len(self._lists), cpu_ev)]
        self._lists = [None for i in self._lists]

    def Start(self):
        for i in range(len(self._chnk)):
            recv, send = Pipe(False)

            T = TemplateThreading(self._function, self._chnk[i], self._indx[i])
            P = Process(target = T.Runner, args = (send, ))
            P.start()
            
            self._chnk[i] = [P, recv, T]
            self.Notify("!!STARTING WORKER " + str(i+1) + "/" + str(self._threads))
       
        for t in range(len(self._chnk)):
            i = self._chnk[t]
            out = i[1].recv()
            i[0].terminate()
            indx = i[2]._i
            for j in range(len(out)):
                self._lists[j+indx[0]] = out[j]
            self.Notify("!!WORKER FINISHED " + str(t+1) + "/" + str(self._threads))
