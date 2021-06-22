import multiprocessing
from BaseFunctions.Alerting import Alerting

class Threading:
    def __init__(self, threads = 6, verb = False):
        self.__threads = threads
        self.__verbose = verb

    def MultiThreading(self, events, Compiler, Output):
        def Running(Runs, Sender, compiler):
            Out = compiler(Runs)
            Sender.send(Out)

        batch_S = round(float(len(events)) / float(self.__threads))+1
        Pipe = []
        Prz = []
        
        Bundle = []
        for i in events:
            Bundle.append(i)

            if len(Bundle) == batch_S:
                recv, send = multiprocessing.Pipe(False)
                P = multiprocessing.Process(target = Running, args = (Bundle, send, Compiler,))
                Pipe.append(recv)
                Prz.append(P)
                P.start()
                Bundle = []

        if len(Bundle) != 0:
            recv, send = multiprocessing.Pipe(False)
            P = multiprocessing.Process(target = Running, args = (Bundle, send, Compiler,))
            Pipe.append(recv)
            Prz.append(P)
            P.start()

        al = Alerting(Prz, self.__verbose)
        al.Notice("COMPILING EVENTS")
        
        for i, j in zip(Prz, Pipe):
            con = j.recv()
            i.join()
            for t in con:
                Output[t] = con[t]
            al.ProgressAlert()
        al.Notice("COMPILING COMPLETE")

