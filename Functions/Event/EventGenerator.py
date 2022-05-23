from Functions.IO.IO import File, Directories, PickleObject, UnpickleObject
from Functions.Tools.Alerting import Debugging
from Functions.Tools.DataTypes import TemplateThreading, Threading
from Functions.Event.Event import Event, EventVariables
import math

class EventGenerator(Debugging, EventVariables, Directories):
    def __init__(self, dir, Verbose = True, Start = 0, Stop = -1, Debug = False):
        Debugging.__init__(self, Threshold = Stop - Start)
        EventVariables.__init__(self)
        Directories.__init__(self, dir)
        self.Events = {}
        self.FileEventIndex = {}
        self.__Debug = Debug
        self.Threads = 12
        self.Caller = "EVENTGENERATOR"
        self.__Start = Start
        self.__Stop = Stop
        self.VerboseLevel = 1
        
    def SpawnEvents(self):
        self.GetFilesInDir()
        for i in self.Files:
            self.Notify("!_______NEW DIRECTORY______: " + str(i))
            for F in self.Files[i]:
                self.Events[i + "/" + F] = []
                F_i = File(i + "/" + F, self.__Debug)
                F_i.VerboseLevel = self.VerboseLevel
                F_i.Trees = self.MinimalTrees
                F_i.Leaves = self.MinimalLeaves
                F_i.CheckKeys()
                F_i.ConvertToArray()
                if self.__Debug:
                    PickleObject(F_i, "Debug.pkl")
                    F_i = UnpickleObject("Debug.pkl")
                
                self.Notify("!SPAWNING EVENTS IN FILE -> " + F)
                for l in range(len(F_i.ArrayLeaves[list(F_i.ArrayLeaves)[0]])):
                    pairs = {}
                    for tr in F_i.Trees:
                        
                        if self.__Start != 0:
                            if self.__Start <= l:
                                self.Count() 
                            else:
                                continue
                        else: 
                            self.Count()
                        
                        E = Event()
                        E.Debug = self.__Debug
                        E.Tree = tr
                        E.iter = l
                        E.ParticleProxy(F_i)
                        pairs[tr] = E

                    self.Events[i + "/" + F].append(pairs)
                    
                    if self.Stop():
                        self.ResetCounter()
                        break
                del F_i
                del F
                
        del self.MinimalLeaves
        del self.MinimalTrees

    def CompileEvent(self, SingleThread = False, ClearVal = True):
        
        def function(Entries):
            for k in Entries:
                for j in k:
                    k[j].CompileEvent(ClearVal = ClearVal)
            return Entries

        self.Caller = "EVENTCOMPILER"
        
        it = 0
        ev = {}
        for f in self.Events:
            self.Notify("!COMPILING EVENTS FROM FILE -> " + f)
            
            Events = self.Events[f]
            entries_percpu = math.ceil(len(Events) / (self.Threads))

            self.Batches = {}
            Thread = []
            for k in range(self.Threads):
                self.Batches[k] = [] 
                for i in Events[k*entries_percpu : (k+1)*entries_percpu]:
                    self.Batches[k].append(i)

                Thread.append(TemplateThreading(k, "", "Batches", self.Batches[k], function))
            th = Threading(Thread, self, self.Threads)
            th.Verbose = True
            if SingleThread:
                th.TestWorker()
            else:
                th.StartWorkers()
            del th
            del Thread

            self.Notify("!FINISHED COMPILING EVENTS FROM FILE -> " + f)
            self.Notify("!!SORTING INTO DICTIONARY -> " + f)
            
            self.FileEventIndex[f] = []
            self.FileEventIndex[f].append(it)
            for k in self.Batches:
                for j in self.Batches[k]:
                    ev[it] = j
                    it += 1
            self.FileEventIndex[f].append(it-1)
            del self.Batches

        self.Events = ev

    def EventIndexFileLookup(self, index):

        for i in self.FileEventIndex:
            min_ = self.FileEventIndex[i][0]
            max_ = self.FileEventIndex[i][1]

            if index >= min_ and index <= max_:
                return i


