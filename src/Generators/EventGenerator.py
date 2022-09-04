from AnalysisTopGNN.IO import File, Directories, PickleObject, UnpickleObject
from AnalysisTopGNN.Tools import Notification
from AnalysisTopGNN.Tools import TemplateThreading, Threading
from AnalysisTopGNN.Tools import RecallObjectFromString
from AnalysisTopGNN.Events import Event

class EventGenerator(Directories):
    def __init__(self, dir = None, Verbose = True, Start = 0, Stop = None):
        Notification.__init__(self)
        self._Dir = dir
        self.Events = {}
        self.FileEventIndex = {}
        self.Files = {}
        self.Threads = 12
        self.Caller = "EVENTGENERATOR"
        self._Start = Start
        self._Stop = Stop
        self.VerboseLevel = 1
        self.EventImplementation = Event()
       
        self.FileEventIndex["Start"] = []
        self.FileEventIndex["End"] = []

    def SpawnEvents(self):
        
        if "__init__" in self.EventImplementation.__dict__:
            self.EventImplementation = self.EventImplementation()
        name = type(self.EventImplementation).__module__ + "." + type(self.EventImplementation).__name__
        obj = RecallObjectFromString(name)
        if len(self.Files) == 0:
            self.GetFilesInDir()
        
        it_a = 0
        for i in self.Files:
            self.Notify("!_______NEW DIRECTORY______: " + str(i))

            for F in self.Files[i]:
                F_i = File(i + "/" + F)
                F_i._Threads = self.Threads
                F_i.Trees += obj.Trees
                F_i.Branches += obj.Branches
                F_i.Leaves += obj.Leaves 
                F_i.ValidateKeys()
                self.Notify("!SPAWNING EVENTS FROM FILE -> " + F)
                
                for tr in F_i.Trees:
                    
                    it = it_a
                    F_i.GetTreeValues(tr)
                    for t in F_i.Iter[self._Start:self._Stop]:
                        E = RecallObjectFromString(name)
                        E.Tree = tr
                        E.iter = it 
                        E._Store = t
                        BaseName = i + "/" + F + "/" + str(it)
                        if BaseName not in self.Events:
                            self.Events[BaseName] = {}
                        self.Events[BaseName] |= {tr : E}
                        it += 1
                it_a = it

        del self.EventImplementation
    
    def CompileEvent(self, SingleThread = False, ClearVal = True):
        
        def function(inp, out = []):
            for k in inp:
                k._Compile(ClearVal)
                out.append(k)
            return out
       
        if SingleThread:
            self.Threads = 1
        
        Events = {int(k.split("/")[-1]) : self.Events[k] for k in self.Events}
        FileNames = {}
        for i in self.Events:
            f_name = "/".join(i.split("/")[:-1])
            i_t = int(i.split("/")[-1])
            
            if f_name not in FileNames:
                FileNames[f_name] = [0, 0]
                FileNames[f_name][0] = i_t
            FileNames[f_name][1] = i_t
        self.FileEventIndex = FileNames

        blk = {}
        tmp = {}
        for i in Events:
            f = self.EventIndexFileLookup(i) 
            
            if f not in blk:
                blk[f] = []
                tmp[f] = []
            blk[f] += [Events[i][k] for k in Events[i]]
            tmp[f] += [[i, k] for k in Events[i]] 

        for i in blk:
            self.Notify("!COMPILING EVENTS FROM FILE: " + i)
            TH = Threading(blk[i], function, threads = self.Threads)
            TH.Start()
            for k in range(len(tmp[i])):
                p = tmp[i][k]
                it, br = p[0], p[1]
                Events[p[0]][p[1]] = TH._lists[k]
        self.Events = Events

    def EventIndexFileLookup(self, index):

        for i in self.FileEventIndex:
            min_ = self.FileEventIndex[i][0]
            max_ = self.FileEventIndex[i][1]

            if index >= min_ and index <= max_:
                return i


