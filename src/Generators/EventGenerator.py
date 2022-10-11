#from AnalysisTopGNN.IO import File, Directories
#from AnalysisTopGNN.Tools import TemplateThreading, Threading, RecallObjectFromString

from AnalysisTopGNN.IO import File
from AnalysisTopGNN.Parameters import Parameters
from AnalysisTopGNN.Notification import EventGenerator
from AnalysisTopGNN.Samples import SampleTracer

class EventGenerator(EventGenerator, SampleTracer): #, Directories, Parameters):
    def __init__(self, InputDir = None, EventStart = 0, EventStop = None):
        self.Caller = "EVENTGENERATOR"
        self.InputDirectory = InputDir
        self.EventStart = EventStart
        self.EventStop = EventStop
        self.Event = None
        self.VerboseLevel = 3
        self.Threads = 12
   
    def __GetEvent(self):
        if "__init__" in self.Event.__dict__:
            self.Event = self.Event()
        _, evnt = self.GetObjectFromString(self.Event.__module__, type(self.Event).__name__)
        return evnt

    def __AddEvent(self, File, val = False):
        if val:
            EventObj = self.__GetEvent()
            EventObj._State = val
            EventObj.Tree = File._Tree
            EventObj.iter = self.Tracer.ROOTInfo[File.ROOTFile].EventIndex[EventObj.Tree]
            return self.Tracer.AddEvent(EventObj) 
        for i in File:
            if self.__AddEvent(File, i):
                return True

    def SpawnEvents(self):
        self.BeginTrace()
        self.CheckEventImplementation()

        Path = self.Event.__module__ + "." + self.Event.__name__
        self.AddInfo("Name", self.Event.__name__)
        self.AddInfo("Module", self.Event.__module__)
        self.AddInfo("Path", Path)
        self.AddInfo("EventCode", self.GetSourceCode(self.Event))
        obj = self.__GetEvent()
        
        self.Files = self.ListFilesInDir(self.InputDirectory, extension = ".root") 
        self.CheckROOTFiles() 
        
        for i in self.Files:
            self.AddSamples(i, self.Files[i])

        for F in self.DictToList(self.Files):
            F_i = File(F, self.Threads)
            F_i.Tracer = self.Tracer
            F_i.Trees += obj.Trees
            F_i.Branches += obj.Branches
            F_i.Leaves += obj.Leaves 
            F_i.ValidateKeys()
            for tr in F_i.Trees:
                F_i.GetTreeValues(tr)
                if self.__AddEvent(F_i):
                    return 

    def CompileEvent(self, SingleThread = False, ClearVal = True):
        
        def function(inp):
            out = []
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
            TH = Threading(blk[i], function, threads = self.Threads, chnk_size = self.chnk)
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


