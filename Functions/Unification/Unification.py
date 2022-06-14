from Functions.Tools.Alerting import Notification 
from Functions.IO.Files import WriteDirectory, Directories 
from Functions.IO.IO import PickleObject, UnpickleObject
from Functions.Event.EventGenerator import EventGenerator 
from difflib import SequenceMatcher

class Unification(WriteDirectory, Directories, Notification):
    def __init__(self):
        WriteDirectory.__init__(self)
        Directories.__init__(self)
        Notification.__init__(self)
        
        # ==== Initial variables ==== #
        self.Caller = "Unification"
        self.VerboseLevel = 2
        self.Verbose = True 
        self.ProjectName = "UNTITLED"
        self.OutputDir = None
        
        # ==== Event Generator variables ==== # 
        self.EventImplementation = None  
        self.InputSamples = None
        self.CompileSingleThread = False
        self.CPUThreads = 12
        self.NEvent_Start = 0
        self.NEvent_Stop = 1
        self.Cache = False
        self.CacheDir = None
        self.__MainDirectory = None
        self.__SampleDir = {}
        
        # ==== TEMP CACHE VARIABLES ==== #
        self._TMP = None

    def __CheckSettings(self):
        if self.ProjectName == "UNTITLED":
            self.Warning("NAME VARIABLE IS SET TO: " + self.ProjectName)
        
        if self.OutputDir == None:
            self.Warning("NO OUTPUT DIRECTORY DEFINED. USING CURRENT DIRECTORY: \n" + self.pwd)
            self.OutputDir = self.pwd

        if len(self.__SampleDir) == 0:
            self.Fail("NO SAMPLES GIVEN. EXITING...")
        else:
            self._TMP = {}
            for k, l in self.__SampleDir.items():
                self._TMP[k] = self.__CheckFiles(l)

    def __BuildStructure(self):
        # Initialize the ROOT directory
        if self.OutputDir.endswith("/"):
            self.OutputDir = self.OutputDir[:-1]
        if self.ProjectName.endswith("/"):
            self.OutputDir = self.OutputDir[:-1]

        self.__MainDirectory = self.OutputDir + "/" + self.ProjectName
        self.ChangeDirToRoot(self.OutputDir)
        self.MakeDir(self.ProjectName)
        self.ChangeDirToRoot(self.__MainDirectory)
        self.__EventGeneratorDir = "EventGenerator"

        # Make a directory for the EventGenerator files 
        if self.Cache == True and self.CacheDir == None:
            Filter = []
            for name in self._TMP:
                Filter += [j for i in list(self._TMP[name]) for j in i.split("/")]
            
            cur = "" 
            for i in range(len(Filter)):
                cur = "/".join(Filter[:i])
                accept = False
                for name in self._TMP:
                    if cur == "/".join(list(self._TMP[name])[0].split("/")[:i]):
                        accept = True 
                    else:
                        accept = False
                        break

                if accept == False:  
                    cur = "/".join(Filter[:i-1]) + "/" 
                    break
            
            for name in self._TMP:
                for FileDir in self._TMP[name]:
                    self.MakeDir(self.__EventGeneratorDir + "/" + name + "/" + FileDir.replace(cur, ""))
                    self.Notify("STARTING THE COMPILATION OF NEW DIRECTORY: " + FileDir)
                    for S in self._TMP[name][FileDir]:
                        self.__StartEventGenerator({FileDir : [S]}, S.replace(".root", ""), self.__EventGeneratorDir + "/" + name + "/" + FileDir.replace(cur, ""))
    
    def __CheckFiles(self, Directory):
        out = {}
        for i in Directory:
            val = self.ListFilesInDir(i)
            for p in val: 
                f, key  = p.split("/")[-1], "/".join(p.split("/")[:-1])
                if key not in out:
                    out[key] = []
                if f not in out[key]:
                    out[key].append(f)
        return out
    
    def __StartEventGenerator(self, SampleDict, Name, OutDir): 
        ev = EventGenerator(None, self.Verbose, self.NEvent_Start, self.NEvent_Stop)
        ev.Files = SampleDict
        ev.VerboseLevel = 0
        ev.Threads = self.CPUThreads 
        ev.SpawnEvents()
        ev.CompileEvent(self.CompileSingleThread)
        PickleObject(ev, Name, OutDir)
    
   
    def AddSample(self, Name,  Directory):
        if isinstance(Name, str) == False:
            self.Warning("NAME NOT A STRING!")
            return 
        if Name not in self.__SampleDir:
            self.__SampleDir[Name] = []

        if isinstance(Directory, list):
            self.__SampleDir[Name] += Directory 

        elif isinstance(Directory, str):
            self.__SampleDir[Name].append(Directory)

        else:
            self.Warning("INPUT DIRECTORY NOT VALID (STRING OR LIST)!")
            return 

    def Launch(self):
        self.__CheckSettings()
        self.__BuildStructure()













