from Functions.Tools.Alerting import Notification 
from Functions.IO.Files import WriteDirectory, Directories 
from Functions.Event.EventGenerator import EventGenerator 

class Unification(WriteDirectory, Directories, Notification):
    def __init__(self):
        WriteDirectory.__init__(self)
        Directories.__init__(self)
        Notification.__init__(self)
        
        # ==== Initial variables ==== #
        self.Caller = "Unification"
        self.VerboseLevel = 2
        self.Verbose = True 
        self.Name = "UNTITLED"
        self.OutputDir = None
        
        # ==== Event Generator variables ==== # 
        self.EventImplementation = None  
        self.InputSamples = None
        self.CPUThreads = 12
        self.NEvent_Start = 0
        self.NEvent_Stop = 1
        self.Cache = False
        self.__MainDirectory = None
        self.__SampleDir = {}
        
        # ==== TEMP CACHE VARIABLES ==== #
        
    def __BuildStructure(self):
        # Initialize the ROOT directory
        if self.OutputDir.endswith("/"):
            self.OutputDir = self.OutputDir[:-1]
        if self.Name.endswith("/"):
            self.OutputDir = self.OutputDir[:-1]

        self.__MainDirectory = self.OutputDir + "/" + self.Name
        self.ChangeDirToRoot(self.OutputDir)
        self.MakeDir(self.Name)

        # Make a directory for the EventGenerator files 
        if self.Cache == True:
            pass
    
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
        print(out)
        return out

    def __CheckSettings(self):
        if self.Name == "UNTITLED":
            self.Warning("NAME VARIABLE IS SET TO: " + self.Name)
        
        if self.OutputDir == None:
            self.Warning("NO OUTPUT DIRECTORY DEFINED. USING CURRENT DIRECTORY: \n" + self.pwd)
            self.OutputDir = self.pwd

        if len(self.__SampleDir) == 0:
            self.Fail("NO SAMPLES GIVEN. EXITING...")
        else:
            for k, l in self.__SampleDir.items():
                self.__CheckFiles(l)
    
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













