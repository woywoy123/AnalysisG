from Functions.Event.EventGenerator import EventGenerator
from Functions.IO.IO import PickleObject, UnpickleObject
from Functions.IO.Files import WriteDirectory, Directories 
import os

def BuildCacheDirectory(Dir = "_Cache", Name = "EventGenerator", rootDir = False):
    d = Dir + "/" + Name
    if rootDir:
        d = "/" + d
    mkdir = WriteDirectory()
    mkdir.MakeDir(d)
    return True

def Generate_Cache(di, Stop = -1, SingleThread = False, Compiler = "EventGenerator", Outdir = "_Cache"):
    ev = EventGenerator(di, Stop = Stop)
    ev.SpawnEvents()
    ev.CompileEvent(SingleThread = SingleThread)
    PickleObject(ev, Compiler, Outdir + "/")

def Generate_Cache_Batches(di, Stop = -1, SingleThread = False, Compiler = "EventGenerator", OutDirectory = "_Cache", CreateCache = True):
    if di.endswith("/") == False:
        di += "/"
    if OutDirectory.endswith("/") == False:
        OutDirectory += "/"
    
    BuildCacheDirectory(OutDirectory, Compiler)
   
    if ".root" in di and CreateCache:
        Generate_Cache(di[:-1], Stop, SingleThread, di[:-1].split("/")[-1].replace(".root", ""), OutDirectory + Compiler)
    elif CreateCache:
        Files = Directories(di).ListFilesInDir(di)
        for f in Files:
            Generate_Cache(di + f, Stop, SingleThread, f.replace(".root", ""), OutDirectory + Compiler)
    
    dic = []
    target = OutDirectory + Compiler
    for f in Directories(target).ListFilesInDir(target):
        dic.append(target + "/" + f)
    return dic



