from AnalysisTopGNN.Generators import EventGenerator
from AnalysisTopGNN.Events import Event
from AnalysisTopGNN.IO import PickleObject, UnpickleObject, WriteDirectory, Directories 
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
    ev.Event = Event
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
            f = f.split("/")[-1]
            Generate_Cache(di + f, Stop, SingleThread, f.replace(".root", ""), OutDirectory + Compiler)
    
    dic = []
    target = OutDirectory + Compiler
    for f in Directories(target).ListFilesInDir(target):
        dic.append(f)
    return dic



