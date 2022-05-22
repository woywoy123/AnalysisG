from Functions.Event.EventGenerator import EventGenerator
from Functions.IO.IO import PickleObject
from Functions.IO.Files import WriteDirectory, Directories 
import os

def EventGenerator_Template(di, Compiler, Stop = -1, SingleThread = False):
    ev = EventGenerator(di, Stop = Stop)
    ev.SpawnEvents()
    ev.CompileEvent(SingleThread = SingleThread)
    PickleObject(ev, Compiler)

def BuildCacheDirectory(Dir = "_Cache", Name = "EventGenerator"):
    if Dir == "_Cache":
        mkdir = WriteDirectory()
        mkdir.MakeDir(Dir + "/" + Name)
    else:
        d = Dir + "/" + Name + "/"
        k = ""
        for i in d.split("/"):
            k += "/" + i
            
            try:
                os.mkdir(k)
            except FileExistsError:
                pass
            except PermissionError:
                pass
        os.chdir(d)
    return True


def Generate_Cache(di, Stop = -1, SingleThread = False, Compiler = "EventGenerator", Outdir = "_Cache"):
    ev = EventGenerator(di, Stop = Stop)
    ev.SpawnEvents()
    ev.CompileEvent(SingleThread = SingleThread)
    PickleObject(ev, Compiler, Outdir + "/" + Compiler)

