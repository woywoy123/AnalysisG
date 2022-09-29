# Event Generator closure test
from AnalysisTopGNN.Generators import EventGenerator
import uproot
from AnalysisTopGNN.Particles import *
from AnalysisTopGNN.Events import Event
from DelphesEvent import Event as EventDelphes
from AnalysisTopGNN.IO import PickleObject, UnpickleObject
import importlib, inspect

def TestEvents(di):
    x = EventGenerator(di, Start = 0, Stop = -1)
    x.Event = EventDelphes
    x.SpawnEvents()
    x.Threads = 10
    x.chnk = 5
    #x.CompileEvent(SingleThread = False)
    return True

def TestSignalMultipleFile(di):
    
    ev = EventGenerator(di, Stop = 1000)
    ev.SpawnEvents()
    ev.Event = Event
    ev.Threads = 10
    ev.CompileEvent(SingleThread = False)
    
    for i in ev.Events:
        if i == 1000-1:
            print(i, ev.Events[i], ev.EventIndexFileLookup(i)) 
            return True

def TestSignalDirectory(di):
    
    ev = EventGenerator(di, Stop = 1000)
    ev.SpawnEvents()
    ev.Threads = 10
    ev.CompileEvent(SingleThread = False)
   
    c = 0
    Passed = False
    for i in ev.Events:
        if c == 1000:
            print(i, ev.Events[i], ev.EventIndexFileLookup(i)) 
            Passed = True
            c = 0
        c+=1
    return Passed

