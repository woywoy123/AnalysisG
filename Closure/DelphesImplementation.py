from Functions.Event.Implementations.EventDelphes import Event
from Functions.Event.EventGenerator import EventGenerator
def TestDelphes(FileDir):
    E = Event()
    ev = EventGenerator(FileDir, Stop = 2)
    ev.EventImplementation = E
    ev.SpawnEvents()
    ev.CompileEvent(True)

    for i in ev.Events:
        event = ev.Events[i]["Delphes"]
        if len(event.Particle) != 0:
            return True
    
    return False
