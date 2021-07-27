# Event Generator closure test
from Functions.Event.Event import EventGenerator

dir = "/CERN/Grid/SignalSamples"
def TestEvent():
    x = EventGenerator(dir)
    x.SpawnEvents()
