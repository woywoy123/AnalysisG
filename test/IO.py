from src.IO.IO import File, HDF5, UnpickleObject, PickleObject
from GenericFunctions import CacheEventGenerator, CompareObjects
from AnalysisTopGNN.Particles.Particles import Particle

def TestReadROOTNominal(file):
    F = File(file)
    F.Trees = ["nominal", "lol", "Test", "nominal"] 
    F.Branches = ["test", "test2", "jet_map_Gtops"]
    F.Leaves = ["el_e"]
    F._Threads = 12
    F.ValidateKeys()
    
    assert F.Trees == ["nominal"]
    assert F.Branches == ["nominal/jet_map_Gtops"]
    assert F.Leaves == ["nominal/el_e"]
    return True

def TestReadROOTDelphes(file):
    F = File(file)
    F.Trees = ["nominal", "lol", "Test", "nominal", "Delphes"] 
    F.Branches = ["test", "test2", "jet_map_Gtops", "Event", "Particle"]
    F.Leaves = ["el_e", "Event.X1", "Event_size", "Particle.M1"]
    F.ValidateKeys()

    assert F.Trees == ["Delphes"]
    assert sorted(F.Branches) == sorted(["Delphes/Event", "Delphes/Particle"])
    assert sorted(F.Leaves) == sorted(["Delphes/Particle/Particle.M1", "Delphes/Event/Event.X1", "Delphes/Event_size"])
    return True

def TestHDF5ReadAndWriteParticle():
    
    X = Particle()
    Y = Particle()
    Z = Particle()
    
    P = Particle()
    P.DataDict = {"Test" : 0}
    P.DictList = {"Test" : [1, 2]}
    P.DictListParticles = {"Test" : [X, Y], "Test2" : [Y, Z]}

    H = HDF5(Name = "ClosureTestHDF5")
    H.StartFile()
    H.DumpObject(P)
    H.EndFile()

    H.OpenFile(Name = "ClosureTestHDF5")
    obj = H.RebuildObject()
    for i in obj:
        obj = obj[i]
        break
   
    assert len(obj.__dict__) == len(P.__dict__)

    for i, j in zip(obj.__dict__, P.__dict__):
        a_val, b_val = obj.__dict__[i], P.__dict__[j]
        if i == "DictListParticles":
            continue
        assert a_val == b_val
    return True

def TestHDF5ReadAndWriteEvent(di, Cache):
    ev = CacheEventGenerator(1, di, "TestHDF5ReadAndWriteEvent", Cache)
    event = ev.Events[0]["nominal"]
    PickleObject(event, "TestHDF5ReadAndWriteEventObject")
    event = UnpickleObject("TestHDF5ReadAndWriteEventObject")

    f = HDF5(Name = "TestHDF5ReadAndWriteEvent")
    f.StartFile()
    f.DumpObject(event)
    f.EndFile()
    
    x = HDF5()
    x.OpenFile(Name = "TestHDF5ReadAndWriteEvent")
    ev = x.RebuildObject()
    for i in ev:
        def Apply(ins, attr, ino):
            setattr(ins, attr, getattr(ino, attr))
        
        Apply(event, "DetectorParticles", ev[i])
        Apply(event, "Electrons", ev[i])
        Apply(event, "Muons", ev[i])
        Apply(event, "Jets", ev[i])
        Apply(event, "TruthJets", ev[i])
        Apply(event, "TruthTops", ev[i])
        Apply(event, "TruthTopChildren", ev[i])
        Apply(event, "TopPreFSR", ev[i])
        Apply(event, "TopPostFSR", ev[i])
        Apply(event, "TopPostFSRChildren", ev[i])
        ev = ev[i]
    CompareObjects(event, ev)

    return True
