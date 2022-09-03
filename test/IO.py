from src.IO.IO import File

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
    print(F.Leaves)
    assert sorted(F.Leaves) == sorted(["Delphes/Particle/Particle.M1", "Delphes/Event/Event.X1", "Delphes/Event_size"])
    return True
