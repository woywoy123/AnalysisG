from src.IO.IO import File 
import uproot
from DelphesEvent import Event
#from AnalysisTopGNN.Events import Event

def FromUproot(file, List = None):
    def Recursion(up, List):
        for i in range(len(List)):
            return Recursion(up[List[i]], List[i+1:])
        return up
    
    F = uproot.open(file, num_workers=12)
    return Recursion(F, List)


def TestReadROOTDelphes(file):
    #F = File(file)
    #F.Trees = ["Delphes"]
    #F.Branches = ["Particle"]
    #F.Leaves = ["Particle.PID", "Particle.Status", "Event.Weight"]
    #F.ValidateKeys()
    #F.GetTreeValues("Delphes")
    #out = F.Iter
    
    #print(len(out))


    #Ev = FromUproot(file, ["Delphes", "Particle", "Particle.PT"])
    #ev = len(Ev.iterate(library = "ak", step_size = 100000))
    #ev = list(ev)
    #print(len(ev[0]))

    e = Event()
    F = File(file, 12)

    d = 4
    print(e.Trees)
    print(e.Branches)
    print(e.Leaves[:d])

    F.Trees = e.Trees
    F.Leaves = e.Leaves
    F.Branches = e.Branches
    F.ValidateKeys()
    F.GetTreeValues("Delphes")
    
    x = []
    for i in F:
        x.append(i)
    
    print(len(x))



