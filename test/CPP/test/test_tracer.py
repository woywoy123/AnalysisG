from AnalysisG.Templates import EventTemplate
from AnalysisG.Templates import ParticleTemplate
from AnalysisG.IO import UpROOT
    
class Particle(ParticleTemplate):
    def __init__(self):
        ParticleTemplate.__init__(self)
        self.pt = self.Type + "_pt"
        self.eta = self.Type + "_eta"
        self.phi = self.Type + "_phi"
        self.e = self.Type + "_e"

class Top(Particle):
    
    def __init__(self):
        self.Type = "top"
        Particle.__init__(self) 
         
class Children(Particle):
    
    def __init__(self):
        self.Type = "children"
        Particle.__init__(self) 
         
class Event(EventTemplate):
    def __init__(self):
        EventTemplate.__init__(self)
        self.Objects = {
                "top" : Top, 
                "Children" : Children()
        }
        self.index = "eventNumber"
        self.weight = "weight_mc"
        self.Trees = ["nominal", "truth"]
        self.met_phi = "met_phi"
        self.CommitHash = "..."
        self.Deprecated = False

def test_tracer_addEvent(): 
    from AnalysisG.Tracer import SampleTracer

    tr = SampleTracer()

    root1 = "./samples/sample1/smpl1.root"
    root2 = "./samples/sample1/smpl2.root"
    ev = Event()
    ev.__interpret__
    
    io = UpROOT([root1, root2])
    io.Trees = ev.Trees
    io.Leaves = ev.Leaves
    
    hashes = {}
    roothashes = {root1 : [], root2 : []}
    for i in io:
        x = ev.clone
        root, index = i["ROOT"], i["EventIndex"]
        x.__compiler__(i)

        tr.AddEvent(x, root, index)         
        hashes |= {p.hash : p for p in x.Trees}
        roothashes[root] += [p.hash for p in x.Trees]

    assert len(io)*2 == len(tr)
    
    # Test iterator 
    for i in tr:
        assert i == hashes[i.hash]
    
    # Test Getter Functions 
    for i in hashes:
        if hashes[i].Tree != "nominal": continue
        assert tr[i] == hashes[i]
    
    # Test if hashes are in tracer
    for i in hashes:
        assert i in tr

    assert "hello" not in tr
    assert root1 in tr
    
    for r in roothashes:
        for hsh in roothashes[r]:
            assert r == tr.HashToROOT(hsh)
    lst = [i for j in range(1000) for i in hashes ]
    assert sum(tr.FastHashSearch(lst).values()) == len(hashes)


if __name__ == "__main__":
    test_tracer_addEvent()
