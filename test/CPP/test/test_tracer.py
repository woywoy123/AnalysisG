from AnalysisG.Templates import EventTemplate
from AnalysisG.Templates import ParticleTemplate
from AnalysisG.IO import UpROOT
from AnalysisG.Tracer import SampleTracer
import shutil

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
        trees = x.__compiler__(i)

        tr.AddEvent(trees, root, index)         
        hashes |= {p.hash : p for p in trees}
        roothashes[root] += [p.hash for p in trees]

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


def EventMaker(root):
    ev = Event()
    ev.__interpret__
    io = UpROOT(root)
    io.Trees = ev.Trees
    io.Leaves = ev.Leaves
    out = []
    for i in io:
        root, index = i["ROOT"], i["EventIndex"]
        trees = ev.__compiler__(i)
        out.append([trees, root, index]) 
    return out

def test_tracer_operators():

    root1 = "./samples/sample1/smpl1.root"
    root2 = "./samples/sample1/smpl2.root"
    
    tr1 = SampleTracer()
    for i in EventMaker(root1): tr1.AddEvent(i[0], i[1], i[2])
    l1 = len(tr1)

    tr2 = SampleTracer()
    for i in EventMaker(root2): tr2.AddEvent(i[0], i[1], i[2])
    l2 = len(tr2)
     
    trsum = tr1 + tr2 
    lsum = len(trsum)    
    
    assert lsum == l2 + l1
    assert len(tr1) == l1
    assert len(tr2) == l2
    
    for i in tr1: assert trsum[i.hash]
    for i in tr2: assert trsum[i.hash]

    del tr1
    del tr2
    
    assert len([trsum.HashToROOT(i.hash) for i in trsum]) == lsum

def test_tracer_hdf5():
    root1 = "./samples/sample1/smpl1.root"
    root2 = "./samples/sample1/smpl2.root"
    
    tr1 = SampleTracer()
    for i in EventMaker(root1): tr1.AddEvent(i[0], i[1], i[2])
    l1 = len(tr1)

    tr2 = SampleTracer()
    for i in EventMaker(root2): tr2.AddEvent(i[0], i[1], i[2])
    l2 = len(tr2)

    #tr1.DumpEvents
    #tr2.DumpEvents

    for i in tr1:
        tr1[i.hash]
        #print(tr1[i.hash])
    del tr1
    del tr2
    from time import sleep
    sleep(1)
    print("....") 

    return 
    s = SampleTracer()
    s.EventCache = True
    s.RestoreEvents

    shutil.rmtree("EventCache")
    shutil.rmtree("Tracer")
    
    for i in s:
        pass
        #i.hash
        #print(s[i.hash])
    from time import sleep
    sleep(1)
    print("__")




if __name__ == "__main__":
    #test_tracer_addEvent()
    #test_tracer_operators()
    test_tracer_hdf5()
    


