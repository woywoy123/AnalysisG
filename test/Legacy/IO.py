# Closure Test Function 
from AnalysisTopGNN.IO import Directories, File, HDF5

def TestDir(di):  
    x = Directories(di)
    x.ListDirs()
    x.GetFilesInDir()
    for i in x.Files:
        l = x.Files[i]
        if len(l) != 0:
            continue
        else: 
            return False
    return True

def TestReadSingleFile(dir_f):
    x = Directories(dir_f)
    x.GetFilesInDir()
    
    for i in x.Files:
        if i + "/" + x.Files[i][0] == dir_f:
            return True

def TestReadFile(di):
    x = Directories(di, Verbose = False)
    x.ListDirs()
    x.GetFilesInDir()
    
    dir_e = ""
    for i in x.Files:
        dir_e = i
        break
    name = x.Files[dir_e][0]

    Tree = "nominal"
    Leaf = "truth_top_child_e"
    fake_T = "XXXX"
    fake_B = "XXXX"
    fake_L = "XXXX"
    
    x = File(dir_e + "/" + name)
    x.Trees += [Tree, fake_T]
    x.Branches += [fake_B]
    x.Leaves += [Leaf, fake_L]
    
    x.CheckKeys()
    
    passed = False
    for i in x.ObjectTrees:
        assert i == Tree
        passed = True
    
    passed = False 
    for i in x.ObjectLeaves:
        assert i == Tree + "/" + Leaf
        passed = True
    
    return passed

def TestFileConvertArray(di):
    x = Directories(di, Verbose = False)
    x.ListDirs()
    x.GetFilesInDir()
    
    dir_e = ""
    for i in x.Files:
        dir_e = i
        break
    name = x.Files[dir_e][0]

    Tree = "nominal"
    Leaf = "truthjet_e"
    
    x = File(dir_e + "/" + name)
    x.Trees += [Tree]
    x.Leaves += [Leaf]
    x.CheckKeys()
    x.ConvertToArray()
    
    passed = False
    for i in x.ArrayLeaves[Tree + "/" + Leaf]:
        assert isinstance((float(i[0])), float) == True
        passed = True
    return passed

def TestHDF5ReadAndWriteParticle():
    from AnalysisTopGNN.Particles.Particles import Particle
    
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
    from GenericFunctions import CacheEventGenerator, CompareObjects
    from AnalysisTopGNN.IO import UnpickleObject, PickleObject  

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
