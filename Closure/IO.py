# Closure Test Function 
from Functions.IO.Files import Directories
from Functions.IO.IO import File


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

