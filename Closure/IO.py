# Closure Test Function 
from Functions.IO.Files import Directories
from Functions.IO.IO import File

di = "/CERN/Grid/SignalSamples"

def TestDir():  
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

dir_f = "/home/tnom6927/Downloads/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root/user.pgadow.24765302._000001.output.root"
def TestReadSingleFile():
    x = Directories(dir_f)
    x.GetFilesInDir()
    
    for i in x.Files:
        if i + "/" + x.Files[i][0] == dir_f:
            return True


def TestReadFile():
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

def TestFileConvertArray():
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
    
    x = File(dir_e + "/" + name)
    x.Trees += [Tree]
    x.Leaves += [Leaf]
    x.CheckKeys()
    x.ConvertToArray()
    
    passed = False
    for i in x.ArrayLeaves[Tree + "/" + Leaf]:
        assert isinstance((float(i[0][0])), float) == True
        passed = True
    return passed

