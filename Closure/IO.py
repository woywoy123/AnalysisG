# Closure Test Function 
from Functions.IO.IO import UpROOT_Reader
from Functions.IO.Files import Directories

di = "/CERN/Grid/SignalSamples"

def TestDir():  
    x = Directories(di)
    x.ListDirs()
    x.GetFilesInDir()

def TestIO():
    Tree = "nominal"
    Branch = "weight_mc"
    Leaves = "truth_top_child_e"
    x = UpROOT_Reader(di)
    x.DefineTrees(Tree)
    x.DefineBranches(Branch)
    x.DefineLeaves(Leaves)
    x.Read()


