import uproot
import pickle
from Functions.IO.Files import Directories
from Functions.Tools.DataTypes import DataTypeCheck, Threading, TemplateThreading
from Functions.Tools.Alerting import Notification

class File(Notification):
    def __init__(self, dir, name, Verbose = False):
        Notification.__init__(self, Verbose = Verbose) 
        self.Caller = "FILE +--> " + dir

        self.__Dir = dir
        self.FileName = name 
        
        self.ArrayBranches = {}
        self.ArrayLeaves = {}
        self.ArrayTrees = {}
        
        self.ObjectBranches = {}
        self.ObjectLeaves = {}
        self.ObjectTrees = {}

        self.Trees = []
        self.Leaves = []
        self.Branches = []

        self.__Reader = uproot.open(self.__Dir)
    
    def CheckKeys(self):

        def ReturnObject(Obj, i, j = -1, k = -1):
            out = ""
            if self.CheckObject(self.__Reader, i):
                out = Obj[i]

            if j != -1:
                if self.CheckObject(out, j):
                    out = Obj[i][j]
                else:
                    out = ""
            if k != -1:
                if self.CheckObject(out, k):
                    out = Obj[i][j][k]
                else:
                    out = ""
            return out
        
        for i in self.Trees:
            treeobj = ReturnObject(self.__Reader, i)
            if treeobj == "":
                self.Warning("SKIPPED TREE -> " + i)
                continue
            self.ObjectTrees[i] = treeobj
            for j in self.Branches:
                branchobj = ReturnObject(self.__Reader, i, j)
                if branchobj == "":
                    self.Warning("SKIPPED BRANCH -> " + j)
                    continue                
                self.ObjectBranches[i + "/" + j] = branchobj

                for k in self.Leaves:
                    leafobj = ReturnObject(self.__Reader, i, j, k) 
                    if leafobj == "":
                        self.Warning("SKIPPED LEAF -> " + k)
                        continue
                    self.ObjectLeaves[i + "/" + j + "/" + k] = leafobj

    def ConvertToArray(self):
        def Convert(obj):
            try:
                return obj.array(library = "np")
            except:
                return []
        
        self.Caller = "CONVERTTOARRAY"
        self.Notify("STARTING CONVERSION")
        runners = []
        for i in self.ObjectTrees:  
            if self.CheckAttribute(self.ObjectTrees[i], "array") and i not in self.ArrayTrees:
                th = TemplateThreading(i, "ObjectTrees", "ArrayTrees", self.ObjectTrees[i], Convert)
                runners.append(th)
        
        for i in self.ObjectBranches:
            if self.CheckAttribute(self.ObjectBranches[i], "array") and i not in self.ArrayBranches:
                th = TemplateThreading(i, "ObjectBranches", "ArrayBranches", self.ObjectBranches[i], Convert)
                runners.append(th)

        for i in self.ObjectLeaves:
            if self.CheckAttribute(self.ObjectLeaves[i], "array") and i not in self.ArrayLeaves:
                th = TemplateThreading(i, "ObjectLeaves", "ArrayLeaves", self.ObjectLeaves[i], Convert)
                runners.append(th)
        
        T = Threading(runners)
        T.StartWorkers()
        for i in T.Result:
            i.SetAttribute(self)



class UpROOT_Reader(Directories, DataTypeCheck):
    
    def __init__(self, dir, Verbose = False):
        Directories.__init__(self, dir, Verbose)
        self.Caller = "UpRootReader"
        
        DataTypeCheck.__init__(self)

        self.__Branches = []
        self.__Trees = []
        self.__Leaves = []

        self.FileObjects = {}
        
    def DefineBranches(self, Branches):
        self.__Branches += self.AddToList(Branches)

    def DefineTrees(self, Trees):
        self.__Trees += self.AddToList(Trees)

    def DefineLeaves(self, Leaves):
        self.__Leaves += self.AddToList(Leaves)
    
    def Read(self):
        self.GetFilesInDir()
        for i in self.Files:
            for j in self.Files[i]:
                r = i + "/" +j
                x = File(r, j, self.Verbose)
                x.Trees = self.__Trees
                x.Leaves = self.__Leaves
                x.Branches = self.__Branches
                self.FileObjects[r] = x   


def PickleObject(obj, filename):

    outfile = open(filename, "wb")
    pickle.dump(obj, outfile)
    outfile.close()

def UnpickleObject(filename):

    infile = open(filename, "rb")
    obj = pickle.load(infile)
    infile.close()
    return obj
