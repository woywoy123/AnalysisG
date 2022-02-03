import uproot
import pickle
from Functions.IO.Files import Directories
from Functions.Tools.DataTypes import Threading, TemplateThreading
from Functions.Tools.Alerting import Notification

class File(Notification):
    def __init__(self, dir, Verbose = False):
        Notification.__init__(self, Verbose = Verbose) 
        self.Caller = "FILE +--> " + dir

        self.__Dir = dir
        
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
  
    def CheckObject(self, Object, Key):
        if Key == -1:
            return False
        try:
            Object[Key]
            return True
        except uproot.exceptions.KeyInFileError:
            return False

    def ReturnObject(self, i, j = -1, k = -1):
        out = self.__Reader
        if self.CheckObject(out, i):
            self.ObjectTrees[i] = self.__Reader[i]
        else:
            self.Warning("SKIPPED TREE -> " + i)
            return None
        
        if self.CheckObject(out[i], j):
            self.ObjectBranches[i + "/" + j] = self.__Reader[i][j]
        elif j != -1: 
            self.Warning("SKIPPED BRANCH -> " + j)

        if self.CheckObject(out[i], k):
            if j != -1:
                self.ObjectLeaves[i + "/" + j + "/" + k] = self.__Reader[i][j][k]
            else:
                self.ObjectLeaves[i + "/" + k] = self.__Reader[i][k]
        elif k != -1: 
            self.Warning("SKIPPED LEAF -> " + k)

    def CheckKeys(self):

        for i in self.Trees:
            self.ReturnObject(i)
        
        for i in self.ObjectTrees:
            for j in self.Branches:
                self.ReturnObject(i, j)

        for i in self.ObjectTrees:
            for j in self.Leaves:
                self.ReturnObject(i, -1, j)

        for i in self.ObjectBranches:
            for j in self.Leaves:
                tr = i.split("/")[0]
                br = i.split("/")[1]
                self.ReturnObject(tr, br, j) 

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
        del T


def PickleObject(obj, filename):

    outfile = open(filename, "wb")
    pickle.dump(obj, outfile)
    outfile.close()

def UnpickleObject(filename):

    infile = open(filename, "rb")
    obj = pickle.load(infile)
    infile.close()
    return obj
