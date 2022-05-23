import uproot
import pickle
import h5py
from Functions.IO.Files import Directories, WriteDirectory
from Functions.Tools.DataTypes import Threading, TemplateThreading
from Functions.Tools.Alerting import Notification
from Functions.Tools.Variables import RecallObjectFromString

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
        self.Notify("!!STARTING CONVERSION")
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
        
        T = Threading(runners, self)
        T.Verbose = self.Verbose
        T.VerboseLevel = self.VerboseLevel
        T.StartWorkers()
        del T
        del runners
        del self.ObjectLeaves
        del self.ObjectBranches
        self.Trees = list(self.ObjectTrees)
        del self.ObjectTrees

def PickleObject(obj, filename, Dir = "_Pickle"):
    if filename.endswith(".pkl"):
        pass
    else:
        filename += ".pkl"
   
    d = WriteDirectory()
    d.MakeDir(Dir)
    if Dir.endswith("/"):
        filename = Dir + filename
    else:
        filename = Dir + "/" + filename

    outfile = open(filename, "wb")
    pickle.dump(obj, outfile)
    outfile.close()

def UnpickleObject(filename, Dir = "_Pickle"):
    if Dir.endswith("/"):
        Dir = Dir[0:len(Dir)-1]
    if filename.endswith(".pkl"):
        pass
    else:
        filename += ".pkl"

    filename = Dir + "/" + filename

    infile = open(filename, "rb")
    obj = pickle.load(infile)
    infile.close()
    return obj

class HDF5(WriteDirectory, Notification):

    def __init__(self, OutDir = "_Pickle", Name = "UNTITLED", Verbose = True):
        WriteDirectory.__init__(self)
        Notification.__init__(self, Verbose)

        self.VerboseLevel = 3
        self.Verbose = Verbose
        self.Caller = "WRITETOHDF5"
        self.__Outdir = OutDir
        self.__Name = Name
        self.__File = None
        self.MakeDir(self.__Outdir + "/" + Name)
        self.__FileDirName = self.__Outdir + "/" + self.__Name + "/" + self.__Name + ".hdf5"
  
    def StartFile(self, Mode = "w"):
        if Mode == "w":
            self.Notify("!!WRITING -> " + self.__FileDirName)    
        if Mode == "r":
            self.Notify("!!READING -> " + self.__FileDirName)
        self.__File = h5py.File(self.__FileDirName, Mode)
    
    def OpenFile(self, SourceDir, Name):
        self.__init__(OutDir = SourceDir, Name = Name, Verbose = self.Verbose)
        self.StartFile(Mode = "r")

    def EndFile(self):
        self.Notify("!!CLOSING FILE -> " + self.__FileDirName)
        self.__File.close()
        self.__File = None

    def DumpObject(self, obj, DirKey):
        def Recursion(dic, ki):
            if isinstance(dic, int):
                return dic

            if isinstance(dic, str):
                return dic

            if isinstance(dic, dict):
                v = [ki]
                v += [[i, Recursion(dic[i], i)] for i in dic]
                return v
            if isinstance(dic, list):
                v = [Recursion(i, i) for i in dic]
                return v

        
        self.__File[DirKey] = [str(type(obj))]
        ob = self.__File[DirKey]
        for key, val in obj.__dict__.items():
            if isinstance(val, dict) == isinstance(val, list):
                ob.attrs[key] = [val]
                continue
            elif isinstance(val, dict):
                print(Recursion(val, key)) # <----- Fix the recursion!!!
            elif isinstance(val, list):
                continue
            print("\n\n\n")
            print(val, key)
            print("\n\n\n")
   
    def RebuildObject(self, DirKey):

        obj = self.__File[DirKey]
        obj_type = str(obj[0].decode("utf-8")).split("'")[1]
        OBJ = RecallObjectFromString(obj_type)
        
        for i in list(obj.attrs):
            setattr(OBJ, i, obj.attrs[i][0])
        return OBJ
